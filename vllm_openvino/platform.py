# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum

import vllm_openvino.envs as envs

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


_MODEL_PRESETS: dict[str, dict[str, int]] = {
    "gemma-4": {"min_kv_cache_gb": 32},
    "gemma4": {"min_kv_cache_gb": 32},
    "qwen3.5": {"min_kv_cache_gb": 16},
    "qwen3_5": {"min_kv_cache_gb": 16},
    "qwen-3.5": {"min_kv_cache_gb": 16},
}


def _inspect_model(model_path: str) -> dict:
    result = {"is_stateful": False, "preset": None}
    path_lower = model_path.lower()
    for key, preset in _MODEL_PRESETS.items():
        if key in path_lower:
            result["preset"] = preset
            break
    if ov is None:
        return result
    from pathlib import Path
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return result
    ir_path = model_dir / "openvino_language_model.xml"
    if not ir_path.exists():
        ir_path = model_dir / "openvino_model.xml"
    if not ir_path.exists():
        return result
    try:
        ov_model = ov.Core().read_model(str(ir_path))
        has_readvalue = any(
            op.get_type_name() == "ReadValue" for op in ov_model.get_ops()
        )
        if has_readvalue:
            result["is_stateful"] = True
        else:
            has_sdpa = any(
                op.get_type_name() == "ScaledDotProductAttention"
                for op in ov_model.get_ops()
            )
            result["is_stateful"] = not has_sdpa
    except Exception:
        pass
    return result


# Constants for memory and block size configuration
GIB_BYTES = 1024 ** 3
CPU_BLOCK_SIZE = 32
GPU_BLOCK_SIZE = 16
DEFAULT_CPU_KV_CACHE_GB = 4

try:
    import openvino as ov
except ImportError as e:
    ov = None  # type: ignore[assignment]
    logger.warning("Failed to import OpenVINO with %r", e)


class OpenVinoPlatform(Platform):
    # PlatformEnum.CPU is used because PlatformEnum.OPENVINO may not exist in
    # vLLM 0.13.0. See upstream-compatibility vault note before changing.
    _enum = PlatformEnum.CPU
    device_name: str = "openvino"
    device_type: str = "cpu"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config) -> str:
        logger.info("Using OpenVINO Attention backend.")
        return "vllm_openvino.attention.backends.openvino.OpenVINOAttentionBackend"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "openvino"

    @classmethod
    def inference_mode(cls):
        return torch.inference_mode(mode=True)

    @classmethod
    def is_openvino_cpu(cls) -> bool:
        return "CPU" in envs.VLLM_OPENVINO_DEVICE

    @classmethod
    def is_openvino_gpu(cls) -> bool:
        return "GPU" in envs.VLLM_OPENVINO_DEVICE

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.debug("Pin memory is not supported on OpenVINO.")
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        if ov is None:
            raise ImportError(
                "OpenVINO is required but not installed. "
                "Install with: pip install openvino>=2026.1.0")

        parallel_config = vllm_config.parallel_config
        assert (parallel_config.world_size == 1
                ), "OpenVINO only supports single CPU socket currently."

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = \
                "vllm_openvino.worker_v1.openvino_worker_v1.OpenVINOWorkerV1"

        # check and update model config
        model_config = vllm_config.model_config
        if not model_config.enforce_eager:
            logger.warning(
                "CUDA graph is not supported on OpenVINO backend, fallback to "
                "the eager mode.")
            model_config.enforce_eager = True

        inspection = _inspect_model(model_config.model)

        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        if scheduler_config and scheduler_config.max_num_seqs != 1:
            if inspection["is_stateful"]:
                logger.warning(
                    "Stateful OpenVINO model detected. Overriding "
                    "max_num_seqs from %d to 1.",
                    scheduler_config.max_num_seqs)
                scheduler_config.max_num_seqs = 1

        # check and update cache config
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = GPU_BLOCK_SIZE

        _kv_precision_map = {
            "u8": "u8", "i8": "i8",
            "f16": "f16", "fp16": "f16",
            "bf16": "bf16",
            "f32": "f32", "fp32": "f32",
        }
        precision_key = envs.VLLM_OPENVINO_KV_CACHE_PRECISION
        cache_dtype = _kv_precision_map.get(precision_key or "")
        if cache_dtype is not None:
            logger.info(
                "KV cache type is overridden to %s via "
                "VLLM_OPENVINO_KV_CACHE_PRECISION env var.", cache_dtype)
            cache_config.cache_dtype = cache_dtype
        else:
            logger.info(
                "KV cache type is not specified via "
                "VLLM_OPENVINO_KV_CACHE_PRECISION env var. "
                "It will be determined automatically by a plugin")
            cache_config.cache_dtype = "dynamic"

        if OpenVinoPlatform.is_openvino_cpu():
            if cache_config.block_size != CPU_BLOCK_SIZE:
                logger.info(
                    f"OpenVINO CPU optimal block size is {CPU_BLOCK_SIZE}, overriding "
                    f"{cache_config.block_size} to {CPU_BLOCK_SIZE}")
                cache_config.block_size = CPU_BLOCK_SIZE
        else:
            if cache_config.block_size != GPU_BLOCK_SIZE:
                logger.info(
                    f"OpenVINO GPU optimal block size is {GPU_BLOCK_SIZE}, overriding "
                    f"{cache_config.block_size} to {GPU_BLOCK_SIZE}")
                cache_config.block_size = GPU_BLOCK_SIZE

        kv_cache_space = envs.VLLM_OPENVINO_KVCACHE_SPACE
        preset = inspection["preset"]
        if kv_cache_space >= 0:
            if kv_cache_space == 0:
                if preset and preset.get("min_kv_cache_gb", 0) > DEFAULT_CPU_KV_CACHE_GB:
                    recommended = preset["min_kv_cache_gb"]
                    cache_config.openvino_kvcache_space_bytes = (
                        recommended * GIB_BYTES)  # type: ignore
                    logger.warning(
                        "%s model detected. Auto-setting "
                        "VLLM_OPENVINO_KVCACHE_SPACE to %d GB. "
                        "Override with the env var if needed.",
                        model_config.model, recommended)
                elif OpenVinoPlatform.is_openvino_cpu():
                    cache_config.openvino_kvcache_space_bytes = (
                        DEFAULT_CPU_KV_CACHE_GB * GIB_BYTES)  # type: ignore
                    logger.warning(
                        "Environment variable VLLM_OPENVINO_KVCACHE_SPACE (GB) "
                        "for OpenVINO backend is not set, using 4 by default.")
                else:
                    cache_config.openvino_kvcache_space_bytes = 0  # type: ignore
                    logger.info(
                        "VLLM_OPENVINO_KVCACHE_SPACE is not set for GPU device. "
                        "KV cache size will be determined automatically via "
                        "profiling run.")
            else:
                cache_config.openvino_kvcache_space_bytes = (
                    kv_cache_space * GIB_BYTES)  # type: ignore
        else:
            raise RuntimeError(
                "Invalid environment variable VLLM_OPENVINO_KVCACHE_SPACE "
                f"{kv_cache_space}, expect a positive integer value.")

        # Disable torch compilation — OpenVINO compiles its own models
        from vllm.config import CompilationMode
        vllm_config.compilation_config.level = 0
        vllm_config.compilation_config.mode = CompilationMode.NONE

        assert vllm_config.lora_config is None, \
            "OpenVINO backend doesn't support LoRA"
        assert cls.is_openvino_cpu() or \
            cls.is_openvino_gpu() or \
            "empty" in envs.VLLM_OPENVINO_DEVICE, \
            "OpenVINO backend supports only CPU, GPU and empty devices"
