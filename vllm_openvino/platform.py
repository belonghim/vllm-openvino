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

try:
    import openvino as ov
except ImportError as e:
    ov = None  # type: ignore[assignment]
    logger.warning("Failed to import OpenVINO with %r", e)


def _is_stateful_model(model_path: str) -> bool:
    if ov is None:
        return False
    from pathlib import Path
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return False
    ir_path = model_dir / "openvino_language_model.xml"
    if not ir_path.exists():
        ir_path = model_dir / "openvino_model.xml"
    if not ir_path.exists():
        return False
    try:
        ov_model = ov.Core().read_model(str(ir_path))
        from vllm_openvino.model_executor.model_loader.openvino import (
            detect_model_type, HYBRID_MAMBA, STATEFUL,
        )
        model_type = detect_model_type(ov_model)
        return model_type in (STATEFUL, HYBRID_MAMBA)
    except (RuntimeError, ValueError):
        return False
    except Exception as e:
        logger.warning("Unexpected error in _is_stateful_model: %s", e)
        return False


# Constants for memory and block size configuration
GIB_BYTES = 1024 ** 3
CPU_BLOCK_SIZE = 32  # Matches vLLM CPU default; larger blocks amortize overhead
GPU_BLOCK_SIZE = 16  # Matches vLLM GPU default; smaller blocks for finer granularity
DEFAULT_CPU_KV_CACHE_GB = 4  # Conservative default for AVX2 systems with limited RAM


class OpenVinoPlatform(Platform):
    # PlatformEnum.CPU is used because PlatformEnum.OPENVINO may not exist in
    # vLLM 0.13.0. See upstream-compatibility vault note before changing.
    _enum = PlatformEnum.CPU
    device_name: str = "openvino"
    device_type: str = "cpu"
    # Match upstream CpuPlatform; init_distributed_environment() uses gloo too.
    dist_backend: str = "gloo"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config) -> str:
        logger.info("[OV-PLATFORM] Using OpenVINO Attention backend.")
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
    def import_ir_kernels(cls) -> None:
        pass

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.manual_seed(seed)

    @classmethod
    def get_current_memory_usage(
        cls, device: "torch.types.Device | None" = None
    ) -> float:
        return 0.0

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.debug("[OV-PLATFORM] Pin memory is not supported on OpenVINO.")
        return False

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype) -> None:
        return None

    @classmethod
    def get_punica_wrapper(cls) -> str:
        return ""

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        pass

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
                "[OV-PLATFORM] CUDA graph is not supported on OpenVINO backend, "
                "fallback to the eager mode.")
            model_config.enforce_eager = True

        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        if scheduler_config and scheduler_config.max_num_seqs != 1:
            if _is_stateful_model(model_config.model):
                logger.warning(
                    "[OV-PLATFORM] Stateful OpenVINO model detected. "
                    "Overriding max_num_seqs from %d to 1.",
                    scheduler_config.max_num_seqs)
                scheduler_config.max_num_seqs = 1

        # check and update cache config
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = GPU_BLOCK_SIZE

        precision_key = envs.VLLM_OPENVINO_KV_CACHE_PRECISION
        cache_dtype = envs.KV_CACHE_PRECISION_MAP.get(precision_key or "")
        if precision_key and cache_dtype is None:
            logger.warning(
                "[OV-PLATFORM] Unrecognized VLLM_OPENVINO_KV_CACHE_PRECISION=%r. "
                "Valid values: %s. Falling back to automatic detection.",
                precision_key, list(envs.KV_CACHE_PRECISION_MAP.keys()))
        if cache_dtype is not None:
            logger.info(
                "[OV-PLATFORM] KV cache type is overridden to %s via "
                "VLLM_OPENVINO_KV_CACHE_PRECISION env var.", cache_dtype)
            cache_config.cache_dtype = cache_dtype
        else:
            logger.info(
                "[OV-PLATFORM] KV cache type is not specified via "
                "VLLM_OPENVINO_KV_CACHE_PRECISION env var. "
                "It will be determined automatically by a plugin")
            cache_config.cache_dtype = "dynamic"

        if OpenVinoPlatform.is_openvino_cpu():
            if cache_config.block_size != CPU_BLOCK_SIZE:
                logger.info(
                    "[OV-PLATFORM] OpenVINO CPU optimal block size is %d, "
                    "overriding %s to %d",
                    CPU_BLOCK_SIZE, cache_config.block_size, CPU_BLOCK_SIZE)
                cache_config.block_size = CPU_BLOCK_SIZE
        else:
            if cache_config.block_size != GPU_BLOCK_SIZE:
                logger.info(
                    "[OV-PLATFORM] OpenVINO GPU optimal block size is %d, "
                    "overriding %s to %d",
                    GPU_BLOCK_SIZE, cache_config.block_size, GPU_BLOCK_SIZE)
                cache_config.block_size = GPU_BLOCK_SIZE

        kv_cache_space = envs.VLLM_OPENVINO_KVCACHE_SPACE
        if kv_cache_space >= 0:
            if kv_cache_space == 0 and OpenVinoPlatform.is_openvino_cpu():
                cache_config.openvino_kvcache_space_bytes = (
                    DEFAULT_CPU_KV_CACHE_GB * GIB_BYTES)  # type: ignore
                logger.warning(
                    "[OV-PLATFORM] Environment variable VLLM_OPENVINO_KVCACHE_SPACE "
                    "(GB) for OpenVINO backend is not set, using %d by default.",
                    DEFAULT_CPU_KV_CACHE_GB)
            elif kv_cache_space == 0:
                cache_config.openvino_kvcache_space_bytes = 0  # type: ignore
                logger.info(
                    "[OV-PLATFORM] VLLM_OPENVINO_KVCACHE_SPACE is not set for GPU "
                    "device. KV cache size will be determined automatically via "
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
