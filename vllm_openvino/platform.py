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
        GiB_bytes = 1024 * 1024 * 1024

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

        # check and update cache config
        cache_config = vllm_config.cache_config
        if cache_config and cache_config.block_size is None:
            cache_config.block_size = 16

        _kv_precision_map = {
            "u8": "u8", "i8": "i8",
            "f16": "f16", "fp16": "f16",
            "bf16": "bf16",
            "f32": "f32", "fp32": "f32",
        }
        precision_key = envs.VLLM_OPENVINO_KV_CACHE_PRECISION
        cache_dtype = _kv_precision_map.get(precision_key)
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
            if cache_config.block_size != 32:
                logger.info(
                    f"OpenVINO CPU optimal block size is 32, overriding "
                    f"{cache_config.block_size} to 32")
                cache_config.block_size = 32
        else:
            if cache_config.block_size != 16:
                logger.info(
                    f"OpenVINO GPU optimal block size is 16, overriding "
                    f"{cache_config.block_size} to 16")
                cache_config.block_size = 16

        kv_cache_space = envs.VLLM_OPENVINO_KVCACHE_SPACE
        if kv_cache_space >= 0:
            if kv_cache_space == 0 and OpenVinoPlatform.is_openvino_cpu():
                cache_config.openvino_kvcache_space_bytes = 4 * GiB_bytes  # type: ignore
                logger.warning(
                    "Environment variable VLLM_OPENVINO_KVCACHE_SPACE (GB) "
                    "for OpenVINO backend is not set, using 4 by default.")
            else:
                cache_config.openvino_kvcache_space_bytes = (  # type: ignore
                    kv_cache_space * GiB_bytes)
                if kv_cache_space == 0 and not OpenVinoPlatform.is_openvino_cpu():
                    logger.info(
                        "VLLM_OPENVINO_KVCACHE_SPACE is not set for GPU device. "
                        "KV cache size will be determined automatically via "
                        "profiling run.")
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
