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


def _find_model_ir_path(model_path: str) -> "Path | None":
    from pathlib import Path
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return None
    ir_path = model_dir / "openvino_language_model.xml"
    if not ir_path.exists():
        ir_path = model_dir / "openvino_model.xml"
    if not ir_path.exists():
        return None
    return ir_path


def _scan_ir_for_patterns(ir_path, patterns: list[bytes]) -> list[bool]:
    """Scan an IR XML file for byte patterns in one pass.

    Keeps a small overlap between chunks so a pattern isn't missed when
    split across a chunk boundary.
    """
    found = [False] * len(patterns)
    max_pattern_len = max(len(p) for p in patterns)
    overlap = max_pattern_len - 1
    carry = b''
    with open(ir_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            window = carry + chunk
            for i, pattern in enumerate(patterns):
                if not found[i] and pattern in window:
                    found[i] = True
            carry = window[-overlap:] if overlap > 0 else b''
    return found


def _is_stateful_model(model_path: str) -> bool:
    ir_path = _find_model_ir_path(model_path)
    if ir_path is None:
        return False
    try:
        (has_readvalue,) = _scan_ir_for_patterns(ir_path, [b'type="ReadValue"'])
        return has_readvalue
    except (IOError, OSError) as e:
        logger.warning("_is_stateful_model: could not read %s: %s", ir_path, e)
        return False


def _is_hybrid_pa_candidate(model_path: str) -> bool:
    """Conv-only hybrid model eligible for the experimental PA path: has
    ReadValue state but no SSM variable_id, and has SDPA attention ops.
    """
    ir_path = _find_model_ir_path(model_path)
    if ir_path is None:
        return False
    try:
        has_readvalue, has_sdpa, has_ssm = _scan_ir_for_patterns(
            ir_path, [b'type="ReadValue"', b'ScaledDotProductAttention', b'variable_id="ssm'])
        return has_readvalue and has_sdpa and not has_ssm
    except (IOError, OSError) as e:
        logger.warning("_is_hybrid_pa_candidate: could not read %s: %s", ir_path, e)
        return False


# Constants for memory and block size configuration
GIB_BYTES = 1024 ** 3
CPU_BLOCK_SIZE = 32  # Matches vLLM CPU default; larger blocks amortize overhead
GPU_BLOCK_SIZE = 16  # Matches vLLM GPU default; smaller blocks for finer granularity
DEFAULT_CPU_KV_CACHE_GB = 4  # Conservative default for AVX2 systems with limited RAM


class OpenVinoPlatform(Platform):
    # PlatformEnum.CPU is used because PlatformEnum.OPENVINO does not exist in
    # vLLM upstream. See upstream-compatibility vault note before changing.
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
                "Install with: pip install openvino>=2026.3.0")

        parallel_config = vllm_config.parallel_config
        if parallel_config.world_size != 1:
            raise ValueError("OpenVINO only supports single CPU socket currently.")

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
                if (envs.VLLM_OPENVINO_HYBRID_PA
                        and _is_hybrid_pa_candidate(model_config.model)):
                    logger.info(
                        "[OV-PLATFORM] Conv-only hybrid model with "
                        "VLLM_OPENVINO_HYBRID_PA=1: attempting PagedAttention "
                        "transformation, keeping max_num_seqs=%d.",
                        scheduler_config.max_num_seqs)
                else:
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
        cache_dtype = envs.KV_CACHE_PRECISION_MAP.get((precision_key or "").lower())
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

        target_block_size = CPU_BLOCK_SIZE if OpenVinoPlatform.is_openvino_cpu() else GPU_BLOCK_SIZE
        if cache_config.block_size != target_block_size:
            logger.info(
                "[OV-PLATFORM] OpenVINO optimal block size is %d, "
                "overriding %s to %d",
                target_block_size, cache_config.block_size, target_block_size)
            cache_config.block_size = target_block_size

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

        if vllm_config.lora_config is not None:
            raise ValueError("OpenVINO backend doesn't support LoRA")
        if not (cls.is_openvino_cpu() or cls.is_openvino_gpu()
                or "empty" in envs.VLLM_OPENVINO_DEVICE):
            raise ValueError(
                "OpenVINO backend supports only CPU, GPU and empty devices")
