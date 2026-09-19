# SPDX-License-Identifier: Apache-2.0
from vllm.logger import init_logger
logger = init_logger(__name__)


def determine_num_available_blocks(current_platform, cache_config, cache_block_size: int, profile_run_func) -> tuple[int, int]:
    """Determine the number of blocks available for the KV cache.

    This determines how many KV blocks can fit into the configured
    KV cache space.
    """
    kvcache_space_bytes = cache_config.openvino_kvcache_space_bytes

    if current_platform.is_openvino_cpu():
        num_device_blocks = int(kvcache_space_bytes // cache_block_size)
        num_swap_blocks = 0
    else:
        if kvcache_space_bytes > 0:
            logger.info(
                "KV_CACHE size was explicitly configured via "
                "VLLM_OPENVINO_KVCACHE_SPACE environment variable, "
                "ignoring profiling run.")
            kv_cache_size = kvcache_space_bytes
        else:
            try:
                kv_cache_size = profile_run_func()
            except (RuntimeError, ValueError) as err:
                raise RuntimeError(
                    "The error occurred during profile run. This might be "
                    "due to insufficient GPU memory. Consider decreasing "
                    "`max_model_len` to limit the maximum simultaneously "
                    "processed tokens.") from err

        num_device_blocks = int(kv_cache_size // cache_block_size)
        num_swap_blocks = int(cache_config.swap_space_bytes //
                              cache_block_size)

    return num_device_blocks, num_swap_blocks

def get_max_allocatable_memory_gpu(ov_core, ov_device: str, key_cache_config: list, value_cache_config: list) -> int:
    import sys
    import openvino.properties.intel_gpu as intel_gpu
    if not hasattr(intel_gpu, "device_max_alloc_mem_size"):
        return sys.maxsize
    if not key_cache_config:
        return sys.maxsize

    max_tensor_alloc_size_gpu = ov_core.get_property(ov_device, intel_gpu.device_max_alloc_mem_size)
    assert len(key_cache_config) == len(value_cache_config), "Key cache config length should be equal to value cache config length."
    return len(key_cache_config) * 2 * max_tensor_alloc_size_gpu


def format_memory_size(size: float) -> str:
    """Convert byte size to human-readable string (B, KB, MB, GB)."""
    units = ["B", "KB", "MB", "GB"]
    unit_index = 0

    while size > 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    return f"{size:.2f} {units[unit_index]}"


# vLLM 0.27+ validates cache_config.cache_dtype against the CacheDType
# literal at engine init (KV cache layout resolution). OpenVINO-specific
# KV precisions ("u8", "i8", ...) live in cache_config.openvino_kv_dtype;
# cache_config.cache_dtype carries the closest valid vLLM literal instead.
# Unknown IR element types (e.g. "nf4") map to "auto" ("backend chooses").
VLLM_CACHE_DTYPE_BY_OV: dict[str, str] = {
    "u8": "int8_per_token_head",
    "i8": "int8_per_token_head",
    "f16": "float16",
    "fp16": "float16",
    "bf16": "bfloat16",
    "f32": "auto",
    "fp32": "auto",
}


def canonical_vllm_cache_dtype(ov_dtype: str | None) -> str:
    if ov_dtype is None:
        return "auto"
    return VLLM_CACHE_DTYPE_BY_OV.get(ov_dtype, "auto")


def ov_cache_dtype(cache_config) -> str:
    """OpenVINO element type for the KV cache. IR-detected value wins;
    "fp16" is the fallback before detection completes."""
    if not hasattr(cache_config, "openvino_kv_dtype"):
        raise ValueError(
            "cache_config.openvino_kv_dtype is missing. The OpenVINO platform "
            "sets it at config time (check_and_update_config); ensure the "
            "OpenVINO platform plugin is active.")
    return cache_config.openvino_kv_dtype or "fp16"


def has_sliding_window(model_config) -> bool:
    for cfg in (model_config,
                getattr(model_config, "hf_config", None),
                getattr(getattr(model_config, "hf_config", None), "text_config", None)):
        if cfg is None:
            continue
        window = getattr(cfg, "sliding_window", None)
        if window is not None and window != 0:
            return True
    return False
