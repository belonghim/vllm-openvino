# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path

import vllm_openvino.envs as envs
from vllm.logger import init_logger
logger = init_logger(__name__)


def detect_cgroup_cpu_quota() -> int | None:
    """Detect the effective CPU count from a cgroup v2/v1 CFS quota.

    Container runtimes (podman/docker --cpus, Kubernetes CPU limits) throttle
    via cgroup CFS quota but leave os.cpu_count()/sched_getaffinity() reporting
    the host's full core count. OpenVINO's thread auto-detection (num_threads
    unset) uses the latter, so it oversubscribes under a tighter quota —
    measured on an 8-quota/24-visible-core host: 91 threads spawned, 45s/~355%
    CPU vs. 34s/~300% CPU when capped to 8 threads for the same workload.
    Returns None if no quota is set (unconstrained) or it can't be read.
    """
    try:
        quota_max_path = Path("/sys/fs/cgroup/cpu.max")
        if quota_max_path.exists():
            quota_str, period_str = quota_max_path.read_text().split()
            if quota_str == "max":
                return None
            quota, period = int(quota_str), int(period_str)
        else:
            cfs_quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
            cfs_period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
            if not (cfs_quota_path.exists() and cfs_period_path.exists()):
                return None
            quota = int(cfs_quota_path.read_text())
            period = int(cfs_period_path.read_text())
            if quota <= 0:
                return None
        if period <= 0:
            return None
        effective = quota // period
        return effective if effective > 0 else None
    except (OSError, ValueError):
        return None


def cpu_thread_limit() -> tuple[int | None, str]:
    """Thread cap shared by OpenVINO, Torch and OMP thread pools.

    Returns (limit, reason); limit None means no cap applies. Thread pools
    sized from os.cpu_count() oversubscribe under a tighter cgroup quota,
    which CFS throttling turns into wasted CPU time for the same work.
    """
    explicit = envs.VLLM_OPENVINO_CPU_THREADS_NUM
    if explicit > 0:
        return explicit, f"VLLM_OPENVINO_CPU_THREADS_NUM={explicit}"
    visible = os.cpu_count() or 0
    quota = detect_cgroup_cpu_quota()
    if quota is not None and quota < visible:
        return quota, f"cgroup CPU quota {quota} below {visible} visible cores"
    return None, "no cgroup quota below the visible core count"


def detect_cgroup_memory_limit() -> int | None:
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
        except OSError:
            continue
        if raw == "max":
            return None
        try:
            value = int(raw)
        except ValueError:
            continue
        return None if value >= 1 << 62 else value
    return None


def model_weights_bytes(model_path: str) -> int:
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return 0
    total = 0
    for blob in model_dir.glob("openvino_*.bin"):
        try:
            total += blob.stat().st_size
        except OSError:
            continue
    return total



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
