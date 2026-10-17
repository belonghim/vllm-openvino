# SPDX-License-Identifier: Apache-2.0
import os
import threading
import time
from pathlib import Path

import vllm.envs as vllm_envs
import vllm_openvino.envs as envs
from vllm.logger import init_logger
logger = init_logger(__name__)


class StepStatsLogger:
    """Per-step timing summary, logged every VLLM_LOG_STATS_INTERVAL while busy.

    Runs in the EngineCore process, where the OpenVINO calls happen; the
    API-server loggers.py line cannot see these timings. Idle windows log nothing.
    """

    def __init__(self) -> None:
        self._interval = vllm_envs.VLLM_LOG_STATS_INTERVAL
        self._thread: threading.Thread | None = None
        self._window_start = time.monotonic()
        self.reset()

    def reset(self) -> None:
        self._steps = 0
        self._tokens = 0
        self._max_step = 0.0
        self._sum = {"prep": 0.0, "infer": 0.0, "other": 0.0, "sample": 0.0}
        self._cur = 0.0

    def add(self, **seconds: float) -> None:
        for key, value in seconds.items():
            self._sum[key] += value
            self._cur += value

    def end_step(self, num_tokens: int) -> None:
        self._steps += 1
        self._tokens += num_tokens
        self._max_step = max(self._max_step, self._cur)
        self._cur = 0.0
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="ov-step-stats", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while True:
            time.sleep(self._interval)
            self._flush()

    def _flush(self) -> None:
        steps, tokens, max_step, sums = (
            self._steps, self._tokens, self._max_step, self._sum)
        now = time.monotonic()
        elapsed, self._window_start = now - self._window_start, now
        self.reset()
        if not steps:
            return
        avg = {k: v / steps * 1000 for k, v in sums.items()}
        logger.info(
            "[OV-RUNNER] %.1fs: %d steps, %.1f scheduled tokens/s, step avg "
            "%.1f ms (prep %.1f, infer %.1f, other forward %.1f, sample %.1f), "
            "max %.1f ms",
            elapsed, steps, tokens / elapsed, sum(avg.values()), avg["prep"],
            avg["infer"], avg["other"], avg["sample"], max_step * 1000)


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


def get_process_rss_bytes() -> int:
    """Read current process RSS from /proc/self/status (Linux)."""
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except (OSError, ValueError):
        pass
    return 0


def get_process_anon_rss_bytes() -> int:
    """Read the anonymous (non-reclaimable) part of this process's RSS.

    VmRSS also counts the memory-mapped model weights, which the kernel
    reclaims under pressure and re-faults from disk; sizing a buffer against
    VmRSS would treat those pages as committed. RssAnon excludes them.
    """
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('RssAnon:'):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except (OSError, ValueError):
        pass
    return 0


def get_cgroup_memory_usage_bytes() -> int:
    """Returns anonymous (non-reclaimable) memory used by this cgroup.

    Uses 'anon' from memory.stat rather than memory.current because
    memory.current includes reclaimable file-backed page cache which can
    inflate to the container limit even when committed memory is much less.
    """
    try:
        with open('/sys/fs/cgroup/memory.stat') as f:
            for line in f:
                if line.startswith('anon '):
                    value = int(line.split()[1])
                    if value > 0:
                        return value
    except (OSError, ValueError):
        pass
    try:
        with open('/sys/fs/cgroup/memory/memory.stat') as f:
            for line in f:
                if line.startswith('rss '):
                    value = int(line.split()[1])
                    if value > 0:
                        return value
    except (OSError, ValueError):
        pass
    return 0


def detect_memory_limit() -> int | None:
    """Effective memory ceiling for this process.

    Returns the cgroup hard limit when one is active (K8s limits.memory,
    podman/docker --memory).  When no hard limit is configured (BestEffort
    or Burstable K8s pods without explicit memory limits, bare metal),
    returns current RSS + /proc/meminfo MemAvailable so the pool is still
    bounded by what the host can provide right now.  Returns None only
    when all /proc reads fail.
    """
    limit = detect_cgroup_memory_limit()
    if limit is not None:
        return limit
    try:
        mem_available_kb = 0
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    mem_available_kb = int(line.split()[1])
                    break
        if mem_available_kb > 0:
            rss = get_process_rss_bytes()
            if rss > 0:
                return rss + mem_available_kb * 1024
    except (OSError, ValueError):
        pass
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
