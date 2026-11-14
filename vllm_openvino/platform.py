# SPDX-License-Identifier: Apache-2.0
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.platforms.interface import Platform, PlatformEnum

import vllm_openvino.envs as envs
from vllm_openvino.utils import (canonical_vllm_cache_dtype,
                                 cpu_thread_limit,
                                 detect_cgroup_memory_limit,
                                 detect_memory_limit,
                                 format_memory_size,
                                 get_mem_available_bytes,
                                 image_input_enabled,
                                 model_weights_bytes)

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


def _cap_torch_threads() -> None:
    if envs.VLLM_OPENVINO_DEVICE.upper() != "CPU":
        return
    if "OMP_NUM_THREADS" in os.environ:
        return
    limit, reason = cpu_thread_limit()
    if not limit:
        return
    # torch.set_num_threads() is the cap that actually takes effect for this
    # process — the OpenMP runtime is already initialized by `import torch`
    # above and does not re-read OMP_NUM_THREADS. The env var is still set so
    # it propagates to any subprocesses (e.g. HF tokenizers workers) and to
    # OMP-linked libraries that lazy-init on first use.
    os.environ["OMP_NUM_THREADS"] = str(limit)
    if torch.get_num_threads() != limit:
        torch.set_num_threads(limit)
    logger.info(
        "[OV-PLATFORM] Capping Torch threads to %d (%s) to avoid CPU "
        "oversubscription.", limit, reason)


def _has_vision_ir(model_path: str) -> bool:
    model_dir = Path(model_path)
    if not model_dir.is_dir():
        return False
    return any(model_dir.glob("openvino_vision_embeddings*.xml"))


def apiserver_overhead_bytes(model_config) -> int:
    if _has_vision_ir(model_config.model) and image_input_enabled(model_config):
        return VISION_APISERVER_OVERHEAD_BYTES
    return APISERVER_OVERHEAD_BYTES


def _fit_kv_cache_to_memory_limit(kv_bytes: int, model_config) -> int:
    limit = detect_memory_limit()
    if limit is None:
        return kv_bytes
    weights = model_weights_bytes(model_config.model)
    api_overhead = apiserver_overhead_bytes(model_config)
    runtime_overhead = CONTAINER_RUNTIME_BASELINE_BYTES + api_overhead
    budget = (limit - weights - runtime_overhead
              - KV_CACHE_MEMORY_MARGIN_BYTES)
    if kv_bytes > budget:
        fitted = max(KV_CACHE_MIN_BYTES, budget)
        logger.warning(
            "[OV-PLATFORM] KV cache space %s does not fit the memory "
            "limit %s after reserving %s runtime baseline, %s APIServer "
            "overhead, %s model weights and %s headroom; sizing the pool to "
            "%s to avoid an OOM kill. Raise the container memory limit or "
            "lower VLLM_OPENVINO_KVCACHE_SPACE to keep a larger pool.",
            format_memory_size(kv_bytes), format_memory_size(limit),
            format_memory_size(CONTAINER_RUNTIME_BASELINE_BYTES),
            format_memory_size(api_overhead),
            format_memory_size(weights),
            format_memory_size(KV_CACHE_MEMORY_MARGIN_BYTES),
            format_memory_size(fitted))
        return fitted
    if kv_bytes * 10 > budget * 9:
        logger.warning(
            "[OV-PLATFORM] KV cache space %s leaves under 10%% headroom under "
            "the memory limit %s (runtime baseline %s, APIServer overhead "
            "%s, model weights %s, headroom %s); an OOM is possible once the "
            "pool is fully touched.",
            format_memory_size(kv_bytes), format_memory_size(limit),
            format_memory_size(CONTAINER_RUNTIME_BASELINE_BYTES),
            format_memory_size(api_overhead),
            format_memory_size(weights),
            format_memory_size(KV_CACHE_MEMORY_MARGIN_BYTES))
    return kv_bytes


def auto_kv_cache_bytes(resident_bytes: int, model_config) -> int:
    limit = detect_cgroup_memory_limit()
    if limit is None:
        # MemAvailable already excludes this process's anonymous memory, so the
        # resident part is added back before it is subtracted again.
        limit = resident_bytes + get_mem_available_bytes()
    budget = (limit - resident_bytes - apiserver_overhead_bytes(model_config)
              - KV_CACHE_MEMORY_MARGIN_BYTES)
    return max(KV_CACHE_MIN_BYTES, budget)


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


_VARIABLE_ID_RE = re.compile(rb'variable_id="([^"]*)"')


def _ir_serving_traits(model_path: str) -> tuple[bool, bool] | None:
    """(has SDPA, hybrid) from one read of the IR XML; None without an IR.

    Hybrid is deliberately narrower than "has ReadValue + SDPA": Gemma-4 also
    matches that (KV cache via ReadValue) but has no ssm/conv state and belongs
    on the plain PagedAttention path — so ssm/conv must appear inside
    variable_id, not anywhere in the file (gemma-4's IR has 278 unrelated
    "conv" hits from `type="Convert"` ops).
    """
    ir_path = _find_model_ir_path(model_path)
    if ir_path is None:
        return None
    try:
        with open(ir_path, 'rb') as f:
            data = f.read()
    except (IOError, OSError) as e:
        logger.warning("Could not read %s: %s", ir_path, e)
        return None
    has_sdpa = b'ScaledDotProductAttention' in data
    hybrid = has_sdpa and any(
        b'ssm' in vid or b'conv' in vid
        for vid in _VARIABLE_ID_RE.findall(data))
    return has_sdpa, hybrid


# Constants for memory and block size configuration
GIB_BYTES = 1024 ** 3
CPU_BLOCK_SIZE = 32  # Matches vLLM CPU default; larger blocks amortize overhead
GPU_BLOCK_SIZE = 16  # Matches vLLM GPU default; smaller blocks for finer granularity
# Covers prefill and allocator transients that peak ~1 GiB above the steady state.
KV_CACHE_MEMORY_MARGIN_BYTES = GIB_BYTES * 11 // 5
KV_CACHE_MIN_BYTES = GIB_BYTES // 4  # below this a pool cannot serve usefully
CONTAINER_RUNTIME_BASELINE_BYTES = 3 * GIB_BYTES // 2  # python/torch/vLLM resident before the pool
# Measured APIServer anon RSS: 0.55-0.87 GiB text, 1.23-1.26 GiB Gemma-4 with
# images disabled (262k-token vocab), 2.1-2.7 GiB vision.
APISERVER_OVERHEAD_BYTES = GIB_BYTES * 3 // 2
VISION_APISERVER_OVERHEAD_BYTES = GIB_BYTES * 11 // 4


class OpenVinoPlatform(Platform):
    # PlatformEnum.CPU is used because PlatformEnum.OPENVINO does not exist in
    # vLLM upstream. See upstream-compatibility vault note before changing.
    _enum = PlatformEnum.CPU
    device_name: str = "openvino"
    device_type: str = "cpu"
    # Match upstream CpuPlatform; init_distributed_environment() uses gloo too.
    dist_backend: str = "gloo"

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config,
                             num_heads: int | None = None) -> str:
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
    def import_kernels(cls) -> None:
        pass

    @classmethod
    def validate_environ(cls, hard_fail: bool) -> None:
        from vllm import envs as vllm_envs

        for env, value in os.environ.items():
            if (env == "VLLM_USE_V2_MODEL_RUNNER"
                    and env not in vllm_envs.environment_variables):
                # Removed with MRV1 (vllm#63373); mirror upstream's check,
                # which this override replaces.
                if value != "1":
                    raise ValueError(
                        f"VLLM_USE_V2_MODEL_RUNNER={value} is no longer "
                        "supported: Model Runner V1 has been removed. Unset "
                        "this variable.")
                logger.warning(
                    "VLLM_USE_V2_MODEL_RUNNER=1 is redundant and can be unset: "
                    "Model Runner V1 has been removed.")
                continue
            if env == "VLLM_OPENVINO_STATEFUL_PA":
                if value.strip().lower() not in envs.TRUE_VALUES:
                    raise ValueError(
                        f"VLLM_OPENVINO_STATEFUL_PA={value} is no longer "
                        "supported: the sequential stateful path has been "
                        "removed. Unset this variable.")
                logger.warning(
                    "VLLM_OPENVINO_STATEFUL_PA is redundant and can be unset: "
                    "every model runs on the PagedAttention or Hybrid-PA path.")
                continue
            if (env.startswith("VLLM_") and env not in vllm_envs.environment_variables
                    and env not in envs.environment_variables):
                if hard_fail:
                    raise ValueError(
                        f"Unknown vLLM environment variable detected: {env}")
                logger.warning(
                    "Unknown vLLM environment variable detected: %s", env)

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
    def validate_request(cls, processed_inputs, params) -> None:
        # Logits are computed only at sampled positions (gather-before-matmul),
        # so prompt logprobs would silently come back truncated.
        if getattr(params, "prompt_logprobs", None) is not None:
            raise VLLMValidationError(
                "prompt_logprobs is not supported on the OpenVINO backend.",
                parameter="prompt_logprobs",
                value=params.prompt_logprobs)

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: "VllmConfig") -> None:
        pass

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        if ov is None:
            raise ImportError(
                "OpenVINO is required but not installed. "
                "Install with: pip install openvino>=2026.4.1")

        _cap_torch_threads()

        if not os.environ.get("TORCH_COMPILE_DISABLE"):
            logger.warning(
                "[OV-PLATFORM] TORCH_COMPILE_DISABLE is not set. "
                "torch.compile is incompatible with OpenVINO; set "
                "TORCH_COMPILE_DISABLE=1 to prevent unexpected failures.")

        if vllm_config.model_config.logits_processors:
            raise ValueError(
                "Custom logits processors are not supported on the OpenVINO "
                "backend; its sampler applies only vLLM's builtin min_p, "
                "logit_bias and min_tokens.")

        parallel_config = vllm_config.parallel_config
        if parallel_config.world_size != 1:
            raise ValueError(
                "OpenVINO does not support tensor or pipeline parallelism "
                f"(world_size={parallel_config.world_size}).")

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

        cache_config = vllm_config.cache_config
        hybrid_pa = False
        traits = _ir_serving_traits(model_config.model)
        if traits is not None:
            has_sdpa, hybrid_pa = traits
            if not has_sdpa:
                raise ValueError(
                    f"The OpenVINO IR in {model_config.model} has no "
                    "ScaledDotProductAttention ops, so it cannot run on the "
                    "PagedAttention or Hybrid-PA path (the sequential stateful "
                    "path was removed). Re-export the model with SDPA.")
            if hybrid_pa:
                logger.info(
                    "[OV-PLATFORM] Hybrid Mamba/attention model detected: "
                    "using the Hybrid-PA path.")

        speculative_config = vllm_config.speculative_config
        if (speculative_config is not None
                and speculative_config.method not in ("ngram", "mtp")):
            raise ValueError(
                "OpenVINO only supports ngram and mtp speculative decoding, "
                f"got method={speculative_config.method!r}.")
        if speculative_config is not None and speculative_config.method == "mtp":
            if not (hybrid_pa and (Path(model_config.model)
                                   / "openvino_mtp_model.xml").exists()):
                raise ValueError(
                    "mtp speculative decoding needs a Hybrid-PA IR that ships "
                    "openvino_mtp_model.xml (Qwen3.5/3.6 exports).")
            # Drafts go back to the scheduler through take_draft_token_ids,
            # as with ngram, which needs synchronous scheduling.
            vllm_config.scheduler_config.async_scheduling = False

        # Recurrent state is not stored in vLLM blocks, so a prefix-cache hit
        # would skip tokens whose state is never restored. Hybrid-PA keeps a
        # conv/SSM checkpoint per scheduler block instead (see the model
        # runner), except with drafts, whose per-token state slots are not
        # checkpointed.
        if (hybrid_pa and speculative_config is not None and cache_config
                and cache_config.enable_prefix_caching):
            logger.warning(
                "[OV-PLATFORM] Disabling prefix caching: Hybrid-PA conv/SSM "
                "checkpoints do not cover speculative decoding.")
            cache_config.enable_prefix_caching = False
        if hybrid_pa and hasattr(cache_config, "mamba_cache_mode"):
            cache_config.mamba_cache_mode = "none"
            cache_config.mamba_block_size = None

        # check and update cache config
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
        else:
            logger.info(
                "[OV-PLATFORM] KV cache type is not specified via "
                "VLLM_OPENVINO_KV_CACHE_PRECISION env var. "
                "It will be determined automatically by a plugin")
        # cache_dtype must be a vLLM CacheDType literal (validated at engine
        # init); the OpenVINO precision lives in openvino_kv_dtype and is
        # resolved to the IR element type during load_model.
        cache_config.openvino_kv_dtype = cache_dtype
        cache_config.cache_dtype = canonical_vllm_cache_dtype(cache_dtype)

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
                cache_config.openvino_kvcache_space_bytes = auto_kv_cache_bytes(  # type: ignore
                    model_weights_bytes(model_config.model)
                    + CONTAINER_RUNTIME_BASELINE_BYTES,
                    model_config)
                logger.info(
                    "[OV-PLATFORM] VLLM_OPENVINO_KVCACHE_SPACE is not set; the "
                    "KV pool is sized from the memory limit after model load "
                    "(startup estimate %s).",
                    format_memory_size(
                        cache_config.openvino_kvcache_space_bytes))
            elif kv_cache_space == 0:
                cache_config.openvino_kvcache_space_bytes = 0  # type: ignore
                logger.info(
                    "[OV-PLATFORM] VLLM_OPENVINO_KVCACHE_SPACE is not set for GPU "
                    "device. KV cache size will be determined automatically via "
                    "profiling run.")
            else:
                kv_cache_bytes = int(kv_cache_space * GIB_BYTES)
                if OpenVinoPlatform.is_openvino_cpu():
                    kv_cache_bytes = _fit_kv_cache_to_memory_limit(
                        kv_cache_bytes, model_config)
                cache_config.openvino_kvcache_space_bytes = kv_cache_bytes  # type: ignore
        else:
            raise RuntimeError(
                "Invalid environment variable VLLM_OPENVINO_KVCACHE_SPACE "
                f"{kv_cache_space}, expect a non-negative number of GiB.")

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
