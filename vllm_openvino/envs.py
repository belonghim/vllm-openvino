# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    VLLM_OPENVINO_DEVICE: str = "CPU"
    VLLM_OPENVINO_KVCACHE_SPACE: int = 0
    VLLM_OPENVINO_KV_CACHE_PRECISION: str | None = None
    VLLM_OPENVINO_PERFORMANCE_MODE: str = "THROUGHPUT"
    VLLM_OPENVINO_CPU_THREADS_NUM: int = 0
    VLLM_OPENVINO_NUM_STREAMS: str | int = 1
    VLLM_OPENVINO_ENABLE_HYPER_THREADING: bool | None = None
    VLLM_OPENVINO_INFERENCE_PRECISION: str | None = None
    VLLM_OPENVINO_ENABLE_CPU_PINNING: bool | None = None
    VLLM_OPENVINO_STATEFUL_PA: bool = True
    VLLM_OPENVINO_HYBRID_PA: bool = True
    VLLM_OPENVINO_CACHE_DIR: str | None = None
    VLLM_OPENVINO_SCHEDULING_CORE_TYPE: str | None = None
    VLLM_OPENVINO_FAST_SAMPLER: bool = True

KV_CACHE_PRECISION_MAP: dict[str, str] = {
    "u8": "u8", "i8": "i8",
    "f16": "f16", "fp16": "f16",
    "bf16": "bf16",
    "f32": "f32", "fp32": "f32",
}

environment_variables: dict[str, Callable[[], Any]] = {
    # OpenVINO device selection
    # default is CPU
    "VLLM_OPENVINO_DEVICE":
    lambda: os.getenv("VLLM_OPENVINO_DEVICE", "CPU").upper(),

    # OpenVINO key-value cache space
    # default is 0 (auto: 4 GB on CPU)
    "VLLM_OPENVINO_KVCACHE_SPACE":
    lambda: int(os.getenv("VLLM_OPENVINO_KVCACHE_SPACE", "0")),

    # OpenVINO KV cache precision
    # default 'undefined', which means plugin will automatically set
    # proper value based on model analysis
    "VLLM_OPENVINO_KV_CACHE_PRECISION":
    lambda: os.getenv("VLLM_OPENVINO_KV_CACHE_PRECISION", None),

    # OpenVINO performance mode: LATENCY or THROUGHPUT
    # LATENCY is recommended for faster first-token response on CPU
    "VLLM_OPENVINO_PERFORMANCE_MODE":
    lambda: os.getenv("VLLM_OPENVINO_PERFORMANCE_MODE", "THROUGHPUT").upper(),

    # CPU-only: cap total inference threads used by OpenVINO CPU plugin
    # 0 means OpenVINO auto-selects threads
    "VLLM_OPENVINO_CPU_THREADS_NUM":
    lambda: int(os.getenv("VLLM_OPENVINO_CPU_THREADS_NUM", "0")),

    # Number of CPU inference streams.
    # vLLM V1 issues a single blocking infer per step, so multiple streams
    # fragment the thread budget without any concurrency to fill them.
    # Keep this at 1 unless the plugin is modified for async inference.
    # AUTO remains an accepted user-supplied value (OpenVINO heuristic).
    "VLLM_OPENVINO_NUM_STREAMS":
    lambda: (lambda v: int(v) if v.isdigit() else v.upper())(
        os.getenv("VLLM_OPENVINO_NUM_STREAMS", "1")),

    # CPU-only: enable/disable hyperthreading. When disabled, uses 1 thread
    # per physical core instead of 2 (useful on oversubscription-prone systems).
    "VLLM_OPENVINO_ENABLE_HYPER_THREADING":
    lambda: (lambda v: None if v in ("", "auto") else v == "true")(
        os.getenv("VLLM_OPENVINO_ENABLE_HYPER_THREADING", "").lower()),

    # CPU-only: inference precision hint (f32, f16, bf16). Forces specific
    # precision for matmul operations. On CPUs without int8 acceleration, this
    # can avoid expensive int8->fp dequantization overhead.
    "VLLM_OPENVINO_INFERENCE_PRECISION":
    lambda: os.getenv("VLLM_OPENVINO_INFERENCE_PRECISION", None),

    # CPU-only: enable/disable CPU core pinning. When enabled, threads are
    # pinned to specific CPU cores to avoid migration penalties.
    "VLLM_OPENVINO_ENABLE_CPU_PINNING":
    lambda: (lambda v: None if v in ("", "auto") else v == "true")(
        os.getenv("VLLM_OPENVINO_ENABLE_CPU_PINNING", "").lower()),

    # PagedAttention transformation for plain-attention stateful models
    # (ReadValue-based KV cache, e.g. optimum-intel default exports like
    # Qwen2.5-Coder-int4-ov). Enables concurrent request batching
    # (max_num_seqs > 1). Sliding-window models and models without SDPA ops
    # fall back to the sequential stateful path. Set to 0 to force the
    # sequential stateful path.
    "VLLM_OPENVINO_STATEFUL_PA":
    lambda: os.getenv("VLLM_OPENVINO_STATEFUL_PA", "1") == "1",

    # Default path for hybrid Mamba/attention models (Qwen3.5, LFM2.5):
    # attempt PagedAttention transformation instead of the sequential
    # stateful path, enabling concurrent request batching (max_num_seqs > 1).
    # Set to 0 to force the sequential stateful path instead (e.g. if a new,
    # unvalidated hybrid model hits a PA transformation issue).
    "VLLM_OPENVINO_HYBRID_PA":
    lambda: os.getenv("VLLM_OPENVINO_HYBRID_PA", "1") == "1",

    # Directory for OpenVINO's compiled-model disk cache. When set, OpenVINO
    # skips recompiling the model on process restart if a matching cached
    # blob exists (keyed by model + OpenVINO version). Unset (default) keeps
    # cold-compile-every-start behavior, since not every deployment has a
    # writable, persistent path available.
    "VLLM_OPENVINO_CACHE_DIR":
    lambda: os.getenv("VLLM_OPENVINO_CACHE_DIR", None),

    # CPU-only: scheduling core type on hybrid P-core/E-core CPUs
    # (PCORE_ONLY, ECORE_ONLY, ANY_CORE). None keeps OpenVINO default.
    "VLLM_OPENVINO_SCHEDULING_CORE_TYPE":
    lambda: (lambda v: v.upper() if v else None)(
        os.getenv("VLLM_OPENVINO_SCHEDULING_CORE_TYPE", None)),

    # Replace vLLM V1's torch.exponential_() Gumbel-max noise draw in
    # compiled_random_sample with a numpy PCG64 standard_exponential fill.
    # Default 1 (enabled): ~4x faster on the (batch, vocab) shapes seen
    # at decode; set to 0 to fall back to stock torch behavior. Only fires
    # when at least one request in the batch has no per-request seed
    # (upstream forward_cpu already routes fully-seeded batches through
    # the per-generator torch path, which this patch does not touch).
    # Caveat: swapping the RNG changes the token stream produced by a
    # given user-supplied seed vs stock vLLM (per-repeat determinism is
    # still preserved on the seeded path, which is unmodified).
    "VLLM_OPENVINO_FAST_SAMPLER":
    lambda: os.getenv("VLLM_OPENVINO_FAST_SAMPLER", "1") == "1",
}

# end-env-vars-definition

def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
