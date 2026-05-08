# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    VLLM_OPENVINO_DEVICE: str = "CPU"
    VLLM_OPENVINO_KVCACHE_SPACE: int = 0
    VLLM_OPENVINO_KV_CACHE_PRECISION: str | None = None
    VLLM_OPENVINO_CPU_THREADS_NUM: int = 0
    VLLM_OPENVINO_CPU_BIND_THREAD: str | None = None
    VLLM_OPENVINO_NUM_STREAMS: str | int = "AUTO"
    VLLM_OPENVINO_ENABLE_HYPER_THREADING: bool | None = None
    VLLM_OPENVINO_INFERENCE_PRECISION: str | None = None
    VLLM_OPENVINO_ENABLE_CPU_PINNING: bool | None = None
    VLLM_OPENVINO_MEMORY_THRESHOLD: float = 1.1
    VLLM_OPENVINO_CPU_BLOCK_SIZE: int = 32
    VLLM_OPENVINO_GPU_BLOCK_SIZE: int = 16

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
    lambda: os.getenv("VLLM_OPENVINO_PERFORMANCE_MODE", "LATENCY").upper(),

    # CPU-only: cap total inference threads used by OpenVINO CPU plugin
    # 0 means OpenVINO auto-selects threads
    "VLLM_OPENVINO_CPU_THREADS_NUM":
    lambda: int(os.getenv("VLLM_OPENVINO_CPU_THREADS_NUM", "0")),

    # CPU-only: thread binding policy (CORE, NUMA, NONE)
    # None means keep OpenVINO default behavior
    "VLLM_OPENVINO_CPU_BIND_THREAD":
    lambda: (lambda v: v.upper() if v else None)(
        os.getenv("VLLM_OPENVINO_CPU_BIND_THREAD", None)),

    # Number of CPU inference streams.
    # AUTO keeps OpenVINO heuristic. Numeric values force explicit streams.
    "VLLM_OPENVINO_NUM_STREAMS":
    lambda: (lambda v: int(v) if v.isdigit() else v.upper())(
        os.getenv("VLLM_OPENVINO_NUM_STREAMS", "AUTO")),

    # CPU-only: enable/disable hyperthreading. When disabled, uses 1 thread
    # per physical core instead of 2 (useful on oversubscription-prone systems).
    "VLLM_OPENVINO_ENABLE_HYPER_THREADING":
    lambda: None if os.getenv("VLLM_OPENVINO_ENABLE_HYPER_THREADING", "").lower() in ("", "auto") else
            os.getenv("VLLM_OPENVINO_ENABLE_HYPER_THREADING", "true").lower() == "true",

    # CPU-only: inference precision hint (f32, f16, bf16). Forces specific
    # precision for matmul operations. On CPUs without int8 acceleration, this
    # can avoid expensive int8->fp dequantization overhead.
    "VLLM_OPENVINO_INFERENCE_PRECISION":
    lambda: os.getenv("VLLM_OPENVINO_INFERENCE_PRECISION", None),

    # CPU-only: enable/disable CPU core pinning. When enabled, threads are
    # pinned to specific CPU cores to avoid migration penalties.
    "VLLM_OPENVINO_ENABLE_CPU_PINNING":
    lambda: None if os.getenv("VLLM_OPENVINO_ENABLE_CPU_PINNING", "").lower() in ("", "auto") else
            os.getenv("VLLM_OPENVINO_ENABLE_CPU_PINNING", "true").lower() == "true",

    # Memory overhead threshold for KV cache allocation
    # (1.1 = 10% overhead for internal OpenVINO allocations)
    "VLLM_OPENVINO_MEMORY_THRESHOLD":
    lambda: float(os.getenv("VLLM_OPENVINO_MEMORY_THRESHOLD", "1.1")),

    # Block size for CPU device (default matches vLLM CPU default)
    "VLLM_OPENVINO_CPU_BLOCK_SIZE":
    lambda: int(os.getenv("VLLM_OPENVINO_CPU_BLOCK_SIZE", "32")),

    # Block size for GPU device (default matches vLLM GPU default)
    "VLLM_OPENVINO_GPU_BLOCK_SIZE":
    lambda: int(os.getenv("VLLM_OPENVINO_GPU_BLOCK_SIZE", "16")),
}

# end-env-vars-definition

def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
