# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_OPENVINO_DEVICE: str = "CPU"
    VLLM_OPENVINO_KVCACHE_SPACE: int = 32
    VLLM_OPENVINO_KV_CACHE_PRECISION: Optional[str] = None

environment_variables: dict[str, Callable[[], Any]] = {
    # OpenVINO device selection
    # default is CPU
    "VLLM_OPENVINO_DEVICE":
    lambda: os.getenv("VLLM_OPENVINO_DEVICE", "CPU").upper(),

    # OpenVINO key-value cache space
    # default is 32GB for Gemma-4 compatibility
    "VLLM_OPENVINO_KVCACHE_SPACE":
    lambda: int(os.getenv("VLLM_OPENVINO_KVCACHE_SPACE", "32")),

    # OpenVINO KV cache precision
    # default 'undefined', which means plugin will automatically set
    # proper value based on model analysis
    "VLLM_OPENVINO_KV_CACHE_PRECISION":
    lambda: os.getenv("VLLM_OPENVINO_KV_CACHE_PRECISION", None),
}

# end-env-vars-definition

def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
