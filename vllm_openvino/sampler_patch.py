# SPDX-License-Identifier: Apache-2.0
"""Numpy PCG64 replacement for vLLM V1's compiled_random_sample.

Upstream `vllm.v1.sample.ops.topk_topp_sampler.compiled_random_sample`
draws Gumbel-max noise via `torch.empty_like(probs).exponential_()`,
which is a single-threaded scalar inverse-CDF loop on CPU. On the
(batch, vocab) shapes this plugin sees at decode (e.g. (8, 65536) for
LFM2.5-350M, (8, 151936) for Qwen3-1.7B), numpy PCG64's
`standard_exponential(dtype=fp32)` fills the same buffer ~4x faster
(see task-14 microbench). The rest of the Gumbel-max math
(softmax -> div -> argmax) is unchanged.

MKL acceleration: when `mkl_random` is importable (requires the MKL shared
libs), the exponential draw uses `mkl_random.RandomState.standard_exponential`
which is ~2x faster than numpy PCG64 on the shapes this plugin sees. The
container image does not bundle MKL; users who install mkl + mkl-random +
intel-openmp into their custom image get the faster path automatically.

Determinism: the seeded-per-request path in `forward_cpu` uses
`torch.Generator` on each row of `q` explicitly and is not routed
through `compiled_random_sample`, so per-request seeds keep their
stock torch semantics. This patch only affects the batches where at
least one request has no seed (`len(generators) != batch_size`).
"""
from __future__ import annotations

import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# MKL is ~2x faster than numpy PCG64 on the (batch, vocab) shapes this
# plugin sees at decode, but the container image does not bundle it. Try
# mkl_random first; fall back to numpy PCG64 when unavailable.
# mkl_random requires the MKL shared libs (libmkl_rt.so.2) at runtime; if
# those are missing the import raises ImportError or OSError, which we
# catch here along with any other init exception.
try:
    from mkl_random import RandomState as _MKL_RNG
    _mkl_rng = _MKL_RNG()
    _HAS_MKL = True
except Exception:
    # Broader than ImportError: covers OSError from dlopen, RuntimeError
    # from broken MKL runtime, etc. Uncatchable symbol-lookup aborts at
    # import (mkl_random present but libmkl_rt/libiomp5 missing) still
    # kill the process — that's a broken pip install, not our concern.
    _mkl_rng = None
    _HAS_MKL = False

_rng = np.random.default_rng()


def openvino_random_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    if _HAS_MKL:
        q_np = _mkl_rng.standard_exponential(size=tuple(probs.shape)).astype(np.float32)
    else:
        q_np = _rng.standard_exponential(size=tuple(probs.shape), dtype=np.float32)
    q = torch.from_numpy(q_np)
    return probs.div(q).argmax(dim=-1).view(-1)


def install() -> None:
    from vllm.v1.sample.ops import topk_topp_sampler as tk
    tk.compiled_random_sample = openvino_random_sample
    if _HAS_MKL:
        logger.info(
            "[OV-SAMPLER] Installed MKL fast-path replacement for "
            "vllm.v1.sample.ops.topk_topp_sampler.compiled_random_sample "
            "(VLLM_OPENVINO_FAST_SAMPLER=1).")
    else:
        logger.info(
            "[OV-SAMPLER] Installed numpy PCG64 fast-path replacement for "
            "vllm.v1.sample.ops.topk_topp_sampler.compiled_random_sample "
            "(VLLM_OPENVINO_FAST_SAMPLER=1). Set VLLM_OPENVINO_FAST_SAMPLER=0 "
            "to fall back to stock torch.exponential_().")
