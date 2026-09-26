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

_rng = np.random.default_rng()


def openvino_random_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q_np = _rng.standard_exponential(size=tuple(probs.shape),
                                     dtype=np.float32)
    q = torch.from_numpy(q_np)
    return probs.div(q).argmax(dim=-1).view(-1)


def install() -> None:
    from vllm.v1.sample.ops import topk_topp_sampler as tk
    tk.compiled_random_sample = openvino_random_sample
    logger.info(
        "[OV-SAMPLER] Installed numpy PCG64 fast-path replacement for "
        "vllm.v1.sample.ops.topk_topp_sampler.compiled_random_sample "
        "(VLLM_OPENVINO_FAST_SAMPLER=1). Set VLLM_OPENVINO_FAST_SAMPLER=0 "
        "to fall back to stock torch.exponential_().")
