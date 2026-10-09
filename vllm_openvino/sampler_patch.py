# SPDX-License-Identifier: Apache-2.0
"""CPU sampling fast paths for vLLM V1.

At decode, sampling runs on the critical path of every step and, for small
models, costs as much as the OpenVINO inference itself. Upstream's CPU
implementations are tuned for neither the batch nor the vocabulary size seen
here (batch <= 128, vocab 65k-250k), so three of them are replaced:

* `compiled_random_sample`: upstream draws a full (batch, vocab) buffer of
  exponential noise (Gumbel-max). A categorical draw needs one uniform number
  per row: softmax -> cumsum -> searchsorted(right=True) is ~7x faster at
  (8, 151936) and samples the same distribution. `right=True` returns the
  first index whose cumulative mass is strictly greater than the draw, so
  zero-probability (masked) tokens are never selected; the draw is scaled by
  the row's own cumulative total, so float32 rounding cannot push it past the
  last bucket.
* `apply_top_k_top_p` with top-p: upstream sorts the entire vocabulary. Top-k
  and the nucleus only ever keep a handful of tokens, so a partial `topk` over
  the candidates is enough. Falls back to upstream when the candidates cannot
  cover the nucleus or k is large.
* `apply_all_penalties`: upstream builds dense (batch, vocab) bin-count and
  mask tensors every step. Repetition penalty only touches tokens that
  appeared in the prompt or output, so it is applied on those positions only.
  Batches that use presence/frequency penalties take the upstream path.

Determinism: `forward_cpu` calls `compiled_random_sample` only when the
batch has no per-request `torch.Generator`. A batch containing any seeded
request takes the explicit per-generator path instead, so per-request
seeds keep their stock torch semantics.
"""
from __future__ import annotations

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Nucleus candidates when no top-k is set. Falls back to upstream when the
# top candidates hold less than `p` of the probability mass.
_NUCLEUS_CANDIDATES = 1024


def openvino_random_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    cdf = probs.cumsum(dim=-1)
    draw = torch.rand(cdf.shape[0], 1) * cdf[:, -1:]
    return torch.searchsorted(cdf, draw, right=True).clamp_(
        max=cdf.shape[-1] - 1).view(-1)


def _make_top_k_top_p(upstream):
    def openvino_apply_top_k_top_p(
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> torch.Tensor:
        # Top-k alone is already a partial topk upstream.
        if p is None:
            return upstream(logits, k, p)
        vocab_size = logits.shape[1]
        num_cand = (min(int(k.max()), vocab_size) if k is not None
                    else _NUCLEUS_CANDIDATES)
        if num_cand * 4 >= vocab_size:
            return upstream(logits, k, p)

        vals, idx = logits.topk(num_cand, dim=-1)
        if k is not None:
            beyond_k = (torch.arange(num_cand)[None, :] >= k.long()[:, None])
            vals.masked_fill_(beyond_k, -float("inf"))
            probs = vals.softmax(dim=-1)
        else:
            probs = (vals - logits.logsumexp(dim=-1, keepdim=True)).exp()
            if bool(((probs.sum(dim=-1) < p) & (p < 1)).any()):
                return upstream(logits, k, p)

        # A token is dropped once the mass ranked above it already reaches p.
        above = probs.cumsum(dim=-1) - probs
        unfiltered = p >= 1
        drop = (above >= p[:, None]) & ~unfiltered[:, None]
        drop[:, 0] = False
        vals.masked_fill_(drop, -float("inf"))
        # Without top-k, rows with p=1 keep the whole vocabulary.
        keep_rows = (logits.clone() if k is None and bool(unfiltered.any())
                     else None)
        out = logits.fill_(-float("inf")).scatter_(-1, idx, vals)
        if keep_rows is not None:
            out = torch.where(unfiltered[:, None], keep_rows, out)
        return out

    return openvino_apply_top_k_top_p


def _make_apply_all_penalties(upstream):
    def openvino_apply_all_penalties(
        logits: torch.Tensor,
        prompt_token_ids: torch.Tensor,
        presence_penalties: torch.Tensor,
        frequency_penalties: torch.Tensor,
        repetition_penalties: torch.Tensor,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        if (bool((presence_penalties != 0).any())
                or bool((frequency_penalties != 0).any())):
            return upstream(logits, prompt_token_ids, presence_penalties,
                            frequency_penalties, repetition_penalties,
                            output_token_ids)

        num_seqs, vocab_size = logits.shape
        token_ids = prompt_token_ids
        max_out = max((len(o) for o in output_token_ids), default=0)
        if max_out:
            out = torch.full((num_seqs, max_out), vocab_size,
                             dtype=torch.int64)
            for i, row in enumerate(output_token_ids):
                if row:
                    out[i, :len(row)] = torch.tensor(row, dtype=torch.int64)
            token_ids = torch.cat([prompt_token_ids, out], dim=1)

        # Padding is vocab_size; async scheduling placeholders are -1.
        rows, cols = ((token_ids >= 0) & (token_ids < vocab_size)).nonzero(
            as_tuple=True)
        tokens = token_ids[rows, cols]
        vals = logits[rows, tokens]
        rep = repetition_penalties[rows]
        logits[rows, tokens] = torch.where(vals > 0, vals * (1.0 / rep),
                                           vals * rep)
        return logits

    return openvino_apply_all_penalties


def install() -> None:
    from vllm.v1.sample import sampler
    from vllm.v1.sample.ops import topk_topp_sampler as tk

    tk.compiled_random_sample = openvino_random_sample
    tk.apply_top_k_top_p = _make_top_k_top_p(tk.apply_top_k_top_p)
    sampler.apply_all_penalties = _make_apply_all_penalties(
        sampler.apply_all_penalties)
    logger.info(
        "[OV-SAMPLER] Installed CPU fast paths (inverse-CDF sample, partial "
        "top-k/top-p, sparse repetition penalty) "
        "(VLLM_OPENVINO_FAST_SAMPLER=1). Set VLLM_OPENVINO_FAST_SAMPLER=0 "
        "to fall back to stock vLLM.")
