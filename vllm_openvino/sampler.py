# SPDX-License-Identifier: Apache-2.0
"""CPU sampler for the OpenVINO runner.

Follows vLLM 0.32's V1 `Sampler` with `raw_logprobs` step for step (logprobs
from the unprocessed logits, allowed-token mask, bad words, penalties,
greedy/temperature, top-k/top-p, random draw), so the plugin does not depend
on the V1 sampler that vLLM is removing with Model Runner V1.

At decode, sampling runs on the critical path of every step and, for small
models, costs as much as the OpenVINO inference itself. With
`VLLM_OPENVINO_FAST_SAMPLER=1` two routines whose upstream cost scales with
the full (batch, vocab) tensor use partial equivalents:

* top-p: upstream sorts the entire vocabulary. Top-k and the nucleus only
  ever keep a handful of tokens, so a partial `topk` over the candidates is
  enough. Falls back to the full sort when the candidates cannot cover the
  nucleus or k is large.
* repetition penalty: upstream builds dense (batch, vocab) bin-count and mask
  tensors every step. Repetition penalty only touches tokens that appeared in
  the prompt or output, so it is applied on those positions only. Batches
  that use presence/frequency penalties take the dense path.

The random draw matches vLLM's CPU path (`forward_cpu`): one seed per row,
taken from the request's `torch.Generator` when the request has a seed. The
C++ Gumbel kernel draws per-token noise over the whole vocabulary; a
categorical draw needs one uniform number per row: softmax -> cumsum ->
searchsorted(right=True) samples the same distribution. `right=True` returns
the first index whose cumulative mass is strictly greater than the draw, so
zero-probability (masked) tokens are never selected; the draw is scaled by
the row's own cumulative total, so float32 rounding cannot push it past the
last bucket.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from vllm.v1.outputs import LogprobsTensors, SamplerOutput

from vllm_openvino import envs

_SAMPLING_EPS = 1e-5
# Nucleus candidates when no top-k is set. Falls back to the full sort when
# the top candidates hold less than `p` of the probability mass.
_NUCLEUS_CANDIDATES = 1024


@dataclass
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool
    top_p: torch.Tensor | None
    top_k: torch.Tensor | None
    # Only rows whose request has its own seed.
    generators: dict[int, torch.Generator]
    # None: no logprobs; 0: sampled token only.
    max_num_logprobs: int | None
    no_penalties: bool
    # Padded with vocab_size; set only when penalties apply.
    prompt_token_ids: torch.Tensor | None
    frequency_penalties: torch.Tensor
    presence_penalties: torch.Tensor
    repetition_penalties: torch.Tensor
    output_token_ids: list[list[int]]
    # True marks tokens to mask.
    allowed_token_ids_mask: torch.Tensor | None
    bad_words_token_ids: dict[int, list[list[int]]]
    logprob_token_ids: dict[int, list[int]] | None = None


class OpenVINOSampler:

    def __init__(self) -> None:
        self.fast = envs.VLLM_OPENVINO_FAST_SAMPLER

    def __call__(self, logits: torch.Tensor,
                 sampling_metadata: SamplingMetadata) -> SamplerOutput:
        md = sampling_metadata
        num_logprobs = md.max_num_logprobs
        raw_logprobs = None
        if num_logprobs is not None or md.logprob_token_ids:
            raw_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)

        logits = self._apply_logits_processors(logits.to(torch.float32), md)
        sampled = self._sample(logits, md).long()

        token_ids_logprobs = None
        if md.logprob_token_ids:
            token_ids_logprobs = _gather_specific_token_logprobs(
                raw_logprobs, md.logprob_token_ids, sampled)

        if token_ids_logprobs is not None:
            logprobs_tensors = token_ids_logprobs
        elif num_logprobs is None:
            logprobs_tensors = None
        elif num_logprobs == -1:
            logprobs_tensors = LogprobsTensors(torch.empty(0), raw_logprobs,
                                               torch.empty(0))
        else:
            logprobs_tensors = _gather_logprobs(raw_logprobs, num_logprobs,
                                                sampled)

        return SamplerOutput(
            sampled_token_ids=sampled.to(torch.int32).unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )

    def _apply_logits_processors(self, logits: torch.Tensor,
                                 md: SamplingMetadata) -> torch.Tensor:
        if md.allowed_token_ids_mask is not None:
            logits.masked_fill_(md.allowed_token_ids_mask, float("-inf"))
        if md.bad_words_token_ids:
            _apply_bad_words(logits, md.bad_words_token_ids,
                             md.output_token_ids)
        if not md.no_penalties:
            logits = self._apply_penalties(logits, md)
        return logits

    def _sample(self, logits: torch.Tensor,
                md: SamplingMetadata) -> torch.Tensor:
        greedy = None
        if not md.all_random:
            greedy = logits.argmax(dim=-1).view(-1)
            if md.all_greedy:
                return greedy

        temp = md.temperature
        if not md.all_random:
            temp = torch.where(temp < _SAMPLING_EPS, 1.0, temp)
        logits = logits.div_(temp.unsqueeze(dim=1))
        logits = self._apply_top_k_top_p(logits, md.top_k, md.top_p)
        sampled = _random_sample(logits, md.generators)
        if greedy is None:
            return sampled
        return torch.where(md.temperature < _SAMPLING_EPS, greedy, sampled,
                           out=greedy)

    def _apply_top_k_top_p(self, logits: torch.Tensor, k: torch.Tensor | None,
                           p: torch.Tensor | None) -> torch.Tensor:
        if p is None:
            return logits if k is None else _apply_top_k_only(logits, k)
        if self.fast:
            out = _apply_top_k_top_p_partial(logits, k, p)
            if out is not None:
                return out
        return _apply_top_k_top_p_sort(logits, k, p)

    def _apply_penalties(self, logits: torch.Tensor,
                         md: SamplingMetadata) -> torch.Tensor:
        if (self.fast and not bool((md.presence_penalties != 0).any())
                and not bool((md.frequency_penalties != 0).any())):
            return _apply_repetition_penalty_sparse(
                logits, md.prompt_token_ids, md.repetition_penalties,
                md.output_token_ids)
        return _apply_penalties_dense(logits, md.prompt_token_ids,
                                      md.presence_penalties,
                                      md.frequency_penalties,
                                      md.repetition_penalties,
                                      md.output_token_ids)


def _random_sample(logits: torch.Tensor,
                   generators: dict[int, torch.Generator]) -> torch.Tensor:
    seeds = torch.randint(0, 2**31, (logits.shape[0], ), dtype=torch.long)
    for i, gen in generators.items():
        seeds[i] = torch.randint(0, 2**31, (1, ), generator=gen,
                                 dtype=torch.long).item()
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    cdf = probs.cumsum(dim=-1)
    draw = torch.empty(cdf.shape[0], 1)
    generator = torch.Generator()
    for row, seed in enumerate(seeds.tolist()):
        draw[row] = torch.rand(1, generator=generator.manual_seed(seed))
    return torch.searchsorted(cdf, draw * cdf[:, -1:], right=True).clamp_(
        max=cdf.shape[-1] - 1).view(-1)


def _apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    no_top_k_mask = k == logits.shape[1]
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    return logits.masked_fill_(logits < top_k_mask, -float("inf"))


def _apply_top_k_top_p_sort(logits: torch.Tensor, k: torch.Tensor | None,
                            p: torch.Tensor) -> torch.Tensor:
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        logits_sort.masked_fill_(logits_sort < top_k_mask, -float("inf"))
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, -float("inf"))
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)


def _apply_top_k_top_p_partial(logits: torch.Tensor, k: torch.Tensor | None,
                               p: torch.Tensor) -> torch.Tensor | None:
    vocab_size = logits.shape[1]
    num_cand = (min(int(k.max()), vocab_size) if k is not None
                else _NUCLEUS_CANDIDATES)
    if num_cand * 4 >= vocab_size:
        return None

    vals, idx = logits.topk(num_cand, dim=-1)
    if k is not None:
        beyond_k = (torch.arange(num_cand)[None, :] >= k.long()[:, None])
        vals.masked_fill_(beyond_k, -float("inf"))
        probs = vals.softmax(dim=-1)
    else:
        probs = (vals - logits.logsumexp(dim=-1, keepdim=True)).exp()
        if bool(((probs.sum(dim=-1) < p) & (p < 1)).any()):
            return None

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


def _pad_output_tokens(output_token_ids: list[list[int]],
                       pad: int) -> torch.Tensor:
    max_out = max((len(o) for o in output_token_ids), default=0)
    out = torch.full((len(output_token_ids), max_out), pad, dtype=torch.int64)
    for i, row in enumerate(output_token_ids):
        if row:
            out[i, :len(row)] = torch.tensor(row, dtype=torch.int64)
    return out


def _apply_repetition_penalty_sparse(
        logits: torch.Tensor, prompt_token_ids: torch.Tensor,
        repetition_penalties: torch.Tensor,
        output_token_ids: list[list[int]]) -> torch.Tensor:
    vocab_size = logits.shape[1]
    token_ids = torch.cat(
        [prompt_token_ids,
         _pad_output_tokens(output_token_ids, vocab_size)], dim=1)
    # Padding is vocab_size; async scheduling placeholders are -1.
    rows, cols = ((token_ids >= 0) & (token_ids < vocab_size)).nonzero(
        as_tuple=True)
    tokens = token_ids[rows, cols]
    vals = logits[rows, tokens]
    rep = repetition_penalties[rows]
    logits[rows, tokens] = torch.where(vals > 0, vals * (1.0 / rep),
                                       vals * rep)
    return logits


def _token_bin_counts(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    bin_counts = torch.zeros((tokens.shape[0], vocab_size + 1),
                             dtype=torch.long)
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    return bin_counts[:, :vocab_size]


def _apply_penalties_dense(logits: torch.Tensor, prompt_token_ids: torch.Tensor,
                           presence_penalties: torch.Tensor,
                           frequency_penalties: torch.Tensor,
                           repetition_penalties: torch.Tensor,
                           output_token_ids: list[list[int]]) -> torch.Tensor:
    vocab_size = logits.shape[1]
    output_tokens = _pad_output_tokens(output_token_ids, vocab_size)
    output_tokens.masked_fill_(output_tokens == -1, vocab_size)
    prompt_mask = _token_bin_counts(prompt_token_ids, vocab_size) > 0
    output_bin_counts = _token_bin_counts(output_tokens, vocab_size)
    output_mask = output_bin_counts > 0

    penalties = torch.where(prompt_mask | output_mask,
                            repetition_penalties.unsqueeze(dim=1), 1.0)
    logits *= torch.where(logits > 0, 1.0 / penalties, penalties)
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def _apply_bad_words(logits: torch.Tensor,
                     bad_words_token_ids: dict[int, list[list[int]]],
                     past_tokens_ids: list[list[int]]) -> None:
    for i, bad_words_ids in bad_words_token_ids.items():
        past = past_tokens_ids[i]
        for bad_word_ids in bad_words_ids:
            if len(bad_word_ids) > len(past) + 1:
                continue
            prefix_length = len(bad_word_ids) - 1
            actual_prefix = past[-prefix_length:] if prefix_length > 0 else []
            if actual_prefix == bad_word_ids[:prefix_length]:
                logits[i, bad_word_ids[-1]] = float("-inf")


def _gather_logprobs(logprobs: torch.Tensor, num_logprobs: int,
                     token_ids: torch.Tensor) -> LogprobsTensors:
    topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)
    token_ids = token_ids.unsqueeze(-1)
    token_logprobs = logprobs.gather(-1, token_ids)
    token_ranks = (logprobs >= token_logprobs).sum(-1)
    indices = torch.cat((token_ids, topk_indices), dim=1).to(torch.int32)
    return LogprobsTensors(indices,
                           torch.cat((token_logprobs, topk_logprobs), dim=1),
                           token_ranks)


def _gather_specific_token_logprobs(
        logprobs: torch.Tensor, logprob_token_ids: dict[int, list[int]],
        sampled: torch.Tensor) -> LogprobsTensors:
    batch_size = logprobs.shape[0]
    max_num_tokens = max(len(tids) for tids in logprob_token_ids.values())
    token_ids = torch.zeros(batch_size, max_num_tokens + 1, dtype=torch.int64)
    valid_mask = torch.zeros(batch_size, max_num_tokens + 1, dtype=torch.bool)
    valid_mask[:, 0] = True
    for req_idx, tids in logprob_token_ids.items():
        token_ids[req_idx, 1:len(tids) + 1] = torch.as_tensor(
            tids, dtype=torch.int64)
        valid_mask[req_idx, 1:len(tids) + 1] = True
    token_ids[:, 0] = sampled

    gathered = logprobs.gather(-1, token_ids).masked_fill(
        ~valid_mask, float("-inf"))
    sampled_logprobs = logprobs.gather(-1, sampled.unsqueeze(-1))
    token_ranks = (logprobs >= sampled_logprobs).sum(-1)
    return LogprobsTensors(
        logprob_token_ids=token_ids.to(torch.int32),
        logprobs=gathered,
        selected_token_ranks=token_ranks,
    )
