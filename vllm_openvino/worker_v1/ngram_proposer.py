# SPDX-License-Identifier: Apache-2.0
"""N-gram drafter trimmed from vLLM 0.32's `NgramProposer`
(`vllm/v1/spec_decode/ngram_proposer.py`), which goes away with Model Runner
V1. Upstream's numba thread count is capped at 1, so the batch loop here is
single-threaded."""
import numpy as np
from numba import njit
from vllm.config import VllmConfig


class NgramProposer:

    def __init__(self, vllm_config: VllmConfig):
        spec_config = vllm_config.speculative_config
        self.min_n = spec_config.prompt_lookup_min
        self.max_n = spec_config.prompt_lookup_max
        self.k = spec_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._drafts = np.zeros((max_num_seqs, self.k), dtype=np.int32)
        self._num_drafts = np.zeros(max_num_seqs, dtype=np.int32)
        # JIT-compile for the InputBatch dtypes now, not on the first decode.
        self.propose(self.k, [[0]], np.zeros(1, dtype=np.int32),
                     np.zeros((1, self.max_n + 1), dtype=np.int32))

    def propose(
        self,
        num_speculative_tokens: int,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        valid = np.array([
            i for i, sampled in enumerate(sampled_token_ids)
            if sampled and num_tokens_no_spec[i] < self.max_model_len
        ], dtype=np.int64)
        drafts: list[list[int]] = [[] for _ in sampled_token_ids]
        if not len(valid):
            return drafts
        _batch_propose(valid, num_tokens_no_spec, token_ids_cpu, self.min_n,
                       self.max_n, self.max_model_len, num_speculative_tokens,
                       self._drafts, self._num_drafts)
        for i in valid:
            if num := self._num_drafts[i]:
                drafts[i] = self._drafts[i, :num].tolist()
        return drafts


@njit
def _batch_propose(valid, num_tokens_no_spec, token_ids_cpu, min_n, max_n,
                   max_model_len, k, drafts, num_drafts):
    for idx in valid:
        output = _find_longest_matched_ngram_and_propose_tokens(
            token_ids_cpu[idx, :num_tokens_no_spec[idx]], min_n, max_n,
            max_model_len, k)
        num_drafts[idx] = output.shape[0]
        if output.shape[0]:
            drafts[idx, :output.shape[0]] = output


@njit
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """Find the longest n-gram within [min_ngram, max_ngram] that matches the
    suffix of the tokens and return up to k tokens that followed it."""
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0, ), dtype=origin_tokens.dtype)

    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0, ), dtype=origin_tokens.dtype)

    # On the reversed tokens, find the rightmost longest prefix match (KMP),
    # keeping LPS only for the first max_ngram prefix.
    tokens = origin_tokens[::-1]
    lps = np.zeros(max_ngram, dtype=np.int32)
    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1
    while i < total_token:
        if tokens[prev_lps] == tokens[i]:
            prev_lps += 1
            # `>=` picks the earliest occurrence in the original order.
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1

    if longest_ngram < min_ngram:
        return np.empty((0, ), dtype=origin_tokens.dtype)

    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position:start_position + k]
