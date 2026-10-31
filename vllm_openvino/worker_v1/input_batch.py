# SPDX-License-Identifier: Apache-2.0
"""Persistent batch state trimmed from vLLM 0.32's `InputBatch` and
`CachedRequestState` (`vllm/v1/worker/gpu_input_batch.py`), which go away with
Model Runner V1. Rows are filled and compacted (`add_request`, `condense`)
exactly as upstream, so request order is unchanged; only the state the runner
and `sampler.py` read is kept (no block table, LoRA, pooling, prompt embeds or
async-scheduling placeholders)."""
import heapq
from dataclasses import dataclass, field

import numpy as np
import torch
from vllm.sampling_params import SamplingParams, SamplingType

from vllm_openvino.sampler import SamplingMetadata


@dataclass
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int]
    mm_features: list
    sampling_params: SamplingParams
    generator: torch.Generator | None
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int
    output_token_ids: list[int] = field(default_factory=list)

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


class InputBatch:

    def __init__(self, max_num_reqs: int, max_model_len: int,
                 vocab_size: int):
        self.max_num_reqs = max_num_reqs
        self.vocab_size = vocab_size

        self._req_ids: list[str | None] = []
        self.req_id_to_index: dict[str, int] = {}
        self.req_output_token_ids: list[list[int] | None] = []
        self.token_ids_cpu = np.zeros((max_num_reqs, max_model_len),
                                      dtype=np.int32)
        self.num_tokens_no_spec = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, dtype=np.int32)

        def row_tensor(dtype):
            tensor = torch.empty(max_num_reqs, dtype=dtype)
            return tensor, tensor.numpy()

        self.temperature, self.temperature_cpu = row_tensor(torch.float32)
        self.top_p, self.top_p_cpu = row_tensor(torch.float32)
        self.top_k, self.top_k_cpu = row_tensor(torch.int32)
        self.frequency_penalties, self.frequency_penalties_cpu = row_tensor(
            torch.float32)
        self.presence_penalties, self.presence_penalties_cpu = row_tensor(
            torch.float32)
        self.repetition_penalties, self.repetition_penalties_cpu = row_tensor(
            torch.float32)
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()
        self.top_p_reqs: set[str] = set()
        self.top_k_reqs: set[str] = set()
        self.frequency_penalties_reqs: set[str] = set()
        self.presence_penalties_reqs: set[str] = set()
        self.repetition_penalties_reqs: set[str] = set()

        # Only rows whose request has its own seed.
        self.generators: dict[int, torch.Generator] = {}
        self.num_logprobs: dict[str, int] = {}
        self.logprob_token_ids: dict[str, list[int]] = {}
        self.has_allowed_token_ids: set[str] = set()
        # True marks tokens to mask; allocated on first use (max_reqs x vocab).
        self.allowed_token_ids_mask: torch.Tensor | None = None
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        # Rows freed by remove_request(), refilled smallest first.
        self._removed: list[int] = []
        self._batch_changed = False
        self.sampling_metadata = self._make_sampling_metadata()

    @property
    def req_ids(self) -> list[str]:
        return self._req_ids  # type: ignore[return-value]

    @property
    def num_reqs(self) -> int:
        return len(self.req_id_to_index)

    def add_request(self, request: CachedRequestState) -> int:
        req_index = (heapq.heappop(self._removed) if self._removed
                     else self.num_reqs)
        self._batch_changed = True
        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids
        self.req_id_to_index[req_id] = req_index

        num_prompt_tokens = len(request.prompt_token_ids)
        end_idx = num_prompt_tokens + len(request.output_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens
        self.token_ids_cpu[req_index, :num_prompt_tokens] = request.prompt_token_ids
        self.token_ids_cpu[req_index, num_prompt_tokens:end_idx] = (
            request.output_token_ids)
        self.num_tokens_no_spec[req_index] = request.num_tokens
        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens

        params = request.sampling_params
        if params.sampling_type == SamplingType.GREEDY:
            self.temperature_cpu[req_index] = 0.0
            self.greedy_reqs.add(req_id)
        else:
            self.temperature_cpu[req_index] = params.temperature
            self.random_reqs.add(req_id)
        self.top_p_cpu[req_index] = params.top_p
        if params.top_p < 1:
            self.top_p_reqs.add(req_id)
        top_k = params.top_k
        if 0 < top_k < self.vocab_size:
            self.top_k_reqs.add(req_id)
        else:
            top_k = self.vocab_size
        self.top_k_cpu[req_index] = top_k
        self.frequency_penalties_cpu[req_index] = params.frequency_penalty
        if params.frequency_penalty != 0.0:
            self.frequency_penalties_reqs.add(req_id)
        self.presence_penalties_cpu[req_index] = params.presence_penalty
        if params.presence_penalty != 0.0:
            self.presence_penalties_reqs.add(req_id)
        self.repetition_penalties_cpu[req_index] = params.repetition_penalty
        if params.repetition_penalty != 1.0:
            self.repetition_penalties_reqs.add(req_id)

        if request.generator is not None:
            self.generators[req_index] = request.generator
        if params.logprobs is not None:
            self.num_logprobs[req_id] = (self.vocab_size if params.logprobs == -1
                                         else params.logprobs)
        if params.logprob_token_ids is not None:
            self.logprob_token_ids[req_id] = params.logprob_token_ids
        if params.allowed_token_ids:
            self.has_allowed_token_ids.add(req_id)
            if self.allowed_token_ids_mask is None:
                self.allowed_token_ids_mask = torch.zeros(
                    self.max_num_reqs, self.vocab_size, dtype=torch.bool)
            self.allowed_token_ids_mask[req_index] = True
            self.allowed_token_ids_mask[req_index][
                params.allowed_token_ids] = False
        if params.bad_words_token_ids:
            self.bad_words_token_ids[req_index] = params.bad_words_token_ids
        return req_index

    def remove_request(self, req_id: str) -> int | None:
        """Must be followed by condense()."""
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is None:
            return None
        heapq.heappush(self._removed, req_index)
        self._batch_changed = True
        self._req_ids[req_index] = None
        self.req_output_token_ids[req_index] = None

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.logprob_token_ids.pop(req_id, None)
        self.has_allowed_token_ids.discard(req_id)
        if self.allowed_token_ids_mask is not None:
            self.allowed_token_ids_mask[req_index].fill_(False)
        self.bad_words_token_ids.pop(req_index, None)
        return req_index

    def condense(self) -> None:
        """Move the highest occupied rows into the lowest free rows."""
        num_reqs = self.num_reqs
        if not self._removed:
            return
        if num_reqs == 0:
            self._req_ids.clear()
            self.req_output_token_ids.clear()
            return

        last_req_index = num_reqs + len(self._removed) - 1
        while self._removed:
            while last_req_index in self._removed:
                last_req_index -= 1
            empty_index = self._removed[0]
            if empty_index >= last_req_index:
                break
            heapq.heappop(self._removed)

            req_id = self._req_ids[last_req_index]
            self._req_ids[empty_index] = req_id
            self._req_ids[last_req_index] = None
            self.req_output_token_ids[empty_index] = (
                self.req_output_token_ids[last_req_index])
            self.req_output_token_ids[last_req_index] = None
            self.req_id_to_index[req_id] = empty_index

            num_tokens = int(self.num_tokens_no_spec[last_req_index])
            self.token_ids_cpu[empty_index, :num_tokens] = (
                self.token_ids_cpu[last_req_index, :num_tokens])
            for array in (self.num_tokens_no_spec, self.num_prompt_tokens,
                          self.num_computed_tokens_cpu, self.temperature_cpu,
                          self.top_p_cpu, self.top_k_cpu,
                          self.frequency_penalties_cpu,
                          self.presence_penalties_cpu,
                          self.repetition_penalties_cpu):
                array[empty_index] = array[last_req_index]
            generator = self.generators.pop(last_req_index, None)
            if generator is not None:
                self.generators[empty_index] = generator
            if self.allowed_token_ids_mask is not None:
                self.allowed_token_ids_mask[empty_index] = (
                    self.allowed_token_ids_mask[last_req_index])
                self.allowed_token_ids_mask[last_req_index].fill_(False)
            bad_words = self.bad_words_token_ids.pop(last_req_index, None)
            if bad_words is not None:
                self.bad_words_token_ids[empty_index] = bad_words
            last_req_index -= 1

        del self._req_ids[num_reqs:]
        del self.req_output_token_ids[num_reqs:]

    def refresh_metadata(self) -> None:
        """Rebuild the sampling metadata if requests were added or removed."""
        self._removed.clear()
        if self._batch_changed:
            self._batch_changed = False
            self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:
        num_reqs = self.num_reqs
        no_penalties = not (self.presence_penalties_reqs
                            or self.frequency_penalties_reqs
                            or self.repetition_penalties_reqs)
        prompt_token_ids = None
        if not no_penalties:
            max_prompt_len = self.num_prompt_tokens[:num_reqs].max()
            prompt = self.token_ids_cpu[:num_reqs, :max_prompt_len].astype(
                np.int64)
            for i in range(num_reqs):
                prompt[i, self.num_prompt_tokens[i]:] = self.vocab_size
            prompt_token_ids = torch.from_numpy(prompt)

        logprob_token_ids = None
        if self.logprob_token_ids:
            logprob_token_ids = {
                self.req_id_to_index[req_id]: token_ids
                for req_id, token_ids in self.logprob_token_ids.items()
                if req_id in self.req_id_to_index
            }

        return SamplingMetadata(
            temperature=(None if not self.random_reqs
                         else self.temperature[:num_reqs].clone()),
            all_greedy=not self.random_reqs,
            all_random=not self.greedy_reqs,
            top_p=self.top_p[:num_reqs].clone() if self.top_p_reqs else None,
            top_k=self.top_k[:num_reqs].clone() if self.top_k_reqs else None,
            generators=self.generators,
            max_num_logprobs=(max(self.num_logprobs.values())
                              if self.num_logprobs else None),
            no_penalties=no_penalties,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[:num_reqs].clone(),
            presence_penalties=self.presence_penalties[:num_reqs].clone(),
            repetition_penalties=self.repetition_penalties[:num_reqs].clone(),
            output_token_ids=(self.req_output_token_ids
                              if not no_penalties or self.bad_words_token_ids
                              else []),
            allowed_token_ids_mask=(self.allowed_token_ids_mask[:num_reqs]
                                    if self.has_allowed_token_ids else None),
            bad_words_token_ids=self.bad_words_token_ids,
            logprob_token_ids=logprob_token_ids,
        )
