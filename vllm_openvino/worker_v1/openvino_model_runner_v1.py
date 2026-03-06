# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import numpy as np
import openvino as ov
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.multimodal import (BatchedTensorInputs,
                             MultiModalKwargs)
from vllm.sampling_params import SamplingType
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.attention.backends.abstract import AttentionMetadata

from vllm_openvino.attention.backends.openvino import OpenVINOAttentionMetadata
from vllm_openvino.model_executor.model_loader.openvino import get_model

from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class OpenVINOModelRunnerV1:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        ov_core: ov.Core = None,
        kv_cache_dtype: Optional[str] = "auto",
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.parallel_config = vllm_config.parallel_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.ov_core = ov_core or ov.Core()
        self.kv_cache_dtype = kv_cache_dtype
        self.model: nn.Module  # Set after load_model()

        # V1 state management
        self.requests: dict[str, CachedRequestState] = {}
        self.input_batch = InputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            device=self.device,
            pin_memory=False,  # OpenVINO/CPU — no pin memory
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
        )

        # KV cache — set by worker after initialize_cache
        self.kv_caches: list = []
        self.block_size: int = 0

    def load_model(self) -> None:
        self.model = get_model(vllm_config=self.vllm_config,
                               kv_cache_dtype=self.kv_cache_dtype,
                               ov_core=self.ov_core)

    def get_model(self) -> nn.Module:
        return self.model

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        """Update cached request states from scheduler output."""
        # Remove finished requests
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.input_batch.remove_request(req_id)

        # Remove unscheduled requests from batch (but keep cached state)
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = set(self.input_batch.req_id_to_index.keys())
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        # Add new requests
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_features=new_req_data.mm_features,
                sampling_params=new_req_data.sampling_params,
                pooling_params=new_req_data.pooling_params,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state
            self.input_batch.add_request(req_state)

        # Update cached (running) requests
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            if req_id not in self.requests:
                continue
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            if new_block_ids is not None:
                if req_id in req_data.resumed_req_ids:
                    req_state.block_ids = new_block_ids
                else:
                    req_state.block_ids = tuple(
                        existing + new
                        for existing, new in zip(req_state.block_ids, new_block_ids)
                    )
            req_state.num_computed_tokens = num_computed_tokens
            if req_id not in self.input_batch.req_id_to_index:
                self.input_batch.add_request(req_state)

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata,
               SamplingMetadata, BatchedTensorInputs]:
        """Prepare the model input based on scheduled requests.
        """
        input_tokens = []
        input_positions = []
        seq_lens = []
        past_lens = []
        query_lens = []

        subsequence_begins = []
        block_indices = []
        block_indices_begins = []

        subsequence_begins.append(0)
        block_indices_begins.append(0)

        if len(self.requests) == 0:
            return (
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
                None,
                SamplingMetadata.empty(),
                {},
            )

        for req_id in self.input_batch.req_ids:
            request = self.requests[req_id]
            block_table = request.block_ids[0]

            block_indices.extend(block_table)
            block_indices_begins.append(block_indices_begins[-1] +
                                        len(block_table))
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            last_token_position = num_scheduled_tokens + request.num_computed_tokens
            tokens = [] if request.num_computed_tokens >= len(request.prompt_token_ids) else request.prompt_token_ids[request.num_computed_tokens:last_token_position]
            tokens += request.output_token_ids[request.num_computed_tokens - len(request.prompt_token_ids): last_token_position - len(request.prompt_token_ids)]
            seq_len = len(tokens) + request.num_computed_tokens
            seq_lens.append(seq_len)
            query_len = len(tokens)
            query_lens.append(query_len)
            input_tokens.extend(tokens)
            positions_range = range(request.num_computed_tokens, seq_len)
            input_positions.extend(list(positions_range))

            past_lens.append(request.num_computed_tokens)
            subsequence_begins.append(subsequence_begins[-1] + query_len)

        sampled_token_indices = np.array(subsequence_begins[1:]) - 1

        max_query_len = max(query_lens)
        assert max_query_len > 0, "Invalid query_lens: {}".format(query_lens)

        input_tokens = ov.Tensor(np.array(input_tokens), ov.Shape([len(input_tokens)]), ov.Type.i64)

        input_positions = ov.Tensor(np.array(input_positions, dtype=np.int64))
        sampled_token_indices_tensor = ov.Tensor(np.array(sampled_token_indices, dtype=np.int64))

        past_lens_tensor = ov.Tensor(np.array(past_lens, dtype=np.int32))
        subsequence_begins_tensor = ov.Tensor(np.array(subsequence_begins, dtype=np.int32))
        block_indices_tensor = ov.Tensor(np.array(block_indices, dtype=np.int32))
        block_indices_begins_tensor = ov.Tensor(np.array(block_indices_begins, dtype=np.int32))
        max_context_len_tensor = ov.Tensor(np.array(max(seq_lens), dtype=np.int32))

        attn_metadata = OpenVINOAttentionMetadata(
            past_lens=past_lens_tensor,
            subsequence_begins=subsequence_begins_tensor,
            block_indices=block_indices_tensor,
            block_indices_begins=block_indices_begins_tensor,
            max_context_len=max_context_len_tensor,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            sampled_token_indices=sampled_token_indices_tensor
        )

        return (
            input_tokens,
            input_positions,
            attn_metadata,
            self.input_batch.sampling_metadata,
            {},
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)

        (
            input_tokens,
            input_positions,
            attn_metadata,
            sampling_metadata,
            multi_modal_kwargs,
        ) = self._prepare_inputs(scheduler_output)

        model_executable = self.model
        execute_model_kwargs = {
            "input_ids":
            input_tokens,
            "positions":
            input_positions,
            "kv_caches":
            self.kv_caches,
            **MultiModalKwargs.as_kwargs(multi_modal_kwargs or {},
                                         device=self.device),
        }

        with set_forward_context(attn_metadata, self.vllm_config, 0):
            hidden_states = model_executable(**execute_model_kwargs)

        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata

        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        sampled_tokens = sampler_output.sampled_token_ids.tolist()

        logprobs_lists = sampler_output.logprobs_tensors.tolist() \
            if sampler_output.logprobs_tensors is not None else None

        valid_sampled_tokens = sampled_tokens

        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            # Ignore the sampled token for partial prefills.
            if seq_len < req_state.num_tokens:
                valid_sampled_tokens[i] = []

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_tokens,
            spec_token_ids=None,
            logprobs=logprobs_lists,
            prompt_logprobs_dict={},
        )
