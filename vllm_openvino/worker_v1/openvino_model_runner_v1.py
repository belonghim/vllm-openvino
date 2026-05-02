# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino as ov
import openvino.properties as ov_props
import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.multimodal import BatchedTensorInputs

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.attention.backend import AttentionMetadata

from vllm_openvino.attention.backends.openvino import OpenVINOAttentionMetadata
from vllm_openvino.model_executor.model_loader.openvino import get_model

from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class OpenVINOModelRunnerV1:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        ov_core: ov.Core | None = None,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.parallel_config = vllm_config.parallel_config
        self.compilation_config = vllm_config.compilation_config
        self.device = device
        self.ov_core = ov_core or ov.Core()
        self.ov_core.set_property({ov_props.enable_mmap: True})
        self.model: nn.Module  # Set after load_model()

        # V1 state management
        self.requests: dict[str, CachedRequestState] = {}
        self.num_cache_groups = 1
        self.input_batch = self._create_input_batch(self.num_cache_groups)

        # KV cache — set by worker after initialize_cache
        self.kv_caches: list = []
        self.ssm_caches: list[ov.Tensor] = []
        self.conv_caches: list[ov.Tensor] = []
        self.block_size: int = 0

        # Pre-allocated fixed-size buffers for _prepare_inputs (avoids per-call allocation)
        max_seqs = self.scheduler_config.max_num_seqs
        self._past_lens_buf = np.zeros(max_seqs, dtype=np.int32)
        self._subseq_begins_buf = np.zeros(max_seqs + 1, dtype=np.int32)
        self._block_idx_begins_buf = np.zeros(max_seqs + 1, dtype=np.int32)
        self._sampled_idx_buf = np.zeros(max_seqs, dtype=np.int64)
        self._input_tokens_buf = np.zeros(
            self.scheduler_config.max_num_batched_tokens,
            dtype=np.int64,
        )
        self._input_positions_buf = np.zeros(
            self.scheduler_config.max_num_batched_tokens,
            dtype=np.int64,
        )

        # Track requests that have multimodal features.
        self._mm_req_ids: set[str] = set()
        self._new_req_ids: set[str] = set()

        # Pre-allocated block-index tensors.
        self._init_block_index_tensors()

    def _init_block_index_tensors(self) -> None:
        max_seqs = self.scheduler_config.max_num_seqs
        block_size = self.cache_config.block_size
        max_blocks_per_seq = max(1, (self.model_config.max_model_len + block_size - 1) // block_size)
        self._max_block_indices = max_seqs * max_blocks_per_seq

        self._block_indices_group_tensors_base = [
            ov.Tensor(ov.Type.i32, ov.Shape([self._max_block_indices]))
            for _ in range(self.num_cache_groups)
        ]
        self._block_indices_group_data = [
            tensor.data for tensor in self._block_indices_group_tensors_base
        ]

        self._block_idx_begins_group_tensors_base = []
        self._block_idx_begins_group_data = []
        self._block_idx_begins_group_bufs = []
        for group_idx in range(self.num_cache_groups):
            if group_idx == 0:
                group_begins_buf = self._block_idx_begins_buf
            else:
                group_begins_buf = np.zeros(max_seqs + 1, dtype=np.int32)
            tensor = ov.Tensor(group_begins_buf, ov.Shape([max_seqs + 1]), ov.Type.i32)
            self._block_idx_begins_group_bufs.append(group_begins_buf)
            self._block_idx_begins_group_tensors_base.append(tensor)
            self._block_idx_begins_group_data.append(tensor.data)

        self._block_idx_group_offsets = np.zeros(self.num_cache_groups, dtype=np.int32)
        self._empty_block_indices_group_tensors = [
            ov.Tensor(np.empty(0, dtype=np.int32))
            for _ in range(self.num_cache_groups)
        ]

    @staticmethod
    def _slice_tensor(base_tensor: ov.Tensor, length: int) -> ov.Tensor:
        return ov.Tensor(base_tensor, ov.Coordinate([0]), ov.Coordinate([length]))

    def _create_input_batch(self, num_cache_groups: int) -> InputBatch:
        block_sizes = [self.cache_config.block_size] * max(1, num_cache_groups)
        return InputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            device=self.device,
            pin_memory=False,  # OpenVINO/CPU — no pin memory
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=block_sizes,
            kernel_block_sizes=block_sizes,
        )

    def configure_cache_groups(self, num_cache_groups: int) -> None:
        self.num_cache_groups = max(1, num_cache_groups)
        self.input_batch = self._create_input_batch(self.num_cache_groups)
        self._init_block_index_tensors()

    def load_model(self) -> None:
        self.model = get_model(vllm_config=self.vllm_config,
                               ov_core=self.ov_core)

    def get_model(self) -> nn.Module:
        return self.model

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        """Update cached request states from scheduler output."""
        # Remove finished requests
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.input_batch.remove_request(req_id)
            self._mm_req_ids.discard(req_id)

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
            self._new_req_ids.add(req_id)
            if req_state.mm_features:
                self._mm_req_ids.add(req_id)

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
    ) -> tuple[torch.Tensor, torch.Tensor, AttentionMetadata,
               SamplingMetadata, BatchedTensorInputs]:
        """Prepare the model input based on scheduled requests.
        """
        if len(self.requests) == 0:
            return (
                torch.empty(0, device=self.device),
                torch.empty(0, device=self.device),
                None,
                self.input_batch.sampling_metadata,
                {},
            )

        token_idx = 0
        pos_idx = 0
        seq_lens = []
        query_lens = []

        n_reqs = 0
        self._subseq_begins_buf[0] = 0
        self._block_idx_group_offsets.fill(0)
        for group_idx in range(self.num_cache_groups):
            self._block_idx_begins_group_data[group_idx][0] = 0

        for req_id in self.input_batch.req_ids:
            req_index = self.input_batch.req_id_to_index[req_id]
            request = self.requests[req_id]
            for group_idx in range(self.num_cache_groups):
                if group_idx < len(request.block_ids):
                    group_block_table = request.block_ids[group_idx]
                elif request.block_ids:
                    # Backward-compatible fallback for single-group block_ids.
                    group_block_table = request.block_ids[0]
                else:
                    group_block_table = []

                num_group_blocks = len(group_block_table)
                if num_group_blocks:
                    group_offset = int(self._block_idx_group_offsets[group_idx])
                    assert group_offset + num_group_blocks <= self._max_block_indices, \
                        f"block_indices overflow: {group_offset + num_group_blocks} > {self._max_block_indices}"
                    self._block_indices_group_data[group_idx][
                        group_offset:group_offset + num_group_blocks
                    ] = group_block_table
                    self._block_idx_group_offsets[group_idx] = (
                        group_offset + num_group_blocks
                    )

                self._block_idx_begins_group_data[group_idx][n_reqs + 1] = (
                    self._block_idx_group_offsets[group_idx]
                )

            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_computed = self.input_batch.num_computed_tokens_cpu[req_index]
            num_tokens_total = num_computed + num_scheduled_tokens

            tokens = self.input_batch.token_ids_cpu[
                req_index, num_computed:num_tokens_total
            ].tolist()

            seq_len = num_tokens_total
            seq_lens.append(seq_len)
            query_len = len(tokens)
            query_lens.append(query_len)
            num_tokens = len(tokens)
            self._input_tokens_buf[token_idx:token_idx + num_tokens] = tokens
            token_idx += num_tokens

            positions_range = range(num_computed, num_tokens_total)
            num_positions = len(positions_range)
            self._input_positions_buf[pos_idx:pos_idx + num_positions] = list(positions_range)
            pos_idx += num_positions

            self._past_lens_buf[n_reqs] = num_computed
            self._subseq_begins_buf[n_reqs + 1] = self._subseq_begins_buf[n_reqs] + query_len
            n_reqs += 1

        self._sampled_idx_buf[:n_reqs] = self._subseq_begins_buf[1:n_reqs + 1] - 1

        multi_modal_kwargs = {}
        all_pixel_values = []
        all_image_position_ids = []

        mm_req_ids = [
            req_id for req_id in self._mm_req_ids
            if req_id in self.input_batch.req_id_to_index
        ]
        mm_req_ids.sort(key=self.input_batch.req_id_to_index.__getitem__)
        for req_id in mm_req_ids:
            request = self.requests[req_id]
            for mm_feature in request.mm_features:
                # mm_feature.data is MultiModalKwargsItem (dict-like) in vLLM 0.19.1
                mm_item = mm_feature.data
                if mm_item is not None:
                    # Extract pixel_values tensor from MultiModalKwargsItem
                    if "pixel_values" in mm_item:
                        all_pixel_values.append(mm_item["pixel_values"].data)
                    else:
                        # Fallback: use first available tensor key
                        for _key, elem in mm_item.items():
                            if hasattr(elem.data, 'shape'):
                                all_pixel_values.append(elem.data)
                                break
                # Convert PlaceholderRange to (start, end) tuple
                pos = mm_feature.mm_position
                all_image_position_ids.append(
                    (pos.offset, pos.offset + pos.length))

        if all_pixel_values:
            pixel_values = torch.stack(all_pixel_values)
            if pixel_values.device != self.device:
                pixel_values = pixel_values.to(self.device)
            multi_modal_kwargs["pixel_values"] = pixel_values

            image_position_ids = torch.tensor(
                all_image_position_ids, dtype=torch.int64)
            if image_position_ids.device != self.device:
                image_position_ids = image_position_ids.to(self.device)
            multi_modal_kwargs["image_position_ids"] = image_position_ids

        max_query_len = max(query_lens)
        assert max_query_len > 0, "Invalid query_lens: {}".format(query_lens)

        input_tokens = ov.Tensor(
            self._input_tokens_buf[:token_idx],
            ov.Shape([token_idx]),
            ov.Type.i64,
        )

        input_positions = ov.Tensor(
            self._input_positions_buf[:pos_idx],
            ov.Shape([pos_idx]),
            ov.Type.i64,
        )
        sampled_token_indices_tensor = ov.Tensor(self._sampled_idx_buf[:n_reqs], ov.Shape([n_reqs]), ov.Type.i64)

        past_lens_tensor = ov.Tensor(self._past_lens_buf[:n_reqs], ov.Shape([n_reqs]), ov.Type.i32)
        subsequence_begins_tensor = ov.Tensor(self._subseq_begins_buf[:n_reqs + 1], ov.Shape([n_reqs + 1]), ov.Type.i32)
        block_indices_group_tensors = []
        block_indices_begins_group_tensors = []
        for group_idx in range(self.num_cache_groups):
            num_blocks = int(self._block_idx_group_offsets[group_idx])
            if num_blocks == 0:
                block_indices_group_tensors.append(
                    self._empty_block_indices_group_tensors[group_idx]
                )
            else:
                block_indices_group_tensors.append(
                    self._slice_tensor(
                        self._block_indices_group_tensors_base[group_idx],
                        num_blocks,
                    )
                )
            block_indices_begins_group_tensors.append(
                self._slice_tensor(
                    self._block_idx_begins_group_tensors_base[group_idx],
                    n_reqs + 1,
                )
            )
        max_context_len_tensor = ov.Tensor(np.array(max(seq_lens), dtype=np.int32))

        attn_metadata = OpenVINOAttentionMetadata(
            past_lens=past_lens_tensor,
            subsequence_begins=subsequence_begins_tensor,
            block_indices=block_indices_group_tensors[0],
            block_indices_groups=block_indices_group_tensors,
            block_indices_begins=block_indices_begins_group_tensors[0],
            block_indices_begins_groups=block_indices_begins_group_tensors,
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
            multi_modal_kwargs,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)

        old_req_ids = list(self.input_batch.req_ids)
        self.input_batch.condense()
        self.input_batch.refresh_metadata()
        new_req_ids = list(self.input_batch.req_ids)

        if not self.model._has_kv_cache_inputs:
            has_running = any(req_id not in self._new_req_ids for req_id in new_req_ids if req_id is not None)
            has_new = any(req_id in self._new_req_ids for req_id in new_req_ids if req_id is not None)
            if has_new and not has_running:
                logger.info("[OV-RUNNER] All slots are new requests, recreating infer request")
                self.model.recreate_infer_request()
            self._new_req_ids.clear()

        (
            input_tokens,
            input_positions,
            attn_metadata,
            sampling_metadata,
            multi_modal_kwargs,
        ) = self._prepare_inputs(scheduler_output)

        actual_num_requests = sum(
            1 for req_id in self.input_batch.req_ids if req_id is not None)
        model_executable = self.model
        execute_model_kwargs = {
            "input_ids": input_tokens,
            "positions": input_positions,
            "kv_caches": self.kv_caches,
            "ssm_caches": self.ssm_caches,
            "conv_caches": self.conv_caches,
            "num_requests": actual_num_requests,
            **multi_modal_kwargs,
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

        logprobs_lists = sampler_output.logprobs_tensors.tolists() \
            if sampler_output.logprobs_tensors is not None else None

        valid_sampled_tokens = sampled_tokens

        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            req_index = self.input_batch.req_id_to_index[req_id]
            num_computed = self.input_batch.num_computed_tokens_cpu[req_index]
            num_scheduled = scheduler_output.num_scheduled_tokens[req_id]
            seq_len = num_computed + num_scheduled

            sampled_entry = sampled_tokens[i]
            sampled_ids = (sampled_entry if isinstance(sampled_entry, list)
                           else [sampled_entry])
            if sampled_ids:
                start_idx = seq_len
                end_idx = start_idx + len(sampled_ids)
                self.input_batch.token_ids_cpu[req_index, start_idx:end_idx] = sampled_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_idx
                req_state.output_token_ids.extend(sampled_ids)

            self.input_batch.num_computed_tokens_cpu[req_index] = seq_len

            # Ignore the sampled token for partial prefills (chunked prefill).
            # seq_len < num_prompt_tokens means we haven't finished the prompt.
            if seq_len < req_state.num_prompt_tokens:
                valid_sampled_tokens[i] = []

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_tokens,
            logprobs=logprobs_lists,
            prompt_logprobs_dict={},
            pooler_output=None,
        )
