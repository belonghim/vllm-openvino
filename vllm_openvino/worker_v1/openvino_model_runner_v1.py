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
        if ov_core is not None:
            self.ov_core = ov_core
        else:
            self.ov_core = ov.Core()
            self.ov_core.set_property({ov_props.enable_mmap: True})
        self.model: nn.Module  # Set after load_model()

        # V1 state management
        self.requests: dict[str, CachedRequestState] = {}
        self.num_cache_groups = 1
        self.input_batch = self._create_input_batch(self.num_cache_groups)

        # KV cache — set by worker after initialize_cache
        self.kv_caches: list[tuple[ov.Tensor, ov.Tensor]] = []
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
        self._max_context_len_buf = np.zeros((), dtype=np.int32)

        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        self._past_lens_tensor_base = ov.Tensor(
            self._past_lens_buf, ov.Shape([max_seqs]), ov.Type.i32)
        self._subseq_begins_tensor_base = ov.Tensor(
            self._subseq_begins_buf, ov.Shape([max_seqs + 1]), ov.Type.i32)
        self._sampled_idx_tensor_base = ov.Tensor(
            self._sampled_idx_buf, ov.Shape([max_seqs]), ov.Type.i64)
        self._input_tokens_tensor_base = ov.Tensor(
            self._input_tokens_buf,
            ov.Shape([max_num_batched_tokens]),
            ov.Type.i64)
        self._input_positions_tensor_base = ov.Tensor(
            self._input_positions_buf,
            ov.Shape([max_num_batched_tokens]),
            ov.Type.i64)

        self._position_range_buf = np.arange(
            self.model_config.max_model_len, dtype=np.int64)

        # Track requests that have multimodal features.
        self._mm_req_ids: set[str] = set()
        self._new_req_ids: set[str] = set()

        # Pre-allocated block-index tensors.
        self._init_block_index_tensors()

        # Private conv-state slot pool for hybrid-PA models (conv-only PA
        # transform). Not vLLM-scheduler-managed: bypasses MambaSpec's
        # multi-group block striping, which is incompatible with these
        # models' single shared conv_state_table.* input set.
        num_conv_slots = max_seqs + 1
        self._conv_slot_by_req: dict[str, int] = {}
        self._conv_slot_free: list[int] = list(range(num_conv_slots))
        self._la_block_indices_buf = np.zeros(2 * max_seqs, dtype=np.int32)
        self._la_block_indices_begins_buf = np.zeros(max_seqs + 1, dtype=np.int32)
        self._la_past_lens_buf = np.zeros(max_seqs, dtype=np.int32)
        self._la_cache_interval_buf = np.zeros(max_seqs, dtype=np.int32)
        self._la_block_indices_tensor_base = ov.Tensor(
            self._la_block_indices_buf, ov.Shape([2 * max_seqs]), ov.Type.i32)
        self._la_block_indices_begins_tensor_base = ov.Tensor(
            self._la_block_indices_begins_buf, ov.Shape([max_seqs + 1]), ov.Type.i32)
        self._la_past_lens_tensor_base = ov.Tensor(
            self._la_past_lens_buf, ov.Shape([max_seqs]), ov.Type.i32)
        self._la_cache_interval_tensor_base = ov.Tensor(
            self._la_cache_interval_buf, ov.Shape([max_seqs]), ov.Type.i32)

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
        self._block_indices_group_tensors_out = [None] * self.num_cache_groups
        self._block_indices_begins_group_tensors_out = [None] * self.num_cache_groups

    @staticmethod
    def _slice_tensor(base_tensor: ov.Tensor, length: int) -> ov.Tensor:
        return ov.Tensor(base_tensor, ov.Coordinate([0]), ov.Coordinate([length]))

    def release_finished_conv_slots(self, scheduler_output: SchedulerOutput) -> None:
        """Release conv slots for finished requests.

        Called unconditionally (even on zero-scheduled-token steps) because
        async scheduling can report a request as finished in a step that
        schedules no new tokens for anyone, which would otherwise bypass
        _update_states() and starve the small (max_num_seqs+1) slot pool.
        """
        for req_id in scheduler_output.finished_req_ids:
            slot = self._conv_slot_by_req.pop(req_id, None)
            if slot is not None:
                self._conv_slot_free.append(slot)

    def _get_conv_slot(self, req_id: str, num_computed: int) -> int:
        slot = self._conv_slot_by_req.get(req_id)
        if slot is None:
            slot = self._conv_slot_free.pop()
            self._conv_slot_by_req[req_id] = slot
        if num_computed == 0:
            # Fresh prefill (first ever, or a preemption re-prefill reusing
            # the same slot): the conv history for this slot is invalid.
            for conv_cache in self.conv_caches:
                conv_cache.data[slot] = 0
        return slot

    def _create_input_batch(self, num_cache_groups: int) -> InputBatch:
        block_size = self.cache_config.block_size
        n_groups = max(1, num_cache_groups)
        block_sizes = [block_size] * n_groups
        max_blocks = max(1, (self.model_config.max_model_len + block_size - 1) // block_size)
        max_num_blocks_per_req = [max_blocks] * n_groups
        return InputBatch(
            max_num_reqs=self.scheduler_config.max_num_seqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            device=self.device,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=block_sizes,
            kernel_block_sizes=block_sizes,
            max_num_blocks_per_req=max_num_blocks_per_req,
            is_pooling_model=getattr(self.model_config, 'is_pooling_model', False),
        )

    def configure_cache_groups(self, num_cache_groups: int) -> None:
        self.num_cache_groups = max(1, num_cache_groups)
        self.input_batch = self._create_input_batch(self.num_cache_groups)
        self._init_block_index_tensors()

    def load_model(
        self,
        preloaded_model_type: str | None = None,
        preloaded_ssm_state_shapes: dict | None = None,
        preloaded_ov_model: "ov.Model | None" = None,
    ) -> None:
        self.model = get_model(
            vllm_config=self.vllm_config,
            ov_core=self.ov_core,
            preloaded_model_type=preloaded_model_type,
            preloaded_ssm_state_shapes=preloaded_ssm_state_shapes,
            preloaded_ov_model=preloaded_ov_model,
        )

    def get_model(self) -> nn.Module:
        return self.model

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        """Update cached request states from scheduler output."""
        # Remove finished requests
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.input_batch.remove_request(req_id)
            self._mm_req_ids.discard(req_id)
            slot = self._conv_slot_by_req.pop(req_id, None)
            if slot is not None:
                self._conv_slot_free.append(slot)

        # Remove unscheduled requests from batch (but keep cached state)
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        for req_id in list(self.input_batch.req_id_to_index):
            if req_id not in scheduled_req_ids or req_id in resumed_req_ids:
                self.input_batch.remove_request(req_id)

        # Add new requests
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=getattr(new_req_data, 'prompt_embeds', None),
                prompt_is_token_ids=getattr(new_req_data, 'prompt_is_token_ids', True),
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
        max_seq_len = 0
        max_query_len = 0

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
            num_tokens = num_tokens_total - num_computed

            seq_len = num_tokens_total
            if seq_len > max_seq_len:
                max_seq_len = seq_len
            query_len = num_tokens
            if query_len > max_query_len:
                max_query_len = query_len
            self._input_tokens_buf[token_idx:token_idx + num_tokens] = \
                self.input_batch.token_ids_cpu[req_index, num_computed:num_tokens_total]
            token_idx += num_tokens

            if num_tokens == 1:
                self._input_positions_buf[pos_idx] = num_computed
            else:
                self._input_positions_buf[pos_idx:pos_idx + num_tokens] = \
                    self._position_range_buf[num_computed:num_tokens_total]
            pos_idx += num_tokens

            self._past_lens_buf[n_reqs] = num_computed
            self._subseq_begins_buf[n_reqs + 1] = self._subseq_begins_buf[n_reqs] + query_len

            if self.conv_caches:
                conv_slot = self._get_conv_slot(req_id, num_computed)
                self._la_block_indices_buf[2 * n_reqs] = conv_slot
                self._la_block_indices_buf[2 * n_reqs + 1] = conv_slot
                self._la_block_indices_begins_buf[n_reqs + 1] = 2 * (n_reqs + 1)
                self._la_past_lens_buf[n_reqs] = num_computed
                self._la_cache_interval_buf[n_reqs] = 0

            n_reqs += 1

        self._sampled_idx_buf[:n_reqs] = self._subseq_begins_buf[1:n_reqs + 1] - 1

        multi_modal_kwargs = {}
        if self._mm_req_ids:
            all_pixel_values = []
            all_pixel_position_ids = []
            all_image_grid_thw = []
            all_image_position_ids = []

            mm_req_ids = [
                req_id for req_id in self._mm_req_ids
                if req_id in self.input_batch.req_id_to_index
            ]
            mm_req_ids.sort(key=self.input_batch.req_id_to_index.__getitem__)
            for req_id in mm_req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_computed = self.input_batch.num_computed_tokens_cpu[req_index]
                if num_computed > 0:
                    continue
                request = self.requests[req_id]
                for mm_feature in request.mm_features:
                    mm_item = mm_feature.data
                    if mm_item is not None:
                        if "pixel_values" in mm_item:
                            all_pixel_values.append(mm_item["pixel_values"].data)
                        else:
                            for _key, elem in mm_item.items():
                                if hasattr(elem.data, 'shape'):
                                    all_pixel_values.append(elem.data)
                                    break
                        if "pixel_position_ids" in mm_item:
                            all_pixel_position_ids.append(
                                mm_item["pixel_position_ids"].data)
                        if "image_grid_thw" in mm_item:
                            all_image_grid_thw.append(
                                mm_item["image_grid_thw"].data)
                    pos = mm_feature.mm_position
                    all_image_position_ids.append(
                        (pos.offset, pos.offset + pos.length))

            if all_pixel_values:
                pixel_values = torch.stack(all_pixel_values)
                if pixel_values.device != self.device:
                    pixel_values = pixel_values.to(self.device)
                multi_modal_kwargs["pixel_values"] = pixel_values

                if all_pixel_position_ids:
                    pixel_position_ids = torch.stack(all_pixel_position_ids)
                    if pixel_position_ids.device != self.device:
                        pixel_position_ids = pixel_position_ids.to(self.device)
                    multi_modal_kwargs["pixel_position_ids"] = pixel_position_ids

                if all_image_grid_thw:
                    image_grid_thw = torch.stack(all_image_grid_thw)
                    if image_grid_thw.device != self.device:
                        image_grid_thw = image_grid_thw.to(self.device)
                    multi_modal_kwargs["image_grid_thw"] = image_grid_thw

                image_position_ids = torch.tensor(
                    all_image_position_ids, dtype=torch.int64)
                if image_position_ids.device != self.device:
                    image_position_ids = image_position_ids.to(self.device)
                multi_modal_kwargs["image_position_ids"] = image_position_ids

        assert max_query_len > 0, "Invalid: all scheduled sequences have zero query length"

        input_tokens = self._slice_tensor(self._input_tokens_tensor_base, token_idx)
        input_positions = self._slice_tensor(self._input_positions_tensor_base, pos_idx)
        sampled_token_indices_tensor = self._slice_tensor(self._sampled_idx_tensor_base, n_reqs)
        past_lens_tensor = self._slice_tensor(self._past_lens_tensor_base, n_reqs)
        subsequence_begins_tensor = self._slice_tensor(self._subseq_begins_tensor_base, n_reqs + 1)
        for group_idx in range(self.num_cache_groups):
            num_blocks = int(self._block_idx_group_offsets[group_idx])
            if num_blocks == 0:
                self._block_indices_group_tensors_out[group_idx] = \
                    self._empty_block_indices_group_tensors[group_idx]
            else:
                self._block_indices_group_tensors_out[group_idx] = \
                    self._slice_tensor(
                        self._block_indices_group_tensors_base[group_idx],
                        num_blocks,
                    )
            self._block_indices_begins_group_tensors_out[group_idx] = \
                self._slice_tensor(
                    self._block_idx_begins_group_tensors_base[group_idx],
                    n_reqs + 1,
                )
        block_indices_group_tensors = self._block_indices_group_tensors_out
        block_indices_begins_group_tensors = self._block_indices_begins_group_tensors_out
        self._max_context_len_buf[()] = max_seq_len
        max_context_len_tensor = ov.Tensor(self._max_context_len_buf, ov.Shape([1]), ov.Type.i32)

        la_block_indices = la_block_indices_begins = None
        la_past_lens = la_cache_interval = None
        if self.conv_caches:
            la_block_indices = self._slice_tensor(self._la_block_indices_tensor_base, 2 * n_reqs)
            la_block_indices_begins = self._slice_tensor(self._la_block_indices_begins_tensor_base, n_reqs + 1)
            la_past_lens = self._slice_tensor(self._la_past_lens_tensor_base, n_reqs)
            la_cache_interval = self._slice_tensor(self._la_cache_interval_tensor_base, n_reqs)

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
            sampled_token_indices=sampled_token_indices_tensor,
            la_block_indices=la_block_indices,
            la_block_indices_begins=la_block_indices_begins,
            la_past_lens=la_past_lens,
            la_cache_interval=la_cache_interval,
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

        self.input_batch.condense()
        self.input_batch.refresh_metadata()
        new_req_ids = self.input_batch.req_ids

        if self.model is not None and not getattr(self.model, '_has_kv_cache_inputs', True):
            has_running = has_new = False
            for req_id in new_req_ids:
                if req_id is None:
                    continue
                if req_id in self._new_req_ids:
                    has_new = True
                else:
                    has_running = True
                if has_running and has_new:
                    break
            if has_new and not has_running and hasattr(self.model, 'recreate_infer_request'):
                logger.info("[OV-RUNNER] All slots are new requests, recreating infer request")
                self.model.recreate_infer_request()

        (
            input_tokens,
            input_positions,
            attn_metadata,
            sampling_metadata,
            multi_modal_kwargs,
        ) = self._prepare_inputs(scheduler_output)
        self._new_req_ids.clear()

        actual_num_requests = len(self.input_batch.req_id_to_index)
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
                sampled_tokens[i] = []

        # Snapshot (not a live reference): with async scheduling, a later
        # step can mutate self.input_batch before the scheduler consumes
        # this step's output, corrupting it if it aliased the live dicts.
        return ModelRunnerOutput(
            req_ids=list(self.input_batch.req_ids),
            req_id_to_index=dict(self.input_batch.req_id_to_index),
            sampled_token_ids=sampled_tokens,
            logprobs=logprobs_lists,
            prompt_logprobs_dict={},
            pooler_output=None,
        )
