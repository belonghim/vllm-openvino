# SPDX-License-Identifier: Apache-2.0
"""Prototype: drive the OpenVINO model with vLLM's standalone CPU MRV2 runner.

Request bookkeeping, the persistent batch, input preparation, sampling and
grammar bitmasks come from `StandaloneCPUModelRunner`; only the model, the KV
cache and the forward call are the plugin's. PagedAttention models only (no
Hybrid-PA, multimodal or drafts). Enabled by VLLM_OPENVINO_MRV2_PROTOTYPE=1.
"""
from typing import Any

import numpy as np
import openvino as ov
import torch
from vllm.forward_context import set_forward_context
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.cpu.block_table import CPUBlockTables
from vllm.v1.worker.cpu.logprob import CPUPromptLogprobsWorker
from vllm.v1.worker.cpu.model_states import init_model_state
from vllm.v1.worker.cpu.sampler import CPUSampler
from vllm.v1.worker.cpu.standalone_model_runner import (
    ExecuteModelState, StandaloneCPUModelRunner)
from vllm.v1.worker.cpu.structured_outputs import CPUStructuredOutputsWorker

from vllm_openvino.attention.backends.openvino import OpenVINOAttentionMetadata
from vllm_openvino.mrv2_prototype import cpu_ops_fallback


class _OVBlockZeroer:
    """Stands in for CPUKVBlockZeroer: zeroes freshly allocated OV blocks."""

    def __init__(self, kv_caches):
        self.kv_caches = kv_caches

    def zero_block_ids(self, block_ids) -> None:
        ids = np.asarray(list(block_ids), dtype=np.int64)
        for key_cache, value_cache in self.kv_caches:
            key_cache.data[ids] = 0
            value_cache.data[ids] = 0


class OpenVINOMRV2Runner(StandaloneCPUModelRunner):

    def __init__(self, vllm_config, ov_runner) -> None:
        cpu_ops_fallback.install()
        super().__init__(vllm_config, torch.device("cpu"))
        self.ov = ov_runner
        self._load_ov_model()

    def _load_ov_model(self) -> None:
        # StandaloneCPUModelRunner.load_model minus the vLLM model loader and
        # its device memory profiler.
        self.model = self.ov.model
        self.model_state = init_model_state(
            self.vllm_config, self.model, self.encoder_cache, self.device)
        self.decode_query_len = self.model_state.num_new_sampled_tokens_per_step
        self.sampler = CPUSampler(
            vllm_config=self.vllm_config, max_num_reqs=self.max_num_reqs,
            vocab_size=self.vocab_size, device=self.device,
            req_states=self.req_states,
            logprobs_mode=self.model_config.logprobs_mode,
            use_fp64_gumbel=self.model_config.use_fp64_gumbel,
            enable_trace_replay=self.model_config.enable_trace_replay,
            return_sampling_mask=self.model_config.return_sampling_mask)
        self.prompt_logprobs_worker = CPUPromptLogprobsWorker(
            self.max_num_reqs, self.device,
            logprobs_mode=self.model_config.logprobs_mode)
        self.structured_outputs_worker = CPUStructuredOutputsWorker(
            max_num_logits=self.max_num_reqs * self.decode_query_len,
            vocab_size=self.vocab_size, device=self.device,
            mask_stride=self.decode_query_len)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig,
                            **kwargs: Any) -> None:
        # The OV caches are allocated by the worker; only the block tables
        # (and the zeroer the base class expects) are set up here.
        self.kv_cache_config = kv_cache_config
        # The group metadata API differs before/after vllm#61452; one
        # full-attention group needs only the block size and table width.
        block_size = self.cache_config.block_size
        self.block_tables = CPUBlockTables(
            block_sizes=[block_size],
            max_num_reqs=self.max_num_reqs,
            max_num_batched_tokens=self.max_num_tokens,
            max_num_blocks_per_group=[-(-self.max_model_len // block_size)],
            device=self.device,
            slot_mapping_enabled=[False],
            sliding_window_reach=[0])
        self.model_state.initialize_kv_cache(kv_cache_config, self.block_tables)
        self.kv_block_zeroer = _OVBlockZeroer(self.ov.kv_caches)
        self.kv_caches = []

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput, *args, **kwargs):
        self.finish_requests(scheduler_output)
        self.free_states(scheduler_output)
        self.add_requests(scheduler_output)
        self.update_requests(scheduler_output)
        if scheduler_output.total_num_scheduled_tokens == 0:
            return self.kv_connector.no_forward(scheduler_output)

        batch_req_state, _ = self.gather_batch_req_state(
            scheduler_output, dummy_run=False)
        input_batch = self.prepare_inputs(scheduler_output, batch_req_state)
        block_table = self.block_tables.gather_block_tables(
            input_batch.idx_mapping,
            num_reqs_padded=input_batch.num_reqs_after_padding,
            num_computed_tokens=self.req_states.num_computed_tokens)[0]
        logits = self._ov_forward(input_batch, block_table)
        self.execute_model_state = ExecuteModelState(
            input_batch=input_batch, hidden_states=logits,
            finished_req_ids=scheduler_output.finished_req_ids,
            ec_connector_output=None)
        return None

    def _ov_forward(self, input_batch, block_table: torch.Tensor) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        seq_lens = input_batch.seq_lens[:num_reqs].numpy()
        block_size = self.cache_config.block_size
        num_blocks = (seq_lens + block_size - 1) // block_size
        table = block_table.numpy()
        block_indices = np.concatenate(
            [table[b, :num_blocks[b]] for b in range(num_reqs)]).astype(np.int32)
        begins = np.zeros(num_reqs + 1, dtype=np.int32)
        np.cumsum(num_blocks, out=begins[1:])
        attn_metadata = OpenVINOAttentionMetadata(
            past_lens=ov.Tensor(input_batch.num_computed_tokens_np.astype(np.int32)),
            subsequence_begins=ov.Tensor(
                input_batch.query_start_loc_np[:num_reqs + 1].astype(np.int32)),
            block_indices=ov.Tensor(block_indices),
            block_indices_groups=None,
            block_indices_begins=ov.Tensor(begins),
            block_indices_begins_groups=None,
            max_context_len=ov.Tensor(np.array([seq_lens.max()], dtype=np.int32)),
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            sampled_token_indices=ov.Tensor(
                input_batch.logits_indices.numpy().astype(np.int64)),
            la_block_indices=None, la_block_indices_begins=None,
            la_past_lens=None, la_cache_interval=None)
        num_tokens = input_batch.num_tokens
        with set_forward_context(attn_metadata, self.vllm_config, 0):
            # ov.Tensor inputs replace the request's bound tensors; numpy or
            # torch inputs are copied into them, which fails when the runner
            # bound smaller ROI tensors before (warmup).
            logits = self.model(
                input_ids=ov.Tensor(
                    input_batch.input_ids[:num_tokens].numpy().astype(np.int64)),
                positions=ov.Tensor(np.ascontiguousarray(
                    input_batch.positions[:num_tokens].numpy())),
                kv_caches=self.ov.kv_caches[:self.ov.num_main_kv_layers],
                ssm_caches=[], conv_caches=[], num_requests=num_reqs)
        # The logits alias the infer request's output; sampling runs after.
        return logits.clone()

    def sample(self, hidden_states, input_batch, grammar_output):
        # The graph returns logits for the sampled rows only (gather before
        # matmul), so there is no compute_logits step.
        logits = hidden_states
        if grammar_output is not None and logits.shape[0] > 0:
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits, input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask)
        sampler_output = self.sampler(logits, input_batch)
        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected
