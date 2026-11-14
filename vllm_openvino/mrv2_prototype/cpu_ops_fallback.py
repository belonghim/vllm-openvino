# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch versions of the `vllm._cpu_ops` the standalone CPU MRV2 runner
calls on the text path, for installs without the CPU `_C` extension
(`VLLM_TARGET_DEVICE=empty` or a CUDA wheel). They follow csrc/cpu/
block_table.cpp, input_batch.cpp and sampling_kernels.cpp; the Gumbel noise
is drawn from torch generators, so seeded samples differ from the C++ op."""
import torch

from vllm import _cpu_ops


def gather_block_table_cpu(block_table, output, idx_mapping, num_blocks,
                           num_reqs_padded):
    for b, req in enumerate(idx_mapping.tolist()):
        n = int(num_blocks[req])
        output[b, :n] = block_table[req, :n]
    output[idx_mapping.shape[0]:num_reqs_padded] = 0


def compute_slot_mapping_cpu(block_table, slot_mapping, idx_mapping,
                             query_start_loc, positions, block_size,
                             mapping_enabled, pad_slot_id):
    num_reqs = idx_mapping.shape[0]
    num_tokens = int(query_start_loc[num_reqs])
    slot_mapping[num_tokens:] = pad_slot_id
    for b, req in enumerate(idx_mapping.tolist()):
        start, end = int(query_start_loc[b]), int(query_start_loc[b + 1])
        if not mapping_enabled or req < 0:
            slot_mapping[start:end] = pad_slot_id
            continue
        pos = positions[start:end]
        blocks = block_table[req, pos // block_size].to(torch.int64)
        slot_mapping[start:end] = blocks * block_size + pos % block_size


def prepare_prefill_inputs_cpu(input_ids, next_prefill_tokens, idx_mapping,
                               query_start_loc, all_token_ids, prefill_len,
                               num_computed_tokens):
    num_lookahead = next_prefill_tokens.shape[0]
    for b, req in enumerate(idx_mapping.tolist()):
        plen, done = int(prefill_len[req]), int(num_computed_tokens[req])
        if done >= plen:
            continue
        start, end = int(query_start_loc[b]), int(query_start_loc[b + 1])
        n = end - start
        input_ids[start:end] = all_token_ids[req, done:done + n]
        for k in range(num_lookahead):
            pos = done + n + k
            next_prefill_tokens[k, req] = (
                all_token_ids[req, pos] if pos < plen else 0)


def prepare_pos_seq_lens_cpu(idx_mapping, query_start_loc, num_computed_tokens,
                             pos, seq_lens):
    num_reqs = idx_mapping.shape[0]
    seq_lens[num_reqs:] = 0
    for b, req in enumerate(idx_mapping.tolist()):
        done = int(num_computed_tokens[req])
        start, end = int(query_start_loc[b]), int(query_start_loc[b + 1])
        seq_lens[b] = done + end - start
        pos[start:end] = torch.arange(done, done + end - start)


def combine_sampled_and_draft_tokens_cpu(input_ids, idx_mapping,
                                         last_sampled_tokens, query_start_loc,
                                         seq_lens, prefill_len, draft_tokens,
                                         cu_num_logits, num_logits,
                                         num_new_sampled_tokens):
    logits_indices = torch.empty(num_logits, dtype=torch.int64)
    for b, req in enumerate(idx_mapping.tolist()):
        lo, hi = int(cu_num_logits[b]), int(cu_num_logits[b + 1])
        n_logits = hi - lo
        n_draft = n_logits - num_new_sampled_tokens
        query_end = int(query_start_loc[b + 1])
        logits_start = query_end - n_logits
        logits_indices[lo:hi] = torch.arange(logits_start, query_end)
        seq_len, plen = int(seq_lens[b]), int(prefill_len[req])
        if seq_len <= plen:
            continue
        if num_new_sampled_tokens > 0 and seq_len - n_logits >= plen:
            input_ids[logits_start] = last_sampled_tokens[req]
        if n_draft:
            input_ids[query_end - n_draft:query_end] = draft_tokens[req, :n_draft]
    return logits_indices


def post_update_cpu(idx_mapping, num_computed_tokens, last_sampled_tokens,
                    output_bin_counts, sampled_tokens, num_sampled,
                    num_rejected, query_start_loc, all_token_ids, total_len):
    for b, req in enumerate(idx_mapping.tolist()):
        if req < 0:
            continue
        total, n = int(total_len[req]), int(num_sampled[b])
        tokens = sampled_tokens[b, :n]
        if n:
            last_sampled_tokens[req] = tokens[-1]
            total_len[req] = total + n
            all_token_ids[req, total:total + n] = tokens.to(torch.int32)
            if output_bin_counts is not None:
                output_bin_counts[req].index_add_(
                    0, tokens, torch.ones(n, dtype=output_bin_counts.dtype))
        query_len = (0 if query_start_loc is None
                     else int(query_start_loc[b + 1] - query_start_loc[b]))
        num_computed_tokens[req] += query_len - int(num_rejected[b])


def gumbel_sample_cpu(logits, expanded_idx_mapping, temperature, seeds,
                      positions, apply_temperature, is_drafting, use_fp64):
    out = torch.empty(logits.shape[0], dtype=torch.int64)
    for row, req in enumerate(expanded_idx_mapping.tolist()):
        temp = float(temperature[req])
        x = logits[row].to(torch.float64 if use_fp64 else torch.float32)
        if temp == 0.0:
            out[row] = int(x.argmax())
            continue
        if apply_temperature:
            x = x / temp
        gen = torch.Generator().manual_seed(
            (int(seeds[req]) * 1000003 + int(positions[row])) & (2**63 - 1))
        u = torch.rand(x.shape, generator=gen, dtype=x.dtype).clamp_(1e-20, 1.0)
        out[row] = int((x - torch.log(-torch.log(u))).argmax())
    return out


def apply_logit_bias_cpu(logits, expanded_idx_mapping, positions,
                         num_allowed_token_ids, allowed_token_ids,
                         num_logit_bias, logit_bias_token_ids, logit_bias,
                         min_lens, num_stop_token_ids, restore_when_all_masked,
                         stop_token_ids, check_all_masked_rows):
    for row, req in enumerate(expanded_idx_mapping.tolist()):
        n_allowed = int(num_allowed_token_ids[req])
        if n_allowed:
            ids = allowed_token_ids[req, :n_allowed].to(torch.int64)
            kept = logits[row, ids].clone()
            logits[row] = float("-inf")
            logits[row, ids] = kept
        n_bias = int(num_logit_bias[req])
        if n_bias:
            ids = logit_bias_token_ids[req, :n_bias].to(torch.int64)
            logits[row, ids] += logit_bias[req, :n_bias]
        n_stop = int(num_stop_token_ids[req])
        if n_stop and int(positions[row]) + 1 < int(min_lens[req]):
            ids = stop_token_ids[req, :n_stop].to(torch.int64)
            saved = logits[row, ids].clone()
            logits[row, ids] = float("-inf")
            if (check_all_masked_rows and int(restore_when_all_masked[req])
                    and torch.isneginf(logits[row]).all()):
                logits[row, ids] = torch.where(
                    torch.isfinite(saved), saved, logits[row, ids])


def install() -> list[str]:
    """Replace the `_cpu_ops` wrappers whose `_C` op is missing, and unwrap
    the runner's `torch.compile(fullgraph=True)` helpers, which raise under
    TORCH_COMPILE_DISABLE=1."""
    from vllm.v1.worker.cpu import logprob, sampler  # noqa: PLC0415
    installed = []
    for name, fn in list(globals().items()):
        if (name.endswith("_cpu") and callable(fn)
                and not hasattr(torch.ops._C, name)):
            setattr(_cpu_ops, name, fn)
            installed.append(name)
    for module, name in ((logprob, "compute_token_logprobs"),
                         (logprob, "_compute_token_ranks"),
                         (sampler, "_apply_min_p")):
        orig = getattr(getattr(module, name), "_torchdynamo_orig_callable", None)
        if orig is not None:
            setattr(module, name, orig)
            installed.append(f"{module.__name__}.{name} (uncompiled)")
    return installed
