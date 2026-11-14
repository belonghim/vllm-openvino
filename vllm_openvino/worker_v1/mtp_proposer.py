# SPDX-License-Identifier: Apache-2.0
"""Multi-token prediction drafter for IRs that ship `openvino_mtp_model.xml`
(Qwen3.5/3.6 exports), following OpenVINO GenAI's MTP strategy.

The MTP layer at position i reads the main model's lm_head input h_i and the
embedding of token i+1 (the main step's own input embedding when token i+1
was in it, so image positions keep their image embedding), and its logits
predict token i+2. Its KV cache is one
more layer on the request's block table. Each step it first catches up on the
positions the main model just committed (prefill chunks included), which
yields the first draft from the last position; every further draft feeds the
MTP layer's own hidden state and the previous draft one position later.
Those chained positions hold draft-only KV that the next catch-up overwrites.
"""
import numpy as np
import openvino as ov
import torch
from vllm.config import VllmConfig


class MtpProposer:

    def __init__(self, vllm_config: VllmConfig, model) -> None:
        self.k = vllm_config.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self.model = model
        self._max_ctx = np.zeros((), dtype=np.int32)
        self._max_ctx_ov = ov.Tensor(self._max_ctx, shared_memory=True)

    def propose(
        self,
        spans: list[tuple[int, int, int, int, bool, list[int]]],
        token_ids_cpu: np.ndarray,
        kv_cache: tuple[ov.Tensor, ov.Tensor],
        block_size: int,
    ) -> list[list[int]]:
        """spans: per drafting request (InputBatch row, positions committed
        before this step, positions committed in this step, row of its first
        token in the main model's output, whether it emitted a token, block
        ids). Returns the drafts of each span."""
        drafts: list[list[int]] = [[] for _ in spans]
        catch_up = [i for i, span in enumerate(spans) if span[2]]
        if not catch_up:
            return drafts
        hidden = self.model.last_hidden_states
        # Copied first: the main embeddings may alias the embedding model's
        # output buffer, which the next embedding infer overwrites.
        main_embeds = [
            self.model.last_inputs_embeds[start + 1:start + num_new].copy()
            for _, _, num_new, start, *_ in (spans[i] for i in catch_up)]
        last_embeds = self._embed([
            token_ids_cpu[spans[i][0], spans[i][1] + spans[i][2]]
            for i in catch_up])
        embeds = []
        for n, part in enumerate(main_embeds):
            embeds += [part, last_embeds[n:n + 1]]
        logits, hidden_rows = self._infer(
            hidden=np.concatenate([
                hidden[spans[i][3]:spans[i][3] + spans[i][2]]
                for i in catch_up]),
            embeds=np.concatenate(embeds),
            past_lens=[spans[i][1] for i in catch_up],
            num_new=[spans[i][2] for i in catch_up],
            block_tables=[spans[i][5] for i in catch_up],
            kv_cache=kv_cache,
        )

        # (span, next position, row in the last MTP output) of requests that
        # emitted a token; partial prefill chunks only catch up.
        chain = []
        for out_row, (i, token) in enumerate(
                zip(catch_up, logits.argmax(-1).tolist())):
            if spans[i][4]:
                drafts[i] = [token]
                chain.append((i, spans[i][1] + spans[i][2], out_row))
        for _ in range(self.k - 1):
            # The scheduler reserves blocks for k lookahead tokens.
            chain = [(i, pos, row) for i, pos, row in chain
                     if pos < min(len(spans[i][5]) * block_size,
                                  self.max_model_len - 1)]
            if not chain:
                break
            logits, hidden_rows = self._infer(
                hidden=hidden_rows[[row for *_, row in chain]],
                embeds=self._embed([drafts[i][-1] for i, *_ in chain]),
                past_lens=[pos for _, pos, _ in chain],
                num_new=[1] * len(chain),
                block_tables=[spans[i][5] for i, *_ in chain],
                kv_cache=kv_cache,
            )
            for out_row, token in enumerate(logits.argmax(-1).tolist()):
                i, pos, _ = chain[out_row]
                drafts[i].append(token)
                chain[out_row] = (i, pos + 1, out_row)
        return drafts

    def _embed(self, token_ids: list[int]) -> np.ndarray:
        return self.model._prepare_embeddings(
            torch.tensor(token_ids, dtype=torch.int64))

    def _infer(self, hidden: np.ndarray, embeds: np.ndarray,
               past_lens: list[int], num_new: list[int],
               block_tables: list[list[int]],
               kv_cache: tuple[ov.Tensor, ov.Tensor],
               ) -> tuple[np.ndarray, np.ndarray]:
        """One MTP step over several requests; returns the logits of each
        request's last position and the MTP hidden states of those rows."""
        ends = np.cumsum(num_new)
        subsequence_begins = np.concatenate(([0], ends)).astype(np.int32)
        positions = np.concatenate([
            np.arange(past, past + n) for past, n in zip(past_lens, num_new)])
        block_counts = [len(blocks) for blocks in block_tables]
        self._max_ctx[()] = max(p + n for p, n in zip(past_lens, num_new))
        request = self.model.mtp_request
        request.infer({
            "hidden_states": np.ascontiguousarray(hidden, dtype=np.float32),
            "inputs_embeds": embeds,
            "position_ids": positions.astype(np.int64),
            "max_context_len": self._max_ctx_ov,
            "past_lens": np.asarray(past_lens, dtype=np.int32),
            "subsequence_begins": subsequence_begins,
            "block_indices_begins": np.concatenate(
                ([0], np.cumsum(block_counts))).astype(np.int32),
            "block_indices": np.concatenate(block_tables).astype(np.int32),
            "sampled_tokens_indices": (ends - 1).astype(np.int64),
            "key_cache.0": kv_cache[0],
            "value_cache.0": kv_cache[1],
        })
        logits = request.get_tensor("logits").data
        mtp_hidden = request.get_tensor("last_hidden_state").data
        return (logits.reshape(len(num_new), -1),
                mtp_hidden[ends - 1])
