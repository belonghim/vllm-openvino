# Checks the runner's model-free logic against brute-force references: the
# grammar bitmask row mapping, embedding runs, Hybrid-PA dependency waves and
# draft verification (greedy, and speculative sampling reproducing the target
# distribution). Run inside the image with the source mounted; smoke-test.sh
# does so.
import sys
from types import SimpleNamespace as NS

import numpy as np
import torch

import vllm_openvino.worker_v1.openvino_worker_v1  # noqa: F401 (must import)
from vllm_openvino.worker_v1.openvino_model_runner_v1 import (
    OpenVINOModelRunnerV1 as Runner, apply_grammar_bitmask)

rng = np.random.default_rng(0)
failures = []


def check(name: str, ok: bool) -> None:
    print(f"  {name}: {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append(name)


def bitmask_case(vocab, words, req_ids, spec, masked_ids, stride):
    logits = torch.zeros(sum(1 + len(spec.get(r, ())) for r in req_ids), vocab)
    bitmask = rng.integers(-2**31, 2**31, (len(masked_ids) * stride, words),
                           dtype=np.int64).astype(np.int32)
    expected = torch.zeros_like(logits)
    row = 0
    for req_id in req_ids:
        rows = 1 + len(spec.get(req_id, ()))
        if req_id in masked_ids:
            base = masked_ids.index(req_id) * stride
            for r in range(rows):
                for v in range(min(vocab, words * 32)):
                    if not (int(bitmask[base + r, v // 32]) >> (v % 32)) & 1:
                        expected[row + r, v] = -float("inf")
        row += rows
    apply_grammar_bitmask(logits, req_ids, spec, NS(
        structured_output_request_ids=masked_ids, grammar_bitmask=bitmask))
    return torch.equal(logits, expected)


check("grammar bitmask, drafts and unmasked requests", bitmask_case(
    70, 3, ["a", "b", "c"], {"b": [1, 2], "c": [3]}, ["c", "a", "gone"], 3))
check("grammar bitmask, vocabulary wider than the bitmask", bitmask_case(
    100, 3, ["a", "b"], {"a": [5]}, ["b", "a"], 2))
check("grammar bitmask, every row masked", bitmask_case(
    64, 2, ["a", "b"], {}, ["a", "b"], 1))


def embed_runs_case(offset, is_embed, lo, hi):
    pos = NS(offset=offset, is_embed=None, embeds_cumsum=None)
    if is_embed is not None:
        pos.is_embed = torch.tensor(is_embed)
        # A list, as vLLM's PlaceholderRange.embeds_cumsum.
        pos.embeds_cumsum = pos.is_embed.cumsum(0).tolist()
        expected = [(offset + i, pos.embeds_cumsum[i] - 1)
                    for i, flag in enumerate(is_embed)
                    if flag and lo <= offset + i < hi]
    else:
        expected = [(t, t - offset) for t in range(lo, hi)]
    got = [(t, embed + t - start)
           for start, end, embed in Runner._embed_runs(pos, lo, hi)
           for t in range(start, end)]
    return got == expected


mask = [False, True, True, False, True, False, False, True, True, True]
check("embed runs", all(
    embed_runs_case(7, mask, lo, hi)
    for lo in range(7, 17) for hi in range(lo + 1, 18))
    and embed_runs_case(7, None, 9, 14))


def waves_case(interval, reqs):
    """reqs: {req_id: (num_computed, num_scheduled, blocks)}"""
    ids = list(reqs)
    stub = NS(
        _la_interval=interval,
        input_batch=NS(
            req_ids=ids, req_id_to_index={r: i for i, r in enumerate(ids)},
            num_computed_tokens_cpu=np.array([reqs[r][0] for r in ids])),
        requests={r: NS(block_ids=(reqs[r][2], )) for r in ids})
    return Runner._la_dependency_waves(stub, NS(
        num_scheduled_tokens={r: reqs[r][1] for r in ids}))


check("waves, independent requests", waves_case(4, {
    "a": (0, 8, [10, 11]), "c": (4, 4, [20, 21])}) is None)
# b hits the blocks a fills in this step; d hits the block b fills.
check("waves, chained same-step prefix hits", waves_case(4, {
    "a": (0, 8, [10, 11]), "b": (8, 2, [10, 11, 12]),
    "c": (4, 4, [20, 21]), "d": (12, 1, [10, 11, 12, 13])})
    == [["a", "c"], ["b"], ["d"]])


def verify_stub(greedy=()):
    return NS(
        _draft_probs={},
        input_batch=NS(
            req_ids=["a", "b", "c"], greedy_reqs=set(greedy),
            sampling_metadata=NS(
                generators={1: torch.Generator().manual_seed(0)})),
        model=NS(
            sampler=NS(random_probs=lambda logits, md, rows:
                       logits.softmax(-1)),
            sample=lambda logits, sampling_metadata: NS(
                sampled_token_ids=logits.argmax(-1), logprobs_tensors=None)))


# Rows: a (no drafts), b (drafts [3, 1]: first accepted, second rejected), c.
logits = torch.zeros(5, 6)
for row, token in enumerate([2, 3, 4, 0, 5]):
    logits[row, token] = 1.0
stub = verify_stub(greedy={"a", "b", "c"})
sampled, _ = Runner._sample_with_drafts(stub, logits, {"b": [3, 1]})
check("greedy draft verification", sampled == [2, [3, 4], 5])


def sampled_first_token_error(q, num_trials=20000):
    """Largest deviation of the emitted first/second token frequencies from
    the target p when the draft is drawn from q (one-hot if q is None)."""
    stub = verify_stub()
    stub._verify_sampled_drafts = lambda *args: Runner._verify_sampled_drafts(
        stub, *args)
    generator = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 6, generator=generator)
    p = logits.softmax(-1)
    first, second, accepted = torch.zeros(6), torch.zeros(6), 0
    for _ in range(num_trials):
        draft = 2 if q is None else int(
            torch.multinomial(q, 1, generator=generator))
        out = stub._verify_sampled_drafts(
            logits, [draft], 1, None if q is None else [q])
        first[out[0]] += 1
        if len(out) == 2:
            second[out[1]] += 1
            accepted += 1
    return max(float((first / num_trials - p[0]).abs().max()),
               float((second / accepted - p[1]).abs().max()))


check("speculative sampling follows p, one-hot draft",
      sampled_first_token_error(None) < 0.02)
# q covers a shorter draft vocabulary than p, as the MTP draft lm_head does.
check("speculative sampling follows p, sampled draft",
      sampled_first_token_error(torch.tensor([0.5, 0.1, 0.3, 0.1])) < 0.02)

print("FAIL: " + ", ".join(failures) if failures else "logic checks passed")
sys.exit(1 if failures else 0)
