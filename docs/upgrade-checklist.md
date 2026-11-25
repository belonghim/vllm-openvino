# vLLM Upgrade Checklist

This plugin uses vLLM V1 internal APIs. Review the following areas whenever
the supported vLLM version changes.

## Before Upgrading

Catch breaking upstream changes before the release: when a PR that touches the
interfaces below is merged, build an image from that commit's wheel (listed at
`https://wheels.vllm.ai/<commit>/vllm/`, which links to the wheel one level
up; wheels appear about an hour after the merge) and run
`scripts/smoke-test.sh` against it:

```bash
podman build -f Containerfile -t vllm-openvino:next \
  --build-arg VLLM_SPEC="https://wheels.vllm.ai/<commit>/vllm-<version>-cp38-abi3-manylinux_2_28_x86_64.whl" .
IMAGE=vllm-openvino:next ./scripts/smoke-test.sh
```

1. Read the vLLM release notes.
2. Compare the imported `vllm.v1.*` modules and public configuration APIs.
3. Check worker, model runner, scheduler output, attention metadata, and
   KV-cache interface signatures.
4. Check platform registration and worker selection.
5. The plugin runs its own runner and keeps its own copies of the MRV1 pieces
   it used; the runner and worker accept the supported version's scheduler
   output only (see "Upstream vLLM Compatibility" in `compatibility.md`), so
   drop a compatibility branch once its version is no longer supported. The
   plugin consumes `WorkerBase`, `bind_kv_cache`, `SchedulerOutput`,
   `ModelRunnerOutput` and other `vllm.*` symbols without running the built-in
   runner, and borrows private M-RoPE methods from vLLM's Qwen3-VL and
   MiniCPM-V models. `check-vllm-imports.py` resolves every such import,
   method and dataclass field (add new ones to its `ATTRS`); the smoke test
   runs it, or check a new wheel directly:

   ```bash
   podman run --rm -v "$PWD:/w:Z" -w /w --entrypoint python <new-image> scripts/check-vllm-imports.py
   ```

High-risk interfaces include:

- `vllm.v1.kv_cache_interface`
- `vllm.v1.attention.backend`
- `vllm.v1.attention.backends.utils`
- `vllm.v1.outputs`
- `vllm.v1.worker.worker_base`
- `vllm.v1.worker.utils`
- `vllm.v1.core.sched.output`

## Source Compatibility

Verify every overridden or consumed symbol in:

- `platform.py`
- `attention/backends/openvino.py`
- `model_executor/model_loader/openvino.py`
- `worker_v1/openvino_worker_v1.py`
- `worker_v1/openvino_model_runner_v1.py`
- `kv_cache.py`

`sampler.py`, `worker_v1/input_batch.py` and `worker_v1/ngram_proposer.py` are
copies of vLLM 0.32's V1 sampler, `InputBatch` and `NgramProposer`. They do
not break when vLLM changes, but compare them with the new release's sampling
semantics (new `SamplingParams` fields, changed defaults). The runner's
`apply_grammar_bitmask` reads the row layout produced by
`StructuredOutputManager.grammar_bitmask`; diff it against vLLM's
`apply_grammar_bitmask` (0.33 moved to a fixed stride per request).

Pay particular attention to `WorkerBase`, `ModelRunnerOutput`,
`SchedulerOutput`, `AttentionMetadata`, `KVCacheSpec`, `FullAttentionSpec`,
`bind_kv_cache`, and `CompilationTimes`.

## Version Pins

Update every place that names a version:

- `pyproject.toml`: `version`, `vllm`, `transformers`, `openvino`
- `Containerfile`: the `VLLM_SPEC` default, `torch`, `torchvision`, `openvino`,
  and `transformers`
- `AGENTS.md` and `CONTRIBUTING.md`: vLLM and OpenVINO versions
- `docs/compatibility.md`: Requirements and the version-specific notes

## OpenVINO Upgrades

The image pins OpenVINO nightly `2026.5.0.dev20261030` from
`storage.openvinotoolkit.org/simple/wheels/nightly` for the MoE prefill fix
(#39086) and 2026.5-exported IRs (LFM3). Move to the 2026.5 release once it
ships and drop the nightly index. When upgrading, re-check LFM2.5-8B-A1B
prefill TTFT. Keep a copy of the pinned nightly wheel
(`pip download --no-deps openvino==<version>` with the nightly index): the
image cannot be rebuilt once the index drops it.

## Validation

1. Run `python3 -m py_compile` on changed Python files.
2. Run the source-mounted Podman test from `podman-testing.md`.
3. Verify server startup and a real API response.
4. For runtime changes, verify single, sequential, and concurrent requests.
5. Compare logs for import errors, signature errors, cache allocation errors,
   and unexpected path selection.
6. Run `scripts/smoke-test.sh`. It first runs `check-vllm-imports.py` and
   `check-logic.py` (the runner's model-free logic against brute-force
   references) in the image, then starts each serving path (PagedAttention,
   Hybrid-PA with block checkpoints, Hybrid-PA with prefix caching off and
   MTP drafts) and checks greedy sequential throughput and an output hash
   against the baseline and a concurrent burst. `OUTPUT CHANGED` is expected
   after an OpenVINO, vLLM or IR change (review the text, then
   `--update-baseline`); with none of those it is a correctness regression.
   It then forces preemption with a small pool and checks
   that every resumed request completes, and runs a same-step prefix hit next
   to MTP drafts on Hybrid-PA. Set `IMAGE=<candidate-image>` to run
   it against an image built for the new version. Throughput moves with host
   load; compare against the previous commit on the same host before calling
   a drop a regression.
7. Repeat it with `SMOKE_ENV=VLLM_USE_RUST_FRONTEND=1`, and send an image
   request (Qwen3.5, Gemma-4) and a `prompt_logprobs` request under it. The
   Rust frontend has its own tokenizer, chat templates and multimodal specs
   and does not call `Platform.validate_request`; vLLM plans to make it the
   default, so record what changed in `compatibility.md` (Known Limitations).
