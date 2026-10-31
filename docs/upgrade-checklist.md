# vLLM Upgrade Checklist

This plugin uses vLLM V1 internal APIs. Review the following areas whenever
the supported vLLM version changes.

## Before Upgrading

1. Read the vLLM release notes.
2. Compare the imported `vllm.v1.*` modules and public configuration APIs.
3. Check worker, model runner, scheduler output, attention metadata, and
   KV-cache interface signatures.
4. Check platform registration and worker selection.
5. Check that vLLM still ships MRV1 (the V1-style model runner this plugin
   uses) and whether `Platform.supports_v2_model_runner` has merged; see
   "Upstream vLLM Compatibility" in `compatibility.md`. The plugin consumes
   `WorkerBase`, `bind_kv_cache`, `SchedulerOutput`, `ModelRunnerOutput` and
   other `vllm.v1.*` symbols without running the built-in runner; check them
   against the new wheel before bumping:

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
semantics (new `SamplingParams` fields, changed defaults).

Pay particular attention to `WorkerBase`, `ModelRunnerOutput`,
`SchedulerOutput`, `AttentionMetadata`, `KVCacheSpec`,
`MambaSpec`, `bind_kv_cache`, and `CompilationTimes`.

## Version Pins

Update every place that names a version:

- `pyproject.toml`: `version`, `vllm`, `transformers`, `openvino`
- `Containerfile`: the vLLM wheel URL, `torch`, `torchvision`, `openvino`,
  and `transformers`
- `AGENTS.md` and `CONTRIBUTING.md`: vLLM and OpenVINO versions
- `docs/compatibility.md`: Requirements and the version-specific notes

## OpenVINO Upgrades

The image pins OpenVINO nightly `2026.5.0.dev20261030` from
`storage.openvinotoolkit.org/simple/wheels/nightly` for the MoE prefill fix
(#39086) and 2026.5-exported IRs (LFM3). Move to the 2026.5 release once it
ships and drop the nightly index. When upgrading, re-check LFM2.5-8B-A1B
prefill TTFT and that vLLM still lacks LFM3 before keeping `register_models`.

## Validation

1. Run `python3 -m py_compile` on changed Python files.
2. Run the source-mounted Podman test from `podman-testing.md`.
3. Verify server startup and a real API response.
4. For runtime changes, verify single, sequential, and concurrent requests.
5. Compare logs for import errors, signature errors, cache allocation errors,
   and unexpected path selection.
6. Run `scripts/smoke-test.sh`. It starts each serving path (PagedAttention,
   Hybrid-PA, stateful) and checks sequential throughput against the baseline
   and a concurrent burst.
