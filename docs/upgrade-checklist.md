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
   `InputBatch`, `CachedRequestState`, `bind_kv_cache` and other `vllm.v1.*`
   symbols without running the built-in runner; check them against the new
   wheel before bumping:

   ```bash
   podman run --rm -v "$PWD:/w:Z" -w /w --entrypoint python <new-image> scripts/check-vllm-imports.py
   ```

High-risk interfaces include:

- `vllm.v1.kv_cache_interface`
- `vllm.v1.attention.backend`
- `vllm.v1.attention.backends.utils`
- `vllm.v1.outputs`
- `vllm.v1.sample.metadata`
- `vllm.v1.sample.sampler` (`apply_all_penalties` is patched by `sampler_patch.py`)
- `vllm.v1.sample.ops.topk_topp_sampler` (`apply_top_k_top_p` is patched by `sampler_patch.py`; `forward_cpu` calls `torch.ops._C.greedy_argmax`/`fused_gumbel_argmax`, which `platform.py` registers because the wheel has no CPU kernels; re-check the `_C` ops `forward_cpu` uses after an upgrade)
- `vllm.v1.worker.gpu_input_batch`
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

Pay particular attention to `WorkerBase`, `ModelRunnerOutput`,
`SchedulerOutput`, `InputBatch`, `AttentionMetadata`, `KVCacheSpec`,
`MambaSpec`, `bind_kv_cache`, and `CompilationTimes`.

## Version Pins

Update every place that names a version:

- `pyproject.toml`: `version`, `vllm`, `transformers`, `openvino`
- `Containerfile`: the vLLM wheel URL, `torch`, `torchvision`, `openvino`,
  `transformers`, and the CPU plugin sha256
- `AGENTS.md` and `CONTRIBUTING.md`: vLLM and OpenVINO versions
- `docs/compatibility.md`: Requirements and the version-specific notes

## OpenVINO Upgrades

The image overwrites the wheel's CPU plugin with a patched build tied to
OpenVINO 2026.4.1. On an OpenVINO version change, follow `patches/README.md`
to drop or rebuild it.

## Validation

1. Run `python3 -m py_compile` on changed Python files.
2. Run the source-mounted Podman test from `podman-testing.md`.
3. Verify server startup and a real API response.
4. For runtime changes, verify single, sequential, and concurrent requests.
5. Compare logs for import errors, signature errors, cache allocation errors,
   and unexpected path selection.
