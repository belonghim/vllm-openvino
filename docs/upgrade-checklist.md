# vLLM Upgrade Checklist

This plugin uses vLLM V1 internal APIs. Review the following areas whenever
the supported vLLM version changes.

## Before Upgrading

1. Read the vLLM release notes.
2. Compare the imported `vllm.v1.*` modules and public configuration APIs.
3. Check worker, model runner, scheduler output, attention metadata, and
   KV-cache interface signatures.
4. Check platform registration and worker selection.

High-risk interfaces include:

- `vllm.v1.kv_cache_interface`
- `vllm.v1.attention.backend`
- `vllm.v1.attention.backends.utils`
- `vllm.v1.outputs`
- `vllm.v1.sample.metadata`
- `vllm.v1.sample.sampler`
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

## Validation

1. Run `python3 -m py_compile` on changed Python files.
2. Run the source-mounted Podman test from `podman-testing.md`.
3. Verify server startup and a real API response.
4. For runtime changes, verify single, sequential, and concurrent requests.
5. Compare logs for import errors, signature errors, cache allocation errors,
   and unexpected path selection.
