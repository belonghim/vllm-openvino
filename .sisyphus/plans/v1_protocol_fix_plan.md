# Fix Plan: V1 Protocol Compliance & Dead Code Removal

**Status**: ✅ COMPLETE (2026-03-06)
**Parent Plan**: `.sisyphus/plans/vllm_upgrade_plan.md` (Phase 2-4 corrections)
**Goal**: Fix issues discovered during direction validation that prevent the smoke test from passing. The overall V1-only architecture direction is correct; the implementation has protocol mismatches, dead code, and wrong signatures.

**Key Architecture Decision**: Drop `GPUModelRunner` inheritance from `OpenVINOModelRunnerV1`. The upstream class is 5,470 lines of CUDA-specific code with ~10 hard crash points in `__init__`. OpenVINO only needs `self.requests`, `self.input_batch`, and `_update_states` pattern (~200 lines).

---

## Result

**End-to-end inference verified** in containerized environment:
- Model: `Qwen2.5-Coder-3B-Instruct-int4-ov`
- Server: vLLM v0.13.0 + OpenVINO backend via podman
- Inference: ✅ HTTP 200, coherent text generation
- Metrics: ✅ `/metrics` returns 537 lines, 20+ `vllm:` prefixed metrics
- Stability: ✅ Multiple consecutive requests without crash

---

## All Resolved Errors (9 total)

| # | Error | File | Fix | Commit |
|---|-------|------|-----|--------|
| 1 | `AssertionError: Invalid kv_cache_dtype: u8` | `kv_cache.py` | `get_attn_backend()` 제거, `OpenVINOAttentionBackend` 직접 사용 | `69cf56d` |
| 2 | `TypeError: bind_kv_cache() missing argument` | `openvino_worker_v1.py` | v0.13.0 시그니처 `bind_kv_cache({}, ctx, [])` (3개 인자) | `69cf56d` |
| 3 | `NotImplementedError: 'get_supported_tasks'` | `openvino_worker_v1.py` | `get_supported_tasks()` → `return ('generate',)` | `a19127a` |
| 4 | `AttributeError: MultiModalKwargs.as_kwargs` | `openvino_model_runner_v1.py` | `MultiModalKwargs.as_kwargs(...)` 제거 | `12a1292` |
| 5 | `AssertionError: not (all_greedy and all_random)` | `openvino_model_runner_v1.py` | `refresh_metadata()` 추가 | `12a1292` |
| 6 | `InductorError: InvalidCxxCompiler` | `platform.py` | `CompilationMode.NONE`, `level=0`; `TORCH_COMPILE_DISABLE=1` 필수 | `ac80a15` |
| 7 | `TypeError: unexpected keyword 'spec_token_ids'` | `openvino_model_runner_v1.py` | `spec_token_ids` 제거 | `12a1292` |
| 8 | `TypeError: missing argument 'pooler_output'` | `openvino_model_runner_v1.py` | `pooler_output=None` 추가 | `12a1292` |
| 9 | `AssertionError: Invalid query_lens: [0]` | `openvino_model_runner_v1.py` | `_prepare_inputs`를 `InputBatch.token_ids_cpu`에서 읽도록 재작성 + 샘플링 후 writeback 추가 | `12a1292` |

---

## Commits

```
d418709 chore: add .gitignore and planning artifacts
12a1292 fix: rewrite _prepare_inputs to read tokens from InputBatch buffer
a19127a fix: align OpenVINO worker with vLLM v0.13.0 V1 protocol
ac80a15 fix: disable torch.compile and dynamo for OpenVINO backend
```

---

## Hot Patching Strategy (Reference)

```bash
podman run -d --name vllm-patch-tmp --entrypoint sleep vllm-0.13.0-ov-patched 3600
podman cp /home/user/project/vllm-openvino/vllm_openvino vllm-patch-tmp:/opt/app-root/
podman commit \
  --change='ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]' \
  vllm-patch-tmp vllm-0.13.0-ov-patched
podman rm -f vllm-patch-tmp
podman run --replace -d --name vllm-server \
  -p 8000:8000 -v ~/hf:/models:Z \
  -e VLLM_OPENVINO_DEVICE=CPU \
  -e VLLM_OPENVINO_KVCACHE_SPACE=8 \
  -e TORCH_COMPILE_DISABLE=1 \
  vllm-0.13.0-ov-patched \
  --model /models/Qwen2.5-Coder-3B-Instruct-int4-ov \
  --dtype auto --max-model-len 2048 --host 0.0.0.0 --port 8000
```

---

## Remaining Work (Not blocking — future cleanup)

These items from the original plan were NOT completed because they are code cleanup tasks, not runtime blockers. The server works without them.

| Task | Description | Priority |
|------|-------------|----------|
| Task 1 | Remove `VLLM_USE_V1` references from model loader | Low |
| Task 2 | Remove dead V0 `OpenVINOWorker` class from `kv_cache.py` | Low |
| Task 5 | Fix `OpenVINOAttentionMetadataBuilder.build()` | Low |
| Task 6 | Clean up Containerfile, README, dead imports | Low |
