# Migration Notes: V0 to V1 Engine

## Overview

The OpenVINO backend was migrated from vLLM's V0 engine to V1-only architecture for compatibility with vLLM v0.13.0+.

## What Changed

### Removed
- V0 `OpenVINOWorker` class (was in `worker/openvino_worker.py`)
- V0 `OpenVINOModelRunner` class (was in `worker/openvino_model_runner.py`)
- `VLLM_USE_V1` environment variable (V1 is now the only engine)
- `GPUModelRunner` inheritance from `OpenVINOModelRunnerV1`

### Moved
- `OpenVINOCacheEngine` → `vllm_openvino/kv_cache.py` (extracted from V0 worker)
- `ModelInput` → `vllm_openvino/model.py` (extracted from V0 model runner)

### New Architecture
- `OpenVINOModelRunnerV1` is standalone (no GPU inheritance)
- Token handling reads from `InputBatch.token_ids_cpu` buffer directly
- Post-sampling writeback updates `token_ids_cpu`, `num_tokens`, `output_token_ids`, and `num_computed_tokens_cpu`
- `platform.py` unconditionally selects `OpenVINOWorkerV1`

### Required Environment Variables
- `TORCH_COMPILE_DISABLE=1` — Disables torch.compile/Inductor (incompatible with OpenVINO)
- `VLLM_OPENVINO_DEVICE=CPU` — Device selection (default: CPU)
- `VLLM_OPENVINO_KVCACHE_SPACE=8` — KV cache size in GB

## Key Fixes (vLLM v0.13.0 API Changes)

| API Change | Fix |
|-----------|-----|
| `bind_kv_cache()` now takes 3 args | `bind_kv_cache({}, ctx, [])` |
| `get_supported_tasks()` required | Returns `('generate',)` |
| `MultiModalKwargs.as_kwargs()` removed | Removed call |
| `ModelRunnerOutput` no `spec_token_ids` | Removed field |
| `ModelRunnerOutput` requires `pooler_output` | Added `pooler_output=None` |
| `InputBatch.refresh_metadata()` required | Added call before `_prepare_inputs` |
| `CompilationMode.NONE` needed | Set in `platform.py` |
