# Work Plan: vLLM OpenVINO Backend - V1-Only Migration (Current: v0.13.0, V1 enabled)

**Goal**: Adapt the `vllm-openvino` backend to be fully compatible with vLLM's V1 engine (current version v0.13.0) while preserving core functionality of serving OpenVINO IR models directly. Ensure metrics compatibility. Remove V0 legacy code to avoid future maintenance burden.

**Current Reality**:
- `pyproject.toml` already specifies `vllm==0.13.0` (from git rev v0.13.0).
- `Containerfile` already sets `VLLM_USE_V1=1`.
- Codebase contains both **V0** (`worker/openvino_worker.py`) and **V1** (`worker_v1/openvino_worker_v1.py`) implementations.
- `platform.py` conditionally selects worker via `VLLM_USE_V1`; V0 path remains but is incompatible with upstream v0.13.0.
- No local test suite exists in this repository; validation must rely on **integration testing** (manual server runs and metrics check) as per user's CI/CD method.

**Key Decisions**:
- **Final Architecture**: V1-only. Remove V0 worker class and migrate any shared logic (e.g., cache engine) to V1-compatible modules.
- **Dependency Management**: No changes needed; v0.13.0 already set. Verify dependencies resolve correctly.
- **Metrics**: Must use V1's built-in metrics system; ensure custom OpenVINO metrics (if any) are registered via V1 APIs.
- **Testing**: Integration-based validation: build container, start server, issue a generation request, verify metrics endpoint.

---

## Phase 1: Current State Assessment & Gap Analysis

### Task 1.1: Inventory V0 vs V1 Components
**Objective**: [x] Identify all V0-specific code that must be removed or refactored.

**Actions**:
1. Read `vllm_openvino/platform.py` lines 77-83 to see the selection logic (if VLLM_USE_V1, use V1; else V0).
2. List files:
   - V0 worker: `vllm_openvino/worker/openvino_worker.py` (contains `OpenVINOWorker` and `OpenVINOCacheEngine`).
   - V1 worker: `vllm_openvino/worker_v1/openvino_worker_v1.py` (contains `OpenVINOWorkerV1`).
   - V0 model runner: `vllm_openvino/worker/openvino_model_runner.py` (if unused).
   - V1 model runner: `vllm_openvino/worker_v1/openvino_model_runner_v1.py`.
   - Attention backend: `vllm_openvino/attention/backends/openvino.py` (shared? likely V1-compatible already).
3. Check imports: `OpenVINOWorkerV1` imports `OpenVINOCacheEngine` from the V0 worker file (line 29). This indicates legacy cache engine is used by V1.

**Deliverable**: `assessment_v0_v1_inventory.md` listing:
- Which files are V0-only, V1-only, or shared.
- Which V0 symbols are imported by V1 code (e.g., `OpenVINOCacheEngine`).

---

### Task 1.2: Identify V1 Compatibility Gaps
**Objective**: [x] Determine what changes are needed to make V1 backend fully V1-compliant.

**Actions**:
1. Review V1 base classes:
   - Worker: `vllm.v1.worker.worker_base.WorkerBase`
   - KV cache: `vllm.v1.kv_cache_interface.KVCache` (and `KVCacheSpec`, `KVCacheConfig`)
   - Model runner: `vllm.v1.worker.gpu_model_runner.GPUModelRunner` (or appropriate base)
   - Scheduler output: `vllm.v1.core.sched.output.SchedulerOutput`
   - Sampling: `vllm.v1.sample.SamplerV1`, `vllm.v1.sample.metadata.SamplingMetadataV1`
2. Check `OpenVINOWorkerV1` implementation:
   - Does it correctly implement all abstract methods of `WorkerBase`?
   - Does it use `KVCache` via `OpenVINOCacheEngine`? If so, is `OpenVINOCacheEngine` implementing `KVCache` interface? Likely not; it's a custom V0-style engine.
3. Check attention backend: `OpenVINOAttentionBackend` - does its `forward` signature match V1 expectations? It likely needs to accept `KVCache` or block tables correctly.
4. Check metric registration: V1 automatically collects metrics from `vllm.metrics`. Verify any custom metrics are registered.

**Deliverable**: `v1_compatibility_gaps.md` listing:
- Missing `KVCache` implementation for `OpenVINOCacheEngine`.
- Any signature mismatches in worker, model runner, or attention.
- Metrics registration status.

---

## Phase 2: Refactor to V1-Only Architecture

### Task 2.1: Extract and Adapt Cache Engine to V1 KVCache Interface
**Objective**: [x] Convert `OpenVINOCacheEngine` into a proper `KVCache` implementation usable by V1.

**Actions**:
1. Create new module: `vllm_openvino/kv_cache.py`.
2. Move the `OpenVINOCacheEngine` class (from `worker/openvino_worker.py`) into this new module.
3. Modify `OpenVINOCacheEngine` to implement `vllm.v1.kv_cache_interface.KVCache`:
   - Implement methods: `get_k_cache()`, `get_v_cache()`, `copy(src_cache, src_blocks, dst_cache, dst_blocks)`, `clone()`, `grow(additional_blocks)`, `get_slot_kv_cache(slot)` if needed.
   - The cache engine should still allocate OpenVINO tensors for K/V caches; the interface abstracts the operations.
4. Update `worker_v1/openvino_worker_v1.py` to import `OpenVINOCacheEngine` from `vllm_openvino.kv_cache` instead of `vllm_openvino.worker.openvino_worker`.
5. In `OpenVINOWorkerV1.initialize()`, instantiate `OpenVINOCacheEngine` as the `self.kv_cache` object that satisfies `KVCache` interface.
6. Remove `OpenVINOCacheEngine` definition from `worker/openvino_worker.py` (leaving only V0 worker class until deletion in later task).

**Acceptance Criteria**:
- `OpenVINOCacheEngine` is imported from `vllm_openvino.kv_cache` by `OpenVINOWorkerV1`.
- No import errors; type checks pass against `KVCache` protocol (if used).
- Cache allocation and block management still function as before.

**QA**:
- Static analysis: `grep` for remaining imports of `OpenVINOCacheEngine` from old path.
- Quick manual test: start server, check logs for cache initialization messages.

---

### Task 2.2: Remove V0 Worker Class and Associated Files
**Objective**: Eliminate V0-only code to avoid accidental usage and simplify codebase.

**Actions**:
1. Delete `vllm_openvino/worker/openvino_worker.py` (after extracting cache engine to `kv_cache.py`).
2. Delete `vllm_openvino/worker/openvino_model_runner.py` if it's V0-specific and not referenced anywhere else. Verify no imports of `OpenVINOModelRunner` exist before deletion.
3. Clean up any other V0-only modules (e.g., `vllm_openvino/worker/__init__.py` if it only exported V0 classes).
4. Optionally, delete the entire `vllm_openvino/worker/` directory if it becomes empty; keep `worker_v1/` as the sole worker package.

**Acceptance Criteria**:
- No Python file imports from `vllm_openvino.worker.openvino_worker` (except maybe in tests, but no tests exist).
- `vllm_openvino.worker` package can be removed or repurposed.

**QA**:
- Run `grep -r "vllm_openvino.worker.openvino_worker" .` to confirm no references.
- Import `vllm_openvino` without errors.

---

### Task 2.3: Update Platform to V1-Only
**Objective**: Remove V0 fallback logic; enforce V1 worker selection.

**Actions**:
1. Edit `vllm_openvino/platform.py`:
   - In `check_and_update_config`, remove the `else` branch that sets `parallel_config.worker_cls` to V0 path.
   - Ensure it only sets:
     ```python
     parallel_config.worker_cls = "vllm_openvino.worker_v1.openvino_worker_v1.OpenVINOWorkerV1"
     ```
   - Remove any comments about V0.
2. Optionally, remove the `VLLM_USE_V1` environment variable check from platform if it's no longer needed; but vLLM core may still read it. Keep side effects harmless (e.g., ignore).
3. Confirm that `supports_v1` returns `True` (already does).

**Acceptance Criteria**:
- `platform.py` unconditionally selects V1 worker class.
- No references to `vllm_openvino.worker.openvino_worker` remain.

**QA**:
- Read `platform.py` to verify.
- Start server; check logs to ensure V1 worker is used.

---

### Task 2.4: Verify Attention Backend V1 Compatibility
**Objective**: Ensure `OpenVINOAttentionBackend` matches V1 expectations.

**Actions**:
1. Read `vllm/attention/backends/__init__.py` in the upstream v0.13.0 source to see V1 attention base class and required methods.
2. If needed, adjust `OpenVINOAttentionBackend.forward` signature to accept V1's `PagedAttention` parameters (likely includes `query`, `key`, `value`, `kv_cache` access via `KVCache`, `block_tables`, etc.).
3. Use `get_attn_backend` from `vllm.attention` (V1 version) - platform already calls it correctly.
4. Remove any V0-specific attributes or imports from `attention/backends/openvino.py`.

**Acceptance Criteria**:
- Attention backend compiles and works with V1's worker calls.
- No V0 attention imports (e.g., from `vllm.attention.ops` that were removed).

**QA**:
- Grep for V0-specific attention symbols (e.g., `AttentionBackend` from V0).
- Run a generation to verify attention computation yields correct shapes.

---

### Task 2.5: Model Runner V1 Finalization
**Objective**: Confirm `OpenVINOModelRunnerV1` is fully V1-compliant.

**Actions**:
1. Verify base class: it should inherit from `vllm.v1.worker.gpu_model_runner.GPUModelRunner` or similar. If not, adjust.
2. Check `prepare_inputs`: it should accept `InputBatch` (V1) and produce tensors for OpenVINO.
3. Check `execute`: should return `ModelRunnerOutput` (V1) with `logits` and optionally `hidden_states`.
4. Ensure sampler usage: uses `vllm.v1.sample.SamplerV1` and `SamplingMetadataV1`.
5. Confirm `get_model` returns a model compatible with V1's expectations (likely still a `torch.nn.Module` compiled to OpenVINO IR).

**Acceptance Criteria**:
- `OpenVINOModelRunnerV1` compiles under v0.13.0.
- All V1 abstract methods implemented.

**QA**:
- Static type checking/linting.
- Quick import test: `from vllm_openvino.worker_v1.openvino_model_runner_v1 import OpenVINOModelRunnerV1`.

---

## Phase 3: Metrics Compatibility

### Task 3.1: Verify V1 Metrics System Integration
**Objective**: Ensure OpenVINO backend correctly emits metrics under V1.

**Actions**:
1. Review vLLM V1 metrics module (`vllm.metrics`). Understand which metrics are automatically recorded (e.g., `vllm:iteration_tokens`, `vllm:request_latency`).
2. In `OpenVINOWorkerV1`, confirm that the base class `WorkerBase` already handles metrics instrumentation. If custom OpenVINO-specific metrics are needed (e.g., OpenVINO inference time), register them using `vllm.metrics.add_metric` or similar.
3. Ensure that when the server runs with `VLLM_USE_V1=1`, the `/metrics` endpoint is exposed by `vllm.entrypoints.openai.api_server` and includes vLLM core metrics plus any custom ones.

**Acceptance Criteria**:
- When server is running, `curl http://localhost:8000/metrics` returns `vllm_*` metrics.
- No V0 metric references (e.g., from `vllm.logger` stats) remain.

**QA**:
- Manual: start server, hit `/metrics`, check for `vllm_` prefix metrics.
- Compare output with a vanilla V1 vLLM server (if available) to ensure naming/structure matches.

---

## Phase 4: Integration Validation (No Local Pytest)

### Task 4.1: Build Container Image
**Objective**: Rebuild container with up-to-date code and V1 enforcement.

**Actions**:
1. Run `podman build -t vllm-0.13.0-ov .` in project root.
2. Verify build completes without errors and image size is reasonable (compare to previous if possible).

**Acceptance Criteria**:
- Image `vllm-0.13.0-ov` created successfully.
- No build errors related to missing dependencies or version conflicts.

**QA**:
- `podman images` shows the image.
- Build logs show `VLLM_USE_V1=1` and `vllm==0.13.0`.

---

### Task 4.2: Smoke Test - Model Load and Generation
**Objective**: Verify that an OpenVINO IR model can be loaded and served.

**Actions**:
1. Run container with port mapping:
   ```bash
   podman run -d --name vllm-test -p 8000:8000 vllm-0.13.0-ov
   ```
2. Wait for server to be ready (check logs: `podman logs -f vllm-test` for "Uvicorn running").
3. Prepare a very small OpenVINO IR model (e.g., from Optimum Intel export of `gpt2` or a tiny Llama variant). Place in a host directory, e.g., `/tmp/ov_model`.
4. Start server inside container with model mount:
   ```bash
   podman run -d --name vllm-test -p 8000:8000 \
     -v /tmp/ov_model:/model \
     vllm-0.13.0-ov \
     --model /model
   ```
5. Send a generate request using `curl`:
   ```bash
   curl http://localhost:8000/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "/model", "prompt": "Hello", "max_tokens": 5}'
   ```
6. Verify response contains non-empty `choices[0].text`.

**Acceptance Criteria**:
- Server starts without import errors or V0-related warnings.
- Model loads successfully (log shows "Loading model from /model").
- Request returns a JSON with generated text.

**QA**:
- HTTP 200 response.
- Response body includes `choices` array with `text`.
- No stack traces in server logs.

---

### Task 4.3: Metrics Sanity Check
**Objective**: Confirm metrics endpoint outputs V1 metrics.

**Actions**:
1. With server running from Task 4.2, run: `curl http://localhost:8000/metrics`.
2. Look for metrics with `vllm_` prefix, e.g., `vllm:iteration_tokens`, `vllm:request_latency`, `vllm:generation_tokens`.
3. Verify that the metrics are being updated (generate a few more requests, re-run `curl`, see counters increase).

**Acceptance Criteria**:
- At least 5 `vllm_*` metrics present.
- Counters increment after additional requests.

**QA**:
- Record output and check for monotonic increases.

---

### Task 4.4: Container Cleanup
**Objective**: Clean up test container.

**Actions**:
```bash
podman rm -f vllm-test
```

**Acceptance Criteria**: Container removed.

---

## Phase 5: Documentation Update

### Task 5.1: Update README and Environment Variables
**Objective**: Document that V1 is now the only supported engine.

**Actions**:
1. In `README.md`, remove or update references to `VLLM_USE_V1` and V0 fallback. State clearly that V1 is the default and only supported mode.
2. Update performance tips if any V0-specific parameters changed.
3. Add a note about the migration: V0 code removed, V1 metrics enabled.

**Acceptance Criteria**:
- Documentation accurately reflects current behavior.
- Users understand that V1 is mandatory.

---

### Task 5.2: Create Migration Notes (Optional)
**Objective**: Provide context for future maintainers.

**Actions**:
Create a file `docs/migration_v0_to_v1.md` summarizing:
- What V0 code was removed.
- How the cache engine was adapted to V1's `KVCache` interface.
- Any pitfalls encountered.

(This is optional but recommended for knowledge retention.)

---

## Final Verification Checklist

- [ ] All V0 worker code deleted (`OpenVINOWorker`).
- [ ] `OpenVINOCacheEngine` moved to `kv_cache.py` and implements `KVCache` methods.
- [ ] `platform.py` unconditionally selects `OpenVINOWorkerV1`.
- [ ] No imports of `vllm_openvino.worker.openvino_worker` remain.
- [ ] Attention backend signature matches V1.
- [ ] Model runner V1 correctly uses `InputBatch` and returns `ModelRunnerOutput`.
- [ ] Container builds successfully.
- [ ] Server starts and serves an OpenVINO IR model with text generation.
- [ ] `/metrics` endpoint returns `vllm_*` metrics.
- [ ] Documentation updated.

---

## Risks and Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| `OpenVINOCacheEngine` V1 adaptation breaks cache semantics | High | Medium | Thoroughly review V1 `KVCache` contract; run integration test to verify generation quality matches pre-migration baseline. |
| Attention backend signature mismatch leads to runtime errors | Medium | Low | Cross-check with V1 base before changes; test with small model. |
| Metrics not propagating to `/metrics` endpoint | Medium | Low | Verify `WorkerBase` instrumentation; manually inspect metric registry. |
| Deleting V0 code accidentally removes needed logic | Medium | Low | Ensure any shared logic is extracted to `kv_cache.py`; keep V0 code until V1 fully validated. |

---

## Decisions Summary

| Decision | Options Considered | Chosen Path | Rationale |
|----------|-------------------|-------------|-----------|
| Codebase state | Upgrade from v0.10.2 → Clean up mixed state | Clean up mixed state (already v0.13.0) | Current state already at v0.13.0 with both V0 and V1; goal is to remove V0. |
| Test strategy | Add local pytest suite → Use integration tests | Use integration tests (container-based) | No local tests exist; adding them is out of scope; integration tests align with user's CI/CD. |
| Cache engine placement | Keep in `worker/` → Move to `kv_cache.py` | Move to `kv_cache.py` | Decouples cache logic from V0 worker; allows reuse by V1 without circular imports. |
| V0 code removal | Keep as fallback → Delete | Delete (after extraction) | V0 unsupported upstream; keeping it would cause confusion and maintenance burden. |
| Validation | Unit tests → Manual smoke test | Manual smoke test + metrics check | No unit tests available; smoke test sufficient to verify end-to-end functionality. |

---

**Success Criteria**:
- Server runs with V1-only backend, serves OpenVINO IR models end-to-end.
- Metrics endpoint emits V1-compatible metrics.
- No V0 code remains in the repository.
- Container image builds and runs as expected.
