# Plan: Low-Priority Dead Code Cleanup

**Status**: Ready for execution
**Tier**: Trivial — 5 tasks, all independent, single-commit scope
**Goal**: Remove dead code, unused imports, and stale planning artifacts from the repo.

---

## Context

After completing the V1 migration (vllm_upgrade_plan + v1_protocol_fix_plan), several dead artifacts remain. Exploration confirmed:

- `kv_cache.py` has **NO** dead V0 class (only 239 lines of active `OpenVINOCacheEngine`)
- `VLLM_USE_V1` has **zero** occurrences in source code (only in docs, which is correct)
- `AttentionMetadataBuilder.build()` correctly raises `NotImplementedError` — no fix needed
- The actual cleanup items are smaller than originally described

---

## Constraints

- Hot patching protocol: after source changes, use `podman run/cp/commit/rm/run`
- `TORCH_COMPILE_DISABLE=1` required
- All changes are safe deletions — no runtime behavior changes

---

## Tasks

### Task 1: Delete dead `model.py`

**File**: `vllm_openvino/model.py` (26 lines)
**Action**: Delete the entire file.
**Why**: `ModelInput` NamedTuple is a V0 artifact. Grep confirms it is **never imported** by any other file in the codebase.
**Verification**: `grep -r "from vllm_openvino.model import\|from vllm_openvino import model\|vllm_openvino.model.ModelInput" vllm_openvino/` returns 0 results.

### Task 2: Remove dead import `SamplingType`

**File**: `vllm_openvino/worker_v1/openvino_model_runner_v1.py`
**Action**: Delete line 13: `from vllm.sampling_params import SamplingType`
**Why**: `SamplingType` is imported but never referenced anywhere else in the file. Grep confirms only 1 occurrence (the import itself).
**Verification**: `grep -n "SamplingType" vllm_openvino/worker_v1/openvino_model_runner_v1.py` returns 0 results after deletion.

### Task 3: Delete stale planning artifacts from repo root

**Files to delete**:
- `assessment_v0_v1_inventory.md` — V0/V1 assessment doc, superseded by completed plans
- `v1_compatibility_gaps.md` — Gap analysis doc, superseded by completed plans

**Action**: `rm assessment_v0_v1_inventory.md v1_compatibility_gaps.md`
**Why**: These were working documents created during the planning phase. All findings have been incorporated into the completed plans and migration docs.
**Verification**: `ls *.md` in repo root should show only `README.md` and `LICENSE.md`.

### Task 4: Delete stale draft

**File**: `.sisyphus/drafts/vllm_upgrade_plan.md`
**Action**: `rm .sisyphus/drafts/vllm_upgrade_plan.md`
**Why**: Draft should have been deleted when the plan was finalized. The final plan is at `.sisyphus/plans/vllm_upgrade_plan.md`.
**Verification**: `.sisyphus/drafts/` directory should be empty.

### Task 5: Update "Remaining Cleanup" in upgrade plan

**File**: `.sisyphus/plans/vllm_upgrade_plan.md`
**Action**: Replace the "Remaining Cleanup" table (lines 80-87) with a note that all cleanup has been completed.

**Replace this**:
```
## Remaining Cleanup (Not blocking — Low priority)

| Item | File(s) | Description |
|------|---------|-------------|
| Remove `VLLM_USE_V1` refs | model loader, Containerfile, README | Dead env var from v0.13.0+ |
| Delete V0 class | `kv_cache.py` | `OpenVINOWorker` class (~391 lines) still present |
| Fix metadata builder | `attention/backends/openvino.py` | `build()` passes wrong fields (never called at runtime) |
| Dead imports | Various `.py` files | Stale V0 imports that don't affect runtime |
```

**With this**:
```
## Remaining Cleanup — ✅ Resolved

All cleanup items have been addressed by `cleanup_dead_code` plan:
- `VLLM_USE_V1` refs: Already absent from source (confirmed by grep)
- V0 class in `kv_cache.py`: Does not exist (file is 239 lines of active `OpenVINOCacheEngine`)
- Metadata builder: `build()` correctly raises `NotImplementedError` as V1 stub — no fix needed
- Dead imports: `SamplingType` removed from `openvino_model_runner_v1.py`
- Dead file: `model.py` deleted (unused `ModelInput` NamedTuple)
- Stale artifacts: `assessment_v0_v1_inventory.md`, `v1_compatibility_gaps.md` deleted
```

**Verification**: Read the file and confirm the section is updated.

---

## Final Verification Wave

After all tasks complete:
1. `git status` — confirm only expected deletions and edits
2. `grep -r "SamplingType" vllm_openvino/` — returns 0
3. `grep -r "ModelInput" vllm_openvino/` — returns 0
4. `ls *.md` in repo root — only `README.md` and `LICENSE.md`
5. Commit: `chore: remove dead code, unused imports, and stale planning artifacts`
6. `git push origin main`
