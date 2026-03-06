# Fix Plan: V1 Protocol Compliance & Dead Code Removal

**Parent Plan**: `.sisyphus/plans/vllm_upgrade_plan.md` (Phase 2-4 corrections)
**Goal**: Fix issues discovered during direction validation that prevent the smoke test from passing. The overall V1-only architecture direction is correct; the implementation has protocol mismatches, dead code, and wrong signatures.

**Key Architecture Decision**: Drop `GPUModelRunner` inheritance from `OpenVINOModelRunnerV1`. The upstream class is 5,470 lines of CUDA-specific code with ~10 hard crash points in `__init__`. OpenVINO only needs `self.requests`, `self.input_batch`, and `_update_states` pattern (~200 lines).

**Execution Strategy**: 6 code fix tasks + 1 integration wave.

---

## ✅ Already Applied Fixes (Runtime Errors Resolved)

These changes are already in the local `vllm_openvino/` codebase:

| # | Error | File | Fix |
|---|-------|------|-----|
| 1 | `AssertionError: Invalid kv_cache_dtype: u8` | `kv_cache.py` | `get_attn_backend()` 제거, `OpenVINOAttentionBackend` 직접 사용 |
| 2 | `TypeError: bind_kv_cache() missing argument 'runner_kv_caches'` | `openvino_worker_v1.py` | v0.13.0 시그니처 `bind_kv_cache({}, ctx, [])` (3개 인자) |
| 3 | `NotImplementedError: 'get_supported_tasks' not implemented` | `openvino_worker_v1.py` | `get_supported_tasks()` 메서드 추가 → `return ('generate',)` |
| 4 | `AttributeError: 'MultiModalKwargs' has no attribute 'as_kwargs'` | `openvino_model_runner_v1.py` | `MultiModalKwargs.as_kwargs(...)` 제거 |
| 5 | `AssertionError: not (all_greedy and all_random)` | `openvino_model_runner_v1.py` | `refresh_metadata()` 추가 |
| 6 | `torch._inductor.exc.InductorError: InvalidCxxCompiler` | `platform.py` | `CompilationMode.NONE`, `level=0` 설정; **`TORCH_COMPILE_DISABLE=1` 환경변수 필수** |
| 7 | `TypeError: ModelRunnerOutput unexpected keyword 'spec_token_ids'` | `openvino_model_runner_v1.py` | `spec_token_ids` 제거 |
| 8 | `TypeError: ModelRunnerOutput missing argument 'pooler_output'` | `openvino_model_runner_v1.py` | `pooler_output=None` 추가 |

---

## ⚠️ Hot Patching Strategy (MANDATORY — No Full Rebuild)

**절대 `podman build`로 전체 이미지 재빌드 하지 말 것. 시간 낭비.**

올바른 방법 (from `vllm_upgrade_plan.md` Phase 4):

```bash
# Step 1: 임시 sleep 컨테이너 실행 (base image 사용)
podman run -d --name vllm-patch-tmp --entrypoint sleep vllm-0.13.0-ov-patched 3600

# Step 2: 수정된 코드를 컨테이너에 복사
podman cp /home/user/project/vllm-openvino/vllm_openvino vllm-patch-tmp:/opt/app-root/

# Step 3: 새 이미지로 커밋
podman commit \
  --change='ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]' \
  vllm-patch-tmp vllm-0.13.0-ov-patched

# Step 4: 임시 컨테이너 삭제 후 새 이미지로 실행
podman rm -f vllm-patch-tmp

podman run --replace -d --name vllm-server \
  -p 8000:8000 \
  -v ~/hf:/models:Z \
  -e VLLM_OPENVINO_DEVICE=CPU \
  -e VLLM_OPENVINO_KVCACHE_SPACE=8 \
  -e TORCH_COMPILE_DISABLE=1 \
  vllm-0.13.0-ov-patched \
  --model /models/Qwen2.5-Coder-3B-Instruct-int4-ov \
  --dtype auto --max-model-len 2048 --host 0.0.0.0 --port 8000
```

**장점**: 코드가 이미지에 bake-in되어 `__pycache__` 스테일 문제 없음. volume mount 불필요.

---

---

## Wave 1 — Quick Wins (Parallel, No Dependencies)

### Task 1: Remove `VLLM_USE_V1` from Model Loader
**Objective**: Eliminate all references to the removed `VLLM_USE_V1` environment variable in the model loader, which causes `AttributeError` at import time.

**File**: `vllm_openvino/model_executor/model_loader/openvino.py`

**Actions**:
1. **Line 9**: Remove `import vllm.envs as vllm_envs` (or keep only if used elsewhere — verify first).
2. **Line 160-163** (Sampler selection): Remove the `if vllm_envs.VLLM_USE_V1:` conditional. Always use `SamplerV1()`:
   ```python
   # BEFORE:
   if vllm_envs.VLLM_USE_V1:
       self.sampler = SamplerV1()
   else:
       self.sampler = Sampler()
   # AFTER:
   self.sampler = SamplerV1()
   ```
   Also remove the V0 `Sampler` import from line 20 if no longer used: `from vllm.model_executor.layers.sampler import Sampler, SamplerOutput` → remove `Sampler` (keep `SamplerOutput` if used elsewhere).

3. **Line 192-193** (gather transformation): Remove the `if vllm_envs.VLLM_USE_V1:` conditional. Always apply:
   ```python
   # BEFORE:
   if vllm_envs.VLLM_USE_V1:
       apply_gather_before_matmul_transformation(pt_model.model)
   # AFTER:
   apply_gather_before_matmul_transformation(pt_model.model)
   ```

4. **Line 224-225** (sampled_token_indices): Remove the `if vllm_envs.VLLM_USE_V1:` conditional. Always append:
   ```python
   # BEFORE:
   if vllm_envs.VLLM_USE_V1:
       inputs.append(attn_metadata.sampled_token_indices)
   # AFTER:
   inputs.append(attn_metadata.sampled_token_indices)
   ```

5. **Line 237-238** (`compute_logits`): Remove the `if not vllm_envs.VLLM_USE_V1:` conditional. Never call `_prune_hidden_states`:
   ```python
   # BEFORE:
   if not vllm_envs.VLLM_USE_V1:
       hidden_states = _prune_hidden_states(hidden_states, sampling_metadata)
   # AFTER:
   # (line removed — V1 always uses pre-gathered logits)
   ```
   Also remove the import of `_prune_hidden_states` from line 19 if no longer used.

6. **Verify**: Remove unused imports that become dead after these changes.

**Acceptance Criteria**:
- `grep -c 'VLLM_USE_V1' vllm_openvino/model_executor/model_loader/openvino.py` → `0`
- `python3 -m py_compile vllm_openvino/model_executor/model_loader/openvino.py` → success

**QA**:
- Static: no `vllm_envs.VLLM_USE_V1` references remain
- Syntax: file compiles without error

---

### Task 2: Remove Dead V0 `OpenVINOWorker` Class from `kv_cache.py`
**Objective**: Delete the entire V0 `OpenVINOWorker` class (~391 lines) that was accidentally left in `kv_cache.py`, along with its V0-only imports.

**File**: `vllm_openvino/kv_cache.py`

**Actions**:
1. **Verify no external imports**: Run `grep -r "from vllm_openvino.kv_cache import OpenVINOWorker" vllm_openvino/` — must return nothing.
2. **Delete lines 263-657**: Remove the entire `OpenVINOWorker` class (starts at `# --- OpenVINOWorker class ---` comment on line 264, class definition on line 266).
3. **Remove now-unused V0 imports** (verify each is truly unused before removing):
   - Line 26: `from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata` → remove entirely
   - Line 29: `from vllm.worker.worker_base import LoRANotSupportedWorkerBase, WorkerBase` → remove entirely
   - Line 14-18: `broadcast_tensor_dict`, `ensure_model_parallel_initialized`, `init_distributed_environment` → remove if unused by `OpenVINOCacheEngine`
   - Line 19: `from vllm.inputs import INPUT_REGISTRY` → remove if unused
   - Line 21: `from vllm.model_executor import set_random_seed` → remove if unused
   - Line 22: `from vllm.model_executor.layers.sampler import SamplerOutput` → remove if unused
   - Line 24: `from vllm.sampling_params import SamplingParams` → remove if unused
   - Line 28: `from vllm_openvino.worker_v1.openvino_model_runner_v1 import OpenVINOModelRunnerV1` → remove if unused
4. **Keep**: `OpenVINOCacheEngine` class (lines 48-258) and its required imports (`ov`, `torch`, `CacheConfig`, `ModelConfig`, `ParallelConfig`, `DeviceConfig`, `get_attn_backend`, `envs`, `KVCache`, `KVCacheSpec`, `current_platform`, `bind_kv_cache`).
5. **Verify**: File should be ~260 lines or fewer after cleanup.

**Acceptance Criteria**:
- `grep -c 'class OpenVINOWorker' vllm_openvino/kv_cache.py` → `0`
- `python3 -m py_compile vllm_openvino/kv_cache.py` → success
- `wc -l vllm_openvino/kv_cache.py` → less than 270

**QA**:
- No V0 class remains
- All remaining imports are used by `OpenVINOCacheEngine`
- No circular import issues

## Wave 2 — Architecture Refactor (Sequential, depends on Wave 1)

### Task 3: Refactor `OpenVINOModelRunnerV1` to Standalone (Drop `GPUModelRunner`)
**Objective**: Remove `GPUModelRunner` inheritance and implement a minimal standalone model runner that satisfies V1 contracts without CUDA dependencies.

**Architecture Decision**: Drop inheritance. GPUModelRunner is 5,470 lines of CUDA code with ~10 crash points in `__init__`. OpenVINO needs only `self.requests`, `self.input_batch`, and a `_update_states` pattern.

**File**: `vllm_openvino/worker_v1/openvino_model_runner_v1.py`

**Actions**:
1. **Remove GPUModelRunner import and inheritance**:
   ```python
   # BEFORE:
   from vllm.v1.worker.gpu_model_runner import GPUModelRunner
   class OpenVINOModelRunnerV1(GPUModelRunner):
   # AFTER:
   class OpenVINOModelRunnerV1:
   ```

2. **Rewrite `__init__`** to set up required state WITHOUT calling GPU super().__init__:
   ```python
   def __init__(self, vllm_config: VllmConfig, device: torch.device,
                ov_core: ov.Core = None, kv_cache_dtype: Optional[str] = "auto"):
       self.vllm_config = vllm_config
       self.model_config = vllm_config.model_config
       self.cache_config = vllm_config.cache_config
       self.scheduler_config = vllm_config.scheduler_config
       self.parallel_config = vllm_config.parallel_config
       self.compilation_config = vllm_config.compilation_config
       self.device = device
       self.ov_core = ov_core or ov.Core()
       self.kv_cache_dtype = kv_cache_dtype
       self.model: nn.Module  # Set after load_model()
       
       # V1 state management
       self.requests: dict[str, CachedRequestState] = {}
       self.input_batch = InputBatch(
           max_num_reqs=self.scheduler_config.max_num_seqs,
           max_model_len=self.model_config.max_model_len,
           max_num_blocks_per_req=cdiv(self.model_config.max_model_len,
                                        self.cache_config.block_size),
           device=self.device,
           pin_memory=False,  # OpenVINO/CPU — no pin memory
           vocab_size=self.model_config.get_vocab_size(),
       )
       
       # KV cache — set by worker after initialize_cache
       self.kv_caches: list = []
       self.block_size: int = 0
   ```
   **NOTE**: The `InputBatch` constructor signature must be verified against the actual v0.13.0 code. Use `python3 -c "import inspect; from vllm.v1.worker.gpu_input_batch import InputBatch; print(inspect.signature(InputBatch.__init__))"` inside the container to get the exact signature.

3. **Implement `_update_states(scheduler_output: SchedulerOutput)`**:
   This is the critical method that populates `self.requests` and `self.input_batch` from scheduler output. Minimal implementation:
   ```python
   def _update_states(self, scheduler_output: SchedulerOutput) -> None:
       # Add new requests
       for new_req in scheduler_output.new_reqs:
           self.requests[new_req.req_id] = CachedRequestState(
               req_id=new_req.req_id,
               prompt_token_ids=new_req.prompt_token_ids,
               mm_inputs=[],
               mm_positions=[],
               sampling_params=new_req.sampling_params,
               block_ids=new_req.block_ids,
               num_computed_tokens=new_req.num_computed_tokens,
               output_token_ids=[],
           )
           self.input_batch.add_request(self.requests[new_req.req_id])
       
       # Remove finished requests
       for req_id in scheduler_output.finished_req_ids:
           if req_id in self.requests:
               self.input_batch.remove_request(req_id)
               del self.requests[req_id]
       
       # Update computed tokens
       for req_id, num_tokens in scheduler_output.num_scheduled_tokens.items():
           if req_id in self.requests:
               req = self.requests[req_id]
               req.num_computed_tokens += num_tokens
       
       # Update block tables
       if scheduler_output.scheduled_new_blocks:
           for req_id, block_ids in scheduler_output.scheduled_new_blocks.items():
               if req_id in self.requests:
                   self.requests[req_id].block_ids.extend(block_ids)
   ```
   **CRITICAL NOTE**: This is a simplified version. The exact `CachedRequestState` constructor and `InputBatch.add_request` signatures MUST be verified against the actual v0.13.0 installed package. Use these commands inside the container:
   ```bash
   python3 -c "import inspect; from vllm.v1.worker.gpu_input_batch import CachedRequestState; print(inspect.signature(CachedRequestState))"
   python3 -c "import inspect; from vllm.v1.worker.gpu_input_batch import InputBatch; print([m for m in dir(InputBatch) if not m.startswith('_')])"
   ```

4. **Remove CUDA-specific imports** that are no longer needed:
   - `from vllm.v1.worker.gpu_model_runner import GPUModelRunner` → remove
   - `from vllm.sequence import SequenceGroupMetadata` → remove
   - `from vllm.sequence import ExecuteModelRequest` → remove (will be re-addressed in Task 4)
   - Keep: `from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch`
   - Keep: `from vllm.v1.outputs import ModelRunnerOutput`
   - Keep: `from vllm.v1.core.sched.output import SchedulerOutput`

5. **Remove unused imports**:
   - `from vllm_openvino.model import ModelInput` → remove (V0 concept)
   - `from vllm.multimodal import MultiModalKwargs` (duplicate on line 22) → remove duplicate

**Acceptance Criteria**:
- `grep -c 'GPUModelRunner' vllm_openvino/worker_v1/openvino_model_runner_v1.py` → `0`
- `grep -c 'SequenceGroupMetadata' vllm_openvino/worker_v1/openvino_model_runner_v1.py` → `0`
- `python3 -m py_compile vllm_openvino/worker_v1/openvino_model_runner_v1.py` → success

**QA**:
- No CUDA imports remain
- `__init__` does not call `super().__init__()` with GPUModelRunner
- `self.requests` and `self.input_batch` are properly initialized

## Wave 3 — Protocol Fix (Sequential, depends on Wave 2)

### Task 4: Fix `execute_model` Signatures and Data Flow
**Objective**: Align worker and model runner `execute_model` signatures with V1 protocol. The V1 engine calls `worker.execute_model(scheduler_output: SchedulerOutput)`, NOT `execute_model(execute_model_req: ExecuteModelRequest)`.

**Files**:
- `vllm_openvino/worker_v1/openvino_worker_v1.py`
- `vllm_openvino/worker_v1/openvino_model_runner_v1.py`

**Actions for Worker** (`openvino_worker_v1.py`):

1. **Change `execute_model` signature** (line 182-195):
   ```python
   # BEFORE:
   def execute_model(self, execute_model_req: Optional[ExecuteModelRequest] = None) -> ModelRunnerOutput:
       if execute_model_req.total_num_scheduled_tokens == 0:
           ...
       return self.model_runner.execute_model(execute_model_req, execute_model_req)
   
   # AFTER:
   def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
       if scheduler_output.total_num_scheduled_tokens == 0:
           return ModelRunnerOutput(
               req_ids=[], req_id_to_index={}, sampled_token_ids=[],
               spec_token_ids=None, logprobs=None, prompt_logprobs_dict={},
           )
       return self.model_runner.execute_model(scheduler_output)
   ```

2. **Remove V0 imports**:
   - `from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata` → remove entirely

**Actions for Model Runner** (`openvino_model_runner_v1.py`):

1. **Change `execute_model` signature** (line 148-213):
   ```python
   # BEFORE:
   def execute_model(self, execute_model_req: ExecuteModelRequest,
                     scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
   
   # AFTER:
   @torch.inference_mode()
   def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
       # First update internal state from scheduler
       self._update_states(scheduler_output)
       
       # Then prepare inputs and run model
       ...
   ```

2. **Rewrite `_prepare_inputs` signature** (line 58-145):
   ```python
   # BEFORE:
   def _prepare_inputs(self, execute_model_req, scheduler_output,
                       seq_group_metadata_list, multimodal_kwargs) -> Tuple[...]:
   
   # AFTER:
   def _prepare_inputs(self, scheduler_output: SchedulerOutput) -> Tuple[
       torch.Tensor, torch.Tensor, AttentionMetadata, SamplingMetadata, BatchedTensorInputs]:
   ```
   The internal logic already reads from `self.requests` and `self.input_batch.req_ids` (lines 90-91), which is correct for V1 pattern AFTER `_update_states` is called. The key changes:
   - Remove `execute_model_req` parameter
   - Remove `seq_group_metadata_list` parameter (V0 concept)
   - Remove `multimodal_kwargs` parameter (extract from `self.requests` if needed)
   - The function body mostly stays the same since it already iterates `self.input_batch.req_ids`

3. **Update `execute_model` body** to use new `_prepare_inputs`:
   ```python
   (input_tokens, input_positions, attn_metadata, sampling_metadata, multi_modal_kwargs,
   ) = self._prepare_inputs(scheduler_output)
   ```
   Remove references to `execute_model_req.seq_group_metadata_list` and `execute_model_req.multimodal_kwargs`.

4. **Remove all V0 imports**:
   - `from vllm.sequence import ExecuteModelRequest` → remove
   - `from vllm.sequence import SequenceGroupMetadata` → remove
   - `from vllm.multimodal import MultiModalKwargs` (line 22, duplicate) → remove

**Acceptance Criteria**:
- `grep -r 'ExecuteModelRequest' vllm_openvino/ --include='*.py' | wc -l` → `0`
- `grep -r 'SequenceGroupMetadata' vllm_openvino/worker_v1/ --include='*.py' | wc -l` → `0`
- `python3 -c "import inspect; from vllm_openvino.worker_v1.openvino_worker_v1 import OpenVINOWorkerV1; sig = inspect.signature(OpenVINOWorkerV1.execute_model); assert 'scheduler_output' in sig.parameters; print('OK')"` → `OK`

**QA**:
- Worker.execute_model takes `scheduler_output: SchedulerOutput`
- ModelRunner.execute_model takes `scheduler_output: SchedulerOutput`
- `_update_states` is called before `_prepare_inputs`
- No V0-only types (`ExecuteModelRequest`, `SequenceGroupMetadata`) referenced

## Wave 4 — Cleanup (Parallel, depends on Wave 3)

### Task 5: Fix `OpenVINOAttentionMetadataBuilder.build()`
**Objective**: Completely rewrite the `build()` method which currently passes 7 non-existent fields and omits 5 required fields, plus references an undefined `device` variable.

**File**: `vllm_openvino/attention/backends/openvino.py`

**Problem Analysis**:
The `OpenVINOAttentionMetadata` dataclass has these fields:
- `past_lens`, `subsequence_begins`, `block_indices`, `block_indices_begins`, `max_context_len`, `multi_modal_placeholder_index_maps`, `enable_kv_scales_calculation`, `sampled_token_indices`

But `build()` (lines 171-190) passes completely wrong fields:
- ❌ `num_actual_tokens`, `max_query_len`, `query_start_loc`, `max_seq_len`, `seq_lens`, `block_table`, `slot_mapping` — NONE of these exist on `OpenVINOAttentionMetadata`
- ❌ Uses undefined `device` variable (line 181)

**Actions**:
1. **First, determine if `build()` is actually called in the OpenVINO V1 flow**:
   - In the current code, `_prepare_inputs` in the model runner builds `OpenVINOAttentionMetadata` directly (lines 128-137 of openvino_model_runner_v1.py)
   - If the V1 engine also calls the builder via the attention layer, both paths must produce valid metadata
   - Most likely: the builder is called by V1's attention infrastructure, while `_prepare_inputs` builds the same metadata for the forward context

2. **Rewrite `build()` to produce valid `OpenVINOAttentionMetadata`**:
   ```python
   def build(self, common_prefix_len: int,
             common_attn_metadata: CommonAttentionMetadata,
             fast_build: bool = False) -> OpenVINOAttentionMetadata:
       # Extract data from common_attn_metadata
       # CommonAttentionMetadata provides: seq_lens, query_start_loc, etc.
       # Map these to OpenVINO's expected format
       
       seq_lens = common_attn_metadata.seq_lens
       query_start_loc = common_attn_metadata.query_start_loc
       
       # Build OpenVINO-specific tensors
       past_lens = ov.Tensor(...)  # derived from seq_lens - query_lens
       subsequence_begins = ov.Tensor(...)  # derived from query_start_loc
       block_indices = ov.Tensor(...)  # from block tables
       block_indices_begins = ov.Tensor(...)
       max_context_len = ov.Tensor(...)
       
       return OpenVINOAttentionMetadata(
           past_lens=past_lens,
           subsequence_begins=subsequence_begins,
           block_indices=block_indices,
           block_indices_begins=block_indices_begins,
           max_context_len=max_context_len,
           multi_modal_placeholder_index_maps=None,
           enable_kv_scales_calculation=False,
           sampled_token_indices=sampled_token_indices,
       )
   ```
   **NOTE**: The exact mapping from `CommonAttentionMetadata` fields to OpenVINO metadata tensors requires inspecting `CommonAttentionMetadata`'s actual fields. Run inside container:
   ```bash
   python3 -c "import inspect; from vllm.v1.attention.backends.utils import CommonAttentionMetadata; print(inspect.signature(CommonAttentionMetadata))"
   ```

3. **Fix `__init__` of the builder** to store `self.device`:
   ```python
   def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
       super().__init__(kv_cache_spec, layer_names, vllm_config, device)
       self._device = device  # Store for use in build()
   ```

4. **Alternative (if builder is never actually called)**: Make `build()` a safe stub that raises `NotImplementedError("OpenVINO builds attention metadata in _prepare_inputs")` — this is simpler but only valid if confirmed the builder path is not used.

**Acceptance Criteria**:
- `grep -c 'num_actual_tokens\|query_start_loc\|slot_mapping\|block_table' vllm_openvino/attention/backends/openvino.py` → `0`
- `python3 -m py_compile vllm_openvino/attention/backends/openvino.py` → success
- `build()` method returns valid `OpenVINOAttentionMetadata` or raises clear `NotImplementedError`

**QA**:
- No undefined variable references
- All fields passed to `OpenVINOAttentionMetadata()` exist on the dataclass
- File compiles without syntax/type errors

---

### Task 6: Clean Up Containerfile, README, and Dead Imports
**Objective**: Remove all remaining `VLLM_USE_V1` references from non-Python files, delete backup files, and clean up remaining dead imports across all files.

**Files**:
- `Containerfile`
- `README.md`
- `vllm_openvino/platform.py.bak`
- Various `.py` files with stale imports

**Actions**:

1. **Containerfile** (lines 12, 23):
   - Line 12: Remove `VLLM_USE_V1=1` from the `RUN` command:
     ```
     # BEFORE:
     RUN VLLM_TARGET_DEVICE="empty" VLLM_USE_V1=1 PIP_EXTRA_INDEX_URL=...
     # AFTER:
     RUN VLLM_TARGET_DEVICE="empty" PIP_EXTRA_INDEX_URL=...
     ```
   - Line 23: Remove `VLLM_USE_V1=1` from the `ENV` line:
     ```
     # BEFORE:
     ENV ... VLLM_USE_V1=1
     # AFTER:
     ENV ... (without VLLM_USE_V1=1)
     ```

2. **README.md** (line 82):
   - Remove the bullet point: `- \`VLLM_USE_V1\` to enable V1 vLLM API, e.g, \`VLLM_USE_V1=1\``
   - Optionally add a note: "V1 engine is used by default (the only supported engine in vLLM v0.13.0+)."

3. **Delete backup file**:
   - `rm vllm_openvino/platform.py.bak` — stale backup with V0 logic

4. **Clean remaining dead imports** (verify each is truly unused first):
   - `openvino_model_runner_v1.py`: Remove `from vllm_openvino.model import ModelInput` if unused after Task 3/4
   - `openvino_model_runner_v1.py`: Remove duplicate `from vllm.multimodal import MultiModalKwargs` (line 22)
   - `openvino_worker_v1.py`: Remove `from vllm.sequence import ExecuteModelRequest, SequenceGroupMetadata` if not already done in Task 4
   - `openvino_worker_v1.py`: Remove any other V0-only imports

5. **Verify no `VLLM_USE_V1` references remain anywhere**:
   ```bash
   grep -r 'VLLM_USE_V1' . --include='*.py' --include='*.md' --include='Containerfile' --include='Dockerfile'
   ```
   Expected: only matches in `assessment_v0_v1_inventory.md` (historical doc) and `.sisyphus/` files.

**Acceptance Criteria**:
- `grep -c 'VLLM_USE_V1' Containerfile` → `0`
- `grep -c 'VLLM_USE_V1' README.md` → `0`
- `test ! -f vllm_openvino/platform.py.bak` → success
- All `.py` files in `vllm_openvino/` compile with `py_compile`

**QA**:
- No user-facing documentation references dead env var
- No backup files with stale logic remain
- Container builds without warnings about unknown env vars

## Final Verification Wave

After all tasks complete, run these verification commands:

```bash
# 1. No VLLM_USE_V1 references in Python code
grep -r 'VLLM_USE_V1' vllm_openvino/ --include='*.py' | wc -l
# Expected: 0

# 2. No ExecuteModelRequest references
grep -r 'ExecuteModelRequest' vllm_openvino/ --include='*.py' | wc -l
# Expected: 0

# 3. No dead V0 worker class
grep -c 'class OpenVINOWorker' vllm_openvino/kv_cache.py
# Expected: 0

# 4. No GPUModelRunner inheritance (if dropped)
grep -c 'GPUModelRunner' vllm_openvino/worker_v1/openvino_model_runner_v1.py
# Expected: 0

# 5. Import smoke test
python3 -c "from vllm_openvino.worker_v1.openvino_worker_v1 import OpenVINOWorkerV1; print('Worker OK')"
python3 -c "from vllm_openvino.kv_cache import OpenVINOCacheEngine; print('Cache OK')"
python3 -c "from vllm_openvino.model_executor.model_loader.openvino import get_model; print('Loader OK')"

# 6. Signature conformance
python3 -c "import inspect; from vllm_openvino.worker_v1.openvino_worker_v1 import OpenVINOWorkerV1; sig = inspect.signature(OpenVINOWorkerV1.execute_model); assert 'scheduler_output' in sig.parameters; print('Signature OK')"

# 7. No VLLM_USE_V1 in Containerfile/README
grep -c 'VLLM_USE_V1' Containerfile README.md
# Expected: 0 for both
```

---

## Wave 5 — Integration Validation (현재 진행 중)

### Task 7: Apply All Fixes via Hot Patching
**Objective**: 로컬 코드 수정사항을 컨테이너 이미지에 bake-in하여 스테일 캐시 문제 없이 반영.

- [ ] Step 1: 임시 sleep 컨테이너 실행
  ```bash
  podman run -d --name vllm-patch-tmp --entrypoint sleep vllm-0.13.0-ov-patched 3600
  ```
- [ ] Step 2: 수정된 코드 복사
  ```bash
  podman cp /home/user/project/vllm-openvino/vllm_openvino vllm-patch-tmp:/opt/app-root/
  ```
- [ ] Step 3: 새 이미지로 커밋
  ```bash
  podman commit \
    --change='ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]' \
    vllm-patch-tmp vllm-0.13.0-ov-patched
  ```
- [ ] Step 4: 임시 컨테이너 삭제 후 새 이미지로 서버 실행
  ```bash
  podman rm -f vllm-patch-tmp
  podman stop vllm-server 2>/dev/null; podman rm vllm-server 2>/dev/null
  podman run --replace -d --name vllm-server \
    -p 8000:8000 \
    -v ~/hf:/models:Z \
    -e VLLM_OPENVINO_DEVICE=CPU \
    -e VLLM_OPENVINO_KVCACHE_SPACE=8 \
    -e TORCH_COMPILE_DISABLE=1 \
    vllm-0.13.0-ov-patched \
    --model /models/Qwen2.5-Coder-3B-Instruct-int4-ov \
    --dtype auto --max-model-len 2048 --host 0.0.0.0 --port 8000
  ```
- [ ] Step 5: 서버 기동 대기 및 확인
  ```bash
  sleep 60 && podman logs vllm-server | tail -20
  # "Application startup complete." 확인
  ```

**Acceptance Criteria**: 로그에 `Application startup complete.` 포함.

---

### Task 8: Inference Smoke Test
**Objective**: curl로 chat completion 요청이 정상 응답을 반환하는지 확인.

- [ ] Inference 테스트
  ```bash
  curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "/models/Qwen2.5-Coder-3B-Instruct-int4-ov", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 32}'
  ```
- [ ] 실패 시: `podman logs vllm-server | grep -A 20 "ERROR"` 로 에러 확인

**Acceptance Criteria**: HTTP 200, 응답에 `"choices"` 배열 포함, 생성된 텍스트 있음.

---

### Task 9: Metrics Check
**Objective**: `/metrics` 엔드포인트 정상 동작 확인.

- [ ] Metrics 테스트
  ```bash
  curl http://localhost:8000/metrics | head -40
  ```

**Acceptance Criteria**: `vllm_` prefix 메트릭 5개 이상 존재.

---

### Task 10: Git Commit
**Objective**: 모든 수정사항을 git commit.

- [ ] 변경사항 확인
  ```bash
  git diff --stat
  ```
- [ ] git add + commit
  ```bash
  git add -A && git commit -m "fix: migrate OpenVINO backend to vLLM v0.13.0 V1 protocol

- Fix bind_kv_cache() 3-argument signature
- Add get_supported_tasks() returning ('generate',)
- Remove MultiModalKwargs.as_kwargs() (deleted in v0.13.0)
- Add refresh_metadata() to fix sampling metadata race
- Disable torch.compile via CompilationMode.NONE + TORCH_COMPILE_DISABLE=1
- Fix ModelRunnerOutput: remove spec_token_ids, add pooler_output=None"
  ```

**Acceptance Criteria**: `git log --oneline -1` 에 커밋 메시지 보임.

---

## Dependency Graph

```
Wave 1 (Parallel):
├── Task 1: VLLM_USE_V1 removal (model loader)
└── Task 2: Dead V0 class removal (kv_cache.py)

Wave 2 (After Wave 1):
└── Task 3: Drop GPUModelRunner inheritance (model runner refactor)

Wave 3 (After Wave 2):
└── Task 4: Fix execute_model signatures + data flow

Wave 4 (After Wave 3, Parallel):
├── Task 5: Fix AttentionMetadataBuilder.build()
└── Task 6: Containerfile/README/import cleanup

Critical Path: Task 1/2 → Task 3 → Task 4 → Task 5
```

---

## Risks and Mitigations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Standalone model runner missing V1 state management | Critical | Replicate minimal `_update_states` from upstream pattern |
| `InputBatch` constructor changes between vLLM versions | Medium | Read actual `InputBatch.__init__` signature from installed package |
| `kv_cache_dtype_str_to_dtype` fails for "dynamic" or "u8" | Medium | Use OpenVINO's own dtype mapping instead |
| OpenVINO attention metadata not called via builder in practice | Low | Verify flow; if metadata is built in `_prepare_inputs`, builder can remain a safe stub |

