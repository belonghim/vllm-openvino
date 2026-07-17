# vLLM Version Upgrade Checklist

vLLM 버전 업그레이드 시 다음 사항을 확인하세요.

## 1. 업그레이드 전 준비

- vLLM 릴리즈 노트를 확인하여 변경 사항을 파악합니다.
- 특히 `vllm.v1.*` 경로의 내부 API 변경 여부를 주의 깊게 살펴봅니다.

## 2. 🔴 높은 위험 import 확인

다음 `vllm.v1.*` 모듈들은 vLLM 버전 업그레이드 시 가장 먼저 깨질 수 있습니다.

| 모듈 경로 | 심볼 |
|-----------|------|
| `vllm.v1.kv_cache_interface` | `KVCacheSpec`, `KVCacheConfig`, `FullAttentionSpec`, `AttentionSpec` |
| `vllm.v1.attention.backend` | `AttentionBackend`, `AttentionMetadata` |
| `vllm.v1.attention.backends.utils` | `CommonAttentionMetadata` |
| `vllm.v1.attention.backend` | `AttentionMetadataBuilder` (v0.18.1에서 `backends.utils` → `backend`로 이동) |
| `vllm.v1.outputs` | `SamplerOutput`, `ModelRunnerOutput` |
| `vllm.v1.sample.metadata` | `SamplingMetadata` |
| `vllm.v1.sample.sampler` | `Sampler` |
| `vllm.v1.worker.gpu_input_batch` | `CachedRequestState`, `InputBatch` |
| `vllm.v1.worker.worker_base` | `WorkerBase` |
| `vllm.v1.worker.utils` | `bind_kv_cache` |
| `vllm.v1.core.sched.output` | `SchedulerOutput`, `NewRequestData`, `CachedRequestData` |

---

## 3. 파일별 import 목록 (위험도)

이 플러그인은 vLLM의 **내부 API**에 깊이 의존합니다.

**위험도 구분**: 🔴 높음 = v1 내부 API, 🟡 중간 = config/public API, 🟢 낮음 = logger/util

| 파일 | Import | 위험도 |
|------|--------|--------|
| `__init__.py` | `vllm.logger → DEFAULT_LOGGING_CONFIG` | 🟢 낮음 |
| `platform.py` | `vllm.platforms.interface → Platform, PlatformEnum` | 🟡 중간 |
| `platform.py` | `vllm.config → VllmConfig, CompilationMode` | 🟡 중간 |
| `platform.py` | `vllm.logger → init_logger` | 🟢 낮음 |
| `kv_cache.py` | `vllm.config → CacheConfig, DeviceConfig, ModelConfig, ParallelConfig` | 🟡 중간 |
| `kv_cache.py` | `vllm.logger → init_logger` | 🟢 낮음 |
| `kv_cache.py` | `vllm.platforms → current_platform` | 🟡 중간 |
| `kv_cache.py` | `vllm.v1.kv_cache_interface → KVCacheSpec` | 🔴 높음 |
| `utils.py` | `vllm.logger → init_logger` | 🟢 낮음 |
| `attention/backends/openvino.py` | `vllm.v1.attention.backend → AttentionBackend` | 🔴 높음 |
| `attention/backends/openvino.py` | `vllm.v1.attention.backends.utils → CommonAttentionMetadata` | 🟡 중간 |
| `attention/backends/openvino.py` | `vllm.v1.attention.backend → AttentionMetadataBuilder` | 🔴 높음 |
| `attention/backends/openvino.py` | `vllm.v1.kv_cache_interface → AttentionSpec` | 🔴 높음 |
| `attention/backends/openvino.py` | `vllm.config → VllmConfig` | 🟡 중간 |
| `model_executor/model_loader/openvino.py` | `vllm.config → ModelConfig, VllmConfig, set_current_vllm_config` | 🟡 중간 |
| `model_executor/model_loader/openvino.py` | `vllm.forward_context → get_forward_context` | 🟡 중간 |
| `model_executor/model_loader/openvino.py` | `vllm.logger → init_logger` | 🟢 낮음 |
| `model_executor/model_loader/openvino.py` | `vllm.model_executor.layers.logits_processor → LogitsProcessor` | 🟡 중간 |
| `model_executor/model_loader/openvino.py` | `vllm.v1.outputs → SamplerOutput` | 🔴 높음 |
| `model_executor/model_loader/openvino.py` | `vllm.v1.sample.metadata → SamplingMetadata` | 🔴 높음 |
| `model_executor/model_loader/openvino.py` | `vllm.v1.sample.sampler → Sampler` | 🔴 높음 |
| `worker_v1/openvino_model_runner_v1.py` | `vllm.config → VllmConfig` | 🟡 중간 |
| `worker_v1/openvino_model_runner_v1.py` | `vllm.forward_context → set_forward_context` | 🟡 중간 |
| `worker_v1/openvino_model_runner_v1.py` | `vllm.logger → init_logger` | 🟢 낮음 |
| `worker_v1/openvino_model_runner_v1.py` | `vllm.multimodal → BatchedTensorInputs` | 🟡 중간 |
| `worker_v1/openvino_model_runner_v1.py` | `vllm.v1.outputs → ModelRunnerOutput` | 🔴 높음 |
| `worker_v1/openvino_model_runner_v1.py` | `vllm.v1.sample.metadata → SamplingMetadata` | 🔴 높음 |
| `worker_v1/openvino_model_runner_v1.py` | `vllm.v1.worker.gpu_input_batch → CachedRequestState, InputBatch` | 🔴 높음 |
| `worker_v1/openvino_model_runner_v1.py` | `vllm.v1.attention.backend → AttentionMetadata` | 🔴 높음 |
| `worker_v1/openvino_model_runner_v1.py` | `vllm.v1.core.sched.output → SchedulerOutput` | 🔴 높음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.config → CacheConfig, VllmConfig` | 🟡 중간 |
| `worker_v1/openvino_worker_v1.py` | `vllm.distributed → ensure_model_parallel_initialized` | 🟡 중간 |
| `worker_v1/openvino_worker_v1.py` | `vllm.logger → init_logger` | 🟢 낮음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.lora.request → LoRARequest` | 🟡 중간 |
| `worker_v1/openvino_worker_v1.py` | `vllm.utils.torch_utils → set_random_seed` | 🟢 낮음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.multimodal → MULTIMODAL_REGISTRY` | 🟡 중간 |
| `worker_v1/openvino_worker_v1.py` | `vllm.platforms → current_platform` | 🟡 중간 |
| `worker_v1/openvino_worker_v1.py` | `vllm.sampling_params → SamplingParams` | 🟡 중간 |
| `worker_v1/openvino_worker_v1.py` | `vllm.v1.worker.utils → bind_kv_cache` | 🔴 높음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.v1.kv_cache_interface → MambaSpec` | 🔴 높음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.v1.outputs → ModelRunnerOutput` | 🔴 높음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.v1.worker.worker_base → WorkerBase` | 🔴 높음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.v1.core.sched.output → SchedulerOutput, NewRequestData, CachedRequestData` | 🔴 높음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.v1.worker.gpu_input_batch → InputBatch` | 🔴 높음 |

---

## 4. 🔴 알려진 시그니처 변경 (업그레이드 시 코드 수정 필요)

특정 vLLM 버전 이후 메서드 시그니처가 바뀌어 **현재 구현이 런타임에 깨질 수 있는** 항목입니다.

| 메서드 | 0.19.1 | 0.20.0+ | 영향 |
|--------|--------|---------|------|
| `WorkerBase.compile_or_warm_up_model()` | `-> None` (caller가 반환값 무시) | `-> CompilationTimes(language_model, encoder)` NamedTuple | 0.20+에서 caller가 `.language_model` 접근. 현재 `return 0.0` → `AttributeError`. 수정 필요: `return CompilationTimes(language_model=0.0, encoder=0.0)` |

업그레이드 시점에 위 표를 우선 처리한 뒤 다른 import 변경을 진행하세요.

---

## 5. 검증 방법

- `python3 -m py_compile vllm_openvino/**/*.py`를 실행하여 문법 오류를 확인합니다.
- Podman 테스트 환경에서 실제 추론을 실행하여 기능적 회귀가 없는지 검증합니다.

---

## 6. v0.24.0 → v0.25.0 분석 (2026-07-17)

### 배경

vLLM v0.25.0 (2026-07-11)는 558 commits, 232 contributors 포함. v0.25.1 (2026-07-14)는 TorchCodec import 지연, mixed-dtype allreduce RMSNorm guard 패치.

릴리즈 주요 테마: MRv2 기본 활성화 (dense 모델), PagedAttention 삭제, Platform memory API 마이그레이션, device selection 리팩토링.

### 소스 레벨 diff 결과

모든 plugin-facing v1 내부 API는 v0.24.0과 **정확히 동일**:
- `WorkerBase.__init__()` — 5개 파라미터 (vllm_config, local_rank, rank, distributed_init_method, is_driver_worker)
- `WorkerBase.init_device()`, `execute_model()`, `sample_tokens()`, `compile_or_warm_up_model()` — 시그니처 불변
- `ModelRunnerOutput` dataclass — 모든 필드 불변 (req_ids, req_id_to_index, sampled_token_ids, logprobs, etc.)
- `SchedulerOutput`, `NewRequestData`, `CachedRequestData` dataclass — 모든 필드 불변
- `FullAttentionSpec`, `AttentionSpec`, `KVCacheSpec` — 하위 호환: `KVQuantMode.INT4_PER_TOKEN_HEAD`, `RSWASpec` 추가만 있음
- `AttentionBackend` ABC — 새 추상 메서드 없음. `rswa_prefix_lens`, `lse_base_on_e`, `token_to_req_indices()` 추가 (모두 선택적)

### 잠재적 리스크 (v0.25.0 현재는 문제 아님)

| 항목 | 상태 | 설명 |
|------|------|------|
| `execute_model()` → `None` 반환 가능 | 📌 모니터링 | v0.25.0에도 return type에 `None` 포함. 플러그인은 `None` 반환 안 함. **향후 vLLM이 이 패턴 강제 시 호환성 이슈 가능** |
| `sample_tokens()` 신규 추상 메서드화 | 📌 모니터링 | 현재는 `raise NotImplementedError` 기본값. **향후 추상 메서드로 변경 시 플러그인에 구현 추가 필요** |
| MRv2 기본 활성화 (#44443) | 🟢 영향 없음 | 플러그인은 V1 ModelRunner 자체 구현 (`OpenVINOModelRunnerV1`) 사용 |
| PagedAttention 삭제 (#47361) | 🟢 영향 없음 | 플러그인은 V0 PagedAttention 미사용. V1 `AttentionBackend` 인터페이스 사용 |
| `mem_get_info` 제거 (#44825) | 🟢 영향 없음 | 플러그인은 `is_openvino_cpu()` 커스텀 메서드만 사용, `mem_get_info()` 미호출 |
| `CUDA_VISIBLE_DEVICES` 중단 (#45026) | 🟢 영향 없음 | `WorkerBase.__init__()` 시그니처 불변. 변경사항은 `WorkerWrapperBase.init_worker()`에서 `kwargs.pop()` 처리 |

### 업그레이드 권장사항

1. v0.25.0으로 업그레이드해도 **plugin 코드 수정 불필요**
2. `docs/upgrade-checklist.md`의 import 목록(섹션 2-3) 여전히 유효
3. 다음 v0.26.0 릴리즈에서 `execute_model`/`sample_tokens` 분리 강제 여부 확인 필요
4. vLLM issue [#41286](https://github.com/vllm-project/vllm/issues/41286) (Model Runner V2 migration) 지속 모니터링

### 검증 완료 파일 (permalink 기반)

각 인터페이스의 v0.24.0과 v0.25.0 태그를 직접 비교하여 diff 확인:
- `vllm/v1/worker/worker_base.py` — diff 없음
- `vllm/v1/outputs.py` — diff 없음
- `vllm/v1/core/sched/output.py` — diff 없음
- `vllm/v1/kv_cache_interface.py` — INT4_PER_TOKEN_HEAD, RSWASpec 추가 (하위 호환)
- `vllm/v1/attention/backend.py` — rswa_prefix_lens, lse_base_on_e, token_to_req_indices 추가 (하위 호환)
