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

## 4. 검증 방법

- `python3 -m py_compile vllm_openvino/**/*.py`를 실행하여 문법 오류를 확인합니다.
- Podman 테스트 환경에서 실제 추론을 실행하여 기능적 회귀가 없는지 검증합니다.
