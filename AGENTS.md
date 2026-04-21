# AGENTS.md — vllm-openvino

> AI 에이전트가 이 프로젝트에서 작업할 때 반드시 읽어야 하는 컨텍스트 문서.

## 프로젝트 정체성

**vllm-openvino**는 [vLLM](https://github.com/vllm-project/vllm)의 **플러그인**으로, Intel OpenVINO를 LLM 추론 백엔드로 추가합니다.

- **vLLM 버전**: 0.13.0+ (V1 엔진 전용, 0.14.1 컨테이너 이미지 테스트 완료)
- **OpenVINO 버전**: >= 2026.0.0
- **플러그인 등록**: `pyproject.toml`의 `[project.entry-points."vllm.platform_plugins"]`
- **단일 개발자 프로젝트** (belonghim)

### 프로젝트 존재 이유

vLLM은 GPU(CUDA/ROCm) 중심의 LLM 서빙 프레임워크다. vllm-openvino는 Intel CPU와 GPU에서도 동일한 OpenAI-compatible API 서버를 운영할 수 있도록, OpenVINO를 vLLM의 추론 백엔드로 추가하는 플러그인이다. vLLM의 스케줄러, PagedAttention, API 서버 인프라를 그대로 활용하면서, 추론 실행 계층만 OpenVINO로 교체한다.

### 설계 원칙

1. **upstream 패턴 추종** — vLLM 내부 구현 패턴을 정확히 따른다. 더 나은 설계가 보여도, upstream과 다른 아키텍처는 버전 업그레이드 시 호환성을 깨뜨린다.
2. **플러그인 경계 준수** — vLLM core를 수정하거나, core의 내부 동작을 가정하지 않는다. vLLM이 제공하는 플러그인 인터페이스만 사용한다.
3. **규모에 맞는 최소주의** — 단일 개발자, 14파일 프로젝트다. 현재 규모를 초과하는 복잡도(테스트 인프라, CI/CD, 추상화 레이어)는 유지 비용이 이점을 넘는다.
4. **현재 문제만 해결** — 실제 사용자 수요가 없거나 vLLM 아키텍처 제약으로 구현 불가능한 기능은 추가하지 않는다.

### 의사결정 체크리스트

새 기능/변경을 수용할지 판단할 때 다음 순서로 평가한다:

1. upstream vLLM에 동일한 패턴이 있는가? → 있으면 그대로 따른다
2. 플러그인 경계를 넘어 vLLM core 수정이 필요한가? → 필요하면 거부한다
3. 현재 존재하는 문제를 해결하는가, 가상의 미래 문제인가? → 가상이면 보류한다
4. 사용자/실제 수요가 확인되었는가? → 없으면 보류한다
5. 복잡도가 현재 프로젝트 규모에 비례하는가? → 과도하면 거부한다

**원칙 충돌 시 우선순위**: upstream 호환성 > 플러그인 경계 > 최소주의

## 디렉토리 구조

```
vllm_openvino/
├── __init__.py                          # 플러그인 등록 (register 함수)
├── platform.py                          # OpenVinoPlatform — vLLM Platform 인터페이스 구현
├── envs.py                              # 환경변수 정의 (VLLM_OPENVINO_*)
├── kv_cache.py                          # OpenVINOCacheEngine — KV 캐시 관리
├── utils.py                             # GPU 메모리 계산 유틸리티
├── attention/backends/openvino.py       # OpenVINO Attention 백엔드
├── model_executor/model_loader/openvino.py  # 모델 로딩 + PagedAttention 변환
└── worker_v1/
    ├── openvino_worker_v1.py            # OpenVINOWorkerV1 — 워커 (KV 캐시 할당, 프로파일링)
    └── openvino_model_runner_v1.py      # ModelRunner — 입력 준비 + 추론 실행
```

## 핵심 의존 관계

이 플러그인은 vLLM의 **내부 API**에 깊이 의존합니다.

**위험도 구분**: 🔴 높음 = v1 내부 API (버전 업그레이드 시 가장 취약), 🟡 중간 = config/public API, 🟢 낮음 = logger/util

### 파일별 import 목록

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
| `worker_v1/openvino_worker_v1.py` | `vllm.v1.kv_cache_interface → KVCacheSpec, KVCacheConfig, FullAttentionSpec` | 🔴 높음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.v1.outputs → ModelRunnerOutput` | 🔴 높음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.v1.worker.worker_base → WorkerBase` | 🔴 높음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.v1.core.sched.output → SchedulerOutput, NewRequestData, CachedRequestData` | 🔴 높음 |
| `worker_v1/openvino_worker_v1.py` | `vllm.v1.worker.gpu_input_batch → InputBatch` | 🔴 높음 |

> ⚠️ 🔴 **높음 위험** import (`vllm.v1.*`)는 vLLM 버전 업그레이드 시 가장 먼저 깨질 수 있습니다. 업그레이드 전에 반드시 각 import의 현재 위치를 확인하세요. 🟡 중간 위험은 config/public API이고, 🟢 낮음은 logger/util로 상대적으로 안정적입니다.

## 빌드 & 실행

```bash
# 설치
VLLM_TARGET_DEVICE="empty" PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" pip install .

# 실행 (CPU)
VLLM_OPENVINO_DEVICE=CPU TORCH_COMPILE_DISABLE=1 VLLM_OPENVINO_KVCACHE_SPACE=8 \
  python -m vllm.entrypoints.openai.api_server --model <model_id>

# 실행 (GPU)
VLLM_OPENVINO_DEVICE=GPU TORCH_COMPILE_DISABLE=1 \
  python -m vllm.entrypoints.openai.api_server --model <model_id>
```

## 소스 마운트 테스트 (podman)

**빌드 없이 코드 수정을 즉시 반영**하는 방법. `vllm_openvino/` 디렉토리를 컨테이너에 마운트하면 파일 수정 후 컨테이너 재시작만으로 적용된다.

### 기본 명령어

```bash
# 서버 시작 (소스 마운트 + 모델 마운트)
podman run --replace -d --name vllm-server -p 8080:8080 \
  -v /home/user/project/vllm-openvino/vllm_openvino:/opt/app-root/vllm_openvino \
  -v /home/user/hf:/models:Z \
  quay.io/joopark/vllm-openvino \
  --port=8080 --model /models/<model_dir> --max-model-len 4096

# 서버 시작 대기 (Application startup complete 메시지 확인)
for i in $(seq 1 30); do
  if podman logs vllm-server 2>&1 | grep -q "Application startup complete"; then
    echo "Ready"; break
  fi
  if podman logs vllm-server 2>&1 | grep -q "EngineCore failed\|Engine core initialization failed"; then
    echo "FAILED"; podman logs vllm-server 2>&1 | grep -A5 "ERROR" | tail -20; break
  fi
  sleep 5
done

# 로그 확인 (에러 필터)
podman logs vllm-server 2>&1 | grep -A10 'ERROR\|Traceback' | grep -v 'Triton\|CUDA'

# 서버 중지
podman stop vllm-server

# 모델 목록 확인
curl -s http://localhost:8080/v1/models | python3 -m json.tool

# 추론 테스트 (completions)
curl -s http://localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/<model_dir>","prompt":"Hello","max_tokens":32}'

# 추론 테스트 (chat)
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/<model_dir>","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

### 반복 디버그 루프 패턴

```
1. 코드 수정 (vllm_openvino/*.py)
2. python3 -m py_compile <파일>  ← 문법 오류 사전 차단
3. podman stop vllm-server
4. podman run --replace -d ...   ← 재시작만으로 반영 (빌드 불필요)
5. 시작 대기 → 에러 확인 → 원인 분석 → 1번으로
```

### 모델 경로 규칙

- 호스트: `~/hf/<model_dir>`
- 컨테이너: `/models/<model_dir>`
- `-v /home/user/hf:/models:Z` 마운트로 연결 (`:Z` = SELinux 레이블)

### 주의사항

- 컨테이너 이미지 `quay.io/joopark/vllm-openvino`에 이미 vLLM + OpenVINO가 설치되어 있음
- `vllm_openvino/` 소스만 마운트로 교체 — vLLM core 등은 이미지 내 버전 사용
- 이미지 내 `vllm_openvino`는 `/opt/app-root/vllm_openvino`에 설치됨

---

## 테스트 인프라

**테스트 인프라 없음.** pytest, conftest.py, 테스트 파일 모두 없습니다.
실제 추론 테스트에는 OpenVINO 런타임 + 모델 파일이 필요하므로, 단위 테스트의 커버리지가 제한적입니다.

---

## ⛔ 하지 말아야 할 것들 (Evaluated & Rejected)

> 아래 항목들은 2026-03-08에 코드 증거 기반으로 평가한 결과, **불필요하거나 시기상조**로 판단되었습니다.
> 같은 고민을 반복하지 마세요.

### 1. PlatformEnum.OPENVINO 전환 — 불필요

**현재 상태**: `platform.py`에서 `_enum = PlatformEnum.CPU`를 사용 (OpenVINO 전용 enum 없음)

**왜 불필요한가**:
- vLLM의 Platform 시스템은 **Python 메서드 디스패치**로 동작함. enum 값이 아니라 클래스 메서드 오버라이드(`get_attn_backend_cls`, `check_and_update_config` 등)가 실제 분기를 결정
- `device_name: str = "openvino"`로 식별은 정상 동작
- PlatformEnum은 로깅/디버깅용 식별자일 뿐, 분기 로직에 사용되지 않음
- PlatformEnum.OPENVINO를 추가하려면 vLLM upstream에 PR을 보내야 하는데, 이 플러그인 단독으로는 불가

**재평가 조건**: vLLM upstream이 `if platform._enum == PlatformEnum.CPU:` 같은 enum 기반 분기를 추가하는 경우

### 2. InputBatch 아키텍처 리팩토링 (ModelRunner → Worker 이동) — 불필요, 오히려 해로움

**현재 상태**: InputBatch가 ModelRunner(`__init__`)와 Worker(`profile_run` 후)에서 각각 생성됨

**왜 불필요한가**:
- 이것은 **vLLM upstream GPUWorkerV1의 정확히 같은 패턴**. Worker가 프로파일링 후 InputBatch를 리셋하는 것은 의도된 설계
- ModelRunner의 초기 InputBatch는 초기화/셋업용 임시 객체
- 리팩토링하면 upstream 패턴에서 벗어나 **미래 호환성이 저하됨**
- 2곳에서 생성되는 건 맞지만, 각각 2줄짜리라 유지보수 부담 미미

**재평가 조건**: vLLM upstream이 InputBatch 소유권 패턴을 변경하는 경우

### 3. 테스트 인프라 추가 — 가치 제한적

**현재 상태**: 테스트 파일 0개, 14개 소스 파일

**왜 제한적인가**:
- 가장 위험한 버그들(추론 실패, GPU 메모리 크래시)은 **OpenVINO 런타임 없이 테스트 불가**
- 테스트 가능한 것: `_flatten_inputs()` 같은 순수 함수, env 파싱, import 검증 — 전체 버그의 ~30%
- 단일 개발자, 14파일 프로젝트에서 테스트 인프라 구축/유지 ROI가 낮음

**만약 추가한다면**: 순수 함수 위주 최소한의 pytest 셋업만. 추론 테스트는 별도 환경 필요.

### 4. CI/CD 파이프라인 — 현재 규모에서 과도

**현재 상태**: `.github/` 디렉토리 없음, 단일 개발자

**왜 과도한가**:
- CI가 할 수 있는 것(lint, py_compile)은 로컬에서 10초면 충분
- CI가 할 수 없는 것(추론 테스트, GPU 테스트)이 실제로 중요한 검증
- GitHub Actions 세팅/유지 오버헤드 > 얻는 가치

**재평가 조건**: 기여자가 2명 이상으로 늘어나거나, 릴리즈 빈도가 증가하는 경우

### 6. `str_to_torch_type` / `str_to_ov_type` 통합 — 불필요

**현재 상태**: `worker_v1/openvino_worker_v1.py`에 `str_to_torch_type`, `kv_cache.py`에 `str_to_ov_type` 각각 정의

**왜 불필요한가**:
- 두 매핑은 서로 **다른 타입 시스템**을 위한 것: 전자는 PyTorch dtype, 후자는 OpenVINO dtype
- 같은 문자열(`"fp16"` 등)을 받지만 반환 타입이 완전히 다름 → 단일 모듈로 통합 불가
- 실제 코드 경로에서 서로 교환하여 사용되지 않음

**재평가 조건**: 두 타입 시스템 간 변환이 공통 모듈로 추상화될 필요가 생기는 경우

### 7. 비동기 추론 파이프라인 — 실현 불가 (스케줄러 구조 제약)

**현재 상태**: `OpenVINOCausalLM.forward()`에서 `ov_request.start_async()` + 즉시 `wait()` 패턴 사용

**왜 실현 불가한가**:
- 비동기 파이프라인(배치 N 추론 중 배치 N+1 준비)은 vLLM 스케줄러가 **배치 N이 완료되어 샘플링된 토큰을 받아야** 다음 배치를 생성 가능
- 즉, 스케줄러가 순차적으로 동작하므로 플러그인 레벨의 비동기 파이프라이닝은 구조적으로 불가
- `start_async + wait`는 OpenVINO의 추론 실행 메커니즘이며, 이를 바꾸려면 vLLM 엔진 자체를 수정해야 함

**재평가 조건**: vLLM V2 엔진이 추론과 스케줄링을 분리하여 플러그인이 배치를 미리 요청할 수 있게 되는 경우

### 8. Structured outputs (문법 유도 디코딩) 지원 — 수요 없음, upstream 통합 필요

**현재 상태**: `sample_tokens()` 메서드에서 `grammar_output`을 무시하고 통과

**왜 제외되었나**:
- Structured outputs는 로짓 프로세서 레벨에서 마스킹이 필요하며, vLLM의 `outlines` 통합 경로를 거쳐야 함
- 단순히 `sample_tokens()`를 수정하는 것으로는 구현 불가
- 현재 사용자 수요 없음

**재평가 조건**: 사용자 요구가 생기거나 vLLM이 backends용 structured output 플러그인 인터페이스를 공개하는 경우

### 5. `openvino._offline_transformations` 교체 — 불필요 (2026.0.0 확인 완료)

**현재 상태**: `model_loader/openvino.py`에서 private API 사용
```python
from openvino._offline_transformations import paged_attention_transformation
```

**왜 불필요한가**:
- `paged_attention_transformation`은 모델을 PagedAttention으로 변환하는 **핵심 기능** — 대체 불가
- OpenVINO 2026.0.0 릴리즈 후 확인 결과: `_offline_transformations.paged_attention_transformation`은 **여전히 존재하고 정상 동작**
- 대체할 public API 없음 — 이전 버전 가드(`is_openvino_version("<", "2026.0.0")`)와 함께 있던 `_modify_cache_parameters()` 함수는 dead code로 판명되어 제거 완료
- `_offline_transformations`는 OpenVINO 생태계에서 사실상 public처럼 널리 사용됨

**재평가 조건**: OpenVINO 향후 버전에서 `paged_attention_transformation`의 public API 이동이 공식 발표되는 경우

---

## OpenVINO 2026.0.0 호환성 변경 이력

OpenVINO 2026.0.0으로 업그레이드 시 발생한 breaking change 및 대응 내역 (2026-03-09 완료).

| 변경 | 영향 파일 | 대응 |
|---|---|---|
| `ov.runtime.Coordinate` 삭제 (`ov.runtime` 모듈 전체 제거) | `attention/backends/openvino.py` | `ov.Coordinate`로 교체 (커밋 `0b6529a`) |
| `ov.Type.undefined` 삭제 | `model_executor/model_loader/openvino.py` | 해당 코드가 dead code(`_modify_cache_parameters()`)임을 확인 후 함수 전체 제거 (커밋 `49f9587`) |
| `paged_attention_transformation` (`_offline_transformations`) | `model_executor/model_loader/openvino.py` | 변경 없음 — private API 유지됨, 정상 동작 |
| `compile_model` 시 KV 캐시 처리 | — | 2026.0에서 플러그인이 자동 처리 → `_modify_cache_parameters()` 불필요 확인 |

> 향후 OpenVINO 버전 업그레이드 시 위 패턴을 참고할 것. 특히 `ov.runtime.*` 같은 하위 모듈 API는 삭제될 가능성이 있음.

---

## 알려진 기술적 특이사항

1. **`TORCH_COMPILE_DISABLE=1` 필수** — vLLM 0.13.0+에서 torch.compile/Inductor가 OpenVINO와 비호환. 이 env var 없으면 크래시
2. **Pin memory 미지원** — `is_pin_memory_available()` → False. CPU/OpenVINO 환경에서는 pin memory 불필요
3. **LoRA 미지원** — `check_and_update_config()`에서 assert로 차단
4. **단일 소켓만 지원** — `parallel_config.world_size == 1` 강제. Tensor/Pipeline 병렬 미지원
5. **KV 캐시 블록 크기** — CPU: 32, GPU: 16 (자동 오버라이드)
6. **KServe modelcar 호환성** — modelcar 방식으로 배포 시 `/mnt/models`가 symlink로 제공됨. 로컬 pre-exported IR은 `ov_core.read_model()` 직접 로딩으로 처리. 최적화: 로컬 IR만 지원 (2026-04-21).
7. **OpenVINO import 실패 처리** — `platform.py`에서 `import openvino` 실패 시 `ov = None`으로 설정하고 warning만 출력. 실제 사용 시점인 `check_and_update_config()`에서 `ImportError`를 raise. **import 시점에서 raise하지 않는 이유**: vLLM 플러그인 디스커버리 메커니즘이 모든 플러그인을 import한 뒤 활성 플러그인을 선택하므로, import 시점 raise는 OpenVINO 플러그인이 아닌 다른 플러그인 사용 시에도 크래시를 유발함.

## vLLM 버전 업그레이드 체크리스트

vLLM 버전 업그레이드 시 다음 사항을 확인하세요:

1.  **업그레이드 전 준비**:
    *   vLLM 릴리즈 노트를 확인하여 변경 사항을 파악합니다.
    *   특히 `vllm.v1.*` 경로의 내부 API 변경 여부를 주의 깊게 살펴봅니다.

2.  **🔴 높은 위험 import 확인**:
    *   `vllm.v1.kv_cache_interface` — `KVCacheSpec`, `KVCacheConfig`, `FullAttentionSpec`, `AttentionSpec`
    *   `vllm.v1.attention.backend` — `AttentionBackend`, `AttentionMetadata`
    *   `vllm.v1.attention.backends.utils` — `CommonAttentionMetadata`
    *   `vllm.v1.attention.backend` — `AttentionMetadataBuilder` (v0.18.1에서 이동됨, 기존: `backends.utils`)
    *   `vllm.v1.outputs` — `SamplerOutput`, `ModelRunnerOutput`
    *   `vllm.v1.sample.metadata` — `SamplingMetadata`
    *   `vllm.v1.sample.sampler` — `Sampler`
    *   `vllm.v1.worker.gpu_input_batch` — `CachedRequestState`, `InputBatch`
    *   `vllm.v1.worker.worker_base` — `WorkerBase`
    *   `vllm.v1.worker.utils` — `bind_kv_cache`
    *   `vllm.v1.core.sched.output` — `SchedulerOutput`, `NewRequestData`, `CachedRequestData`
    (자세한 내용은 "핵심 의존 관계" 섹션을 참조하세요.)

3.  **검증 방법**:
    *   `python3 -m py_compile vllm_openvino/**/*.py`를 실행하여 문법 오류를 확인합니다.
    *   Podman 테스트 환경에서 실제 추론을 실행하여 기능적 회귀가 없는지 검증합니다.

