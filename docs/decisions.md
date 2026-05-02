# Evaluated & Rejected Decisions

> 아래 항목들은 코드 증거 기반으로 평가한 결과, **불필요하거나 시기상조**로 판단되었습니다.
> 같은 고민을 반복하지 마세요.

---

## 1. PlatformEnum.OPENVINO 전환 — 불필요

**현재 상태**: `platform.py`에서 `_enum = PlatformEnum.CPU`를 사용 (OpenVINO 전용 enum 없음)

**왜 불필요한가**:
- vLLM의 Platform 시스템은 **Python 메서드 디스패치**로 동작함. enum 값이 아니라 클래스 메서드 오버라이드(`get_attn_backend_cls`, `check_and_update_config` 등)가 실제 분기를 결정
- `device_name: str = "openvino"`로 식별은 정상 동작
- PlatformEnum은 로깅/디버깅용 식별자일 뿐, 분기 로직에 사용되지 않음
- PlatformEnum.OPENVINO를 추가하려면 vLLM upstream에 PR을 보내야 하는데, 이 플러그인 단독으로는 불가

**재평가 조건**: vLLM upstream이 `if platform._enum == PlatformEnum.CPU:` 같은 enum 기반 분기를 추가하는 경우

---

## 2. InputBatch 아키텍처 리팩토링 (ModelRunner → Worker 이동) — 불필요, 오히려 해로움

**현재 상태**: InputBatch가 ModelRunner(`__init__`)와 Worker(`profile_run` 후)에서 각각 생성됨

**왜 불필요한가**:
- 이것은 **vLLM upstream GPUWorkerV1의 정확히 같은 패턴**. Worker가 프로파일링 후 InputBatch를 리셋하는 것은 의도된 설계
- ModelRunner의 초기 InputBatch는 초기화/셋업용 임시 객체
- 리팩토링하면 upstream 패턴에서 벗어나 **미래 호환성이 저하됨**
- 2곳에서 생성되는 건 맞지만, 각각 2줄짜리라 유지보수 부담 미미

**재평가 조건**: vLLM upstream이 InputBatch 소유권 패턴을 변경하는 경우

---

## 3. 테스트 인프라 추가 — 가치 제한적

**현재 상태**: 테스트 파일 0개, 14개 소스 파일

**왜 제한적인가**:
- 가장 위험한 버그들(추론 실패, GPU 메모리 크래시)은 **OpenVINO 런타임 없이 테스트 불가**
- 테스트 가능한 것: `_flatten_inputs()` 같은 순수 함수, env 파싱, import 검증 — 전체 버그의 ~30%
- 단일 개발자, 14파일 프로젝트에서 테스트 인프라 구축/유지 ROI가 낮음

**만약 추가한다면**: 순수 함수 위주 최소한의 pytest 셋업만. 추론 테스트는 별도 환경 필요.

---

## 4. CI/CD 파이프라인 — 현재 규모에서 과도

**현재 상태**: `.github/` 디렉토리 없음, 단일 개발자

**왜 과도한가**:
- CI가 할 수 있는 것(lint, py_compile)은 로컬에서 10초면 충분
- CI가 할 수 없는 것(추론 테스트, GPU 테스트)이 실제로 중요한 검증
- GitHub Actions 세팅/유지 오버헤드 > 얻는 가치

**재평가 조건**: 기여자가 2명 이상으로 늘어나거나, 릴리즈 빈도가 증가하는 경우

---

## 5. `openvino._offline_transformations` 교체 — 불필요 (2026.0.0 확인 완료)

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

## 6. `str_to_torch_type` / `str_to_ov_type` 통합 — 불필요

**현재 상태**: `worker_v1/openvino_worker_v1.py`에 `str_to_torch_type`, `kv_cache.py`에 `str_to_ov_type` 각각 정의

**왜 불필요한가**:
- 두 매핑은 서로 **다른 타입 시스템**을 위한 것: 전자는 PyTorch dtype, 후자는 OpenVINO dtype
- 같은 문자열(`"fp16"` 등)을 받지만 반환 타입이 완전히 다름 → 단일 모듈로 통합 불가
- 실제 코드 경로에서 서로 교환하여 사용되지 않음

**재평가 조건**: 두 타입 시스템 간 변환이 공통 모듈로 추상화될 필요가 생기는 경우

---

## 7. 비동기 추론 파이프라인 — 실현 불가 (스케줄러 구조 제약)

**현재 상태**: `OpenVINOCausalLM.forward()`에서 `ov_request.start_async()` + 즉시 `wait()` 패턴 사용

**왜 실현 불가한가**:
- 비동기 파이프라인(배치 N 추론 중 배치 N+1 준비)은 vLLM 스케줄러가 **배치 N이 완료되어 샘플링된 토큰을 받아야** 다음 배치를 생성 가능
- 즉, 스케줄러가 순차적으로 동작하므로 플러그인 레벨의 비동기 파이프라이닝은 구조적으로 불가
- `start_async + wait`는 OpenVINO의 추론 실행 메커니즘이며, 이를 바꾸려면 vLLM 엔진 자체를 수정해야 함

**재평가 조건**: vLLM V2 엔진이 추론과 스케줄링을 분리하여 플러그인이 배치를 미리 요청할 수 있게 되는 경우

---

## 8. Structured outputs (문법 유도 디코딩) 지원 — 수요 없음, upstream 통합 필요

**현재 상태**: `sample_tokens()` 메서드에서 `grammar_output`을 무시하고 통과

**왜 제외되었나**:
- Structured outputs는 로짓 프로세서 레벨에서 마스킹이 필요하며, vLLM의 `outlines` 통합 경로를 거쳐야 함
- 단순히 `sample_tokens()`를 수정하는 것으로는 구현 불가
- 현재 사용자 수요 없음

**재평가 조건**: 사용자 요구가 생기거나 vLLM이 backends용 structured output 플러그인 인터페이스를 공개하는 경우

---

## Refactoring History

### 2026-05-02: Safe Refactoring (Magic Numbers & Utilities)

**Completed:**
1. Extracted `format_memory_size()` from `worker_v1/openvino_worker_v1.py:profile_run()` to `utils.py` as module-level function (Task 1)
2. Added named constants in `platform.py`: `GIB_BYTES`, `CPU_BLOCK_SIZE`, `GPU_BLOCK_SIZE`, `DEFAULT_CPU_KV_CACHE_GB` (Task 2)
3. Added `USED_MEMORY_THRESHOLD` constant in `worker_v1/openvino_worker_v1.py` (Task 3)

**Rationale:** Improve code readability by eliminating magic numbers. All changes follow upstream patterns and maintain minimalism.

**Skipped (from original plan):**
- Step C (document duplicate type maps): AGENTS.md already explains these are intentionally separate (different type systems: OpenVINO vs PyTorch)
- Step D (break down long functions): Would break upstream pattern compatibility, against project's "upstream 패턴 추종" principle
