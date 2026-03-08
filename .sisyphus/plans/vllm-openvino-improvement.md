# vllm-openvino 개선 + Vault 지식 정리 통합 계획

## TL;DR

> **Quick Summary**: vllm-openvino 프로젝트의 45+ 개선점을 수정하고, 발견한 지식을 Obsidian vault 5개 노트로 정리하여 재사용 가능하게 한다.
> 
> **Deliverables**:
> - Vault 노트 5개 (한국어 중심, Obsidian에 작성)
> - 코드 버그 수정 (GPU crash, dead stub, 로그 오타 등)
> - 코드 품질 개선 (미사용 import, .bak 파일, .gitignore)
> - Containerfile 보안 개선 (non-root user)
> 
> **Estimated Effort**: Medium (2-3일)
> **Parallel Execution**: YES — 4 waves
> **Critical Path**: V1(vault notes) → T1-T3(critical bugs) → T4-T10(cosmetic) → T11(container)

---

## Context

### Original Request
vllm-openvino 프로젝트에 개선점이 있는지 찾고, 나중에도 재사용될 수 있도록 관련 지식을 Obsidian vault에 잘 정리할 계획을 세워달라.

### Interview Summary
**Key Discussions**:
- 전체 소스코드 14개 .py 파일 직접 분석 완료
- 5개 병렬 에이전트 (explore 3개 + librarian 2개) 투입
- 6개 카테고리에서 45+ 개선점 발견
- 사용자 확인: vault 5개 노트 구조, 한국어, 코드 개선 포함

**Research Findings**:
- vLLM 플러그인 시스템: entry_points 패턴 올바르게 구현됨
- V1 아키텍처: WorkerBase → ModelRunnerBase 분리 패턴 vs 현재 ModelRunner가 InputBatch 관리 (불일치)
- 17+ 내부 vLLM import — 모든 vLLM 업데이트에서 breakage 위험
- OpenVINO private API (`_offline_transformations`) — deprecation 위험
- GitHub 이슈: #14933(분산추론 미지원), #11398(의존성 충돌) 등

### Metis Review
**Identified Gaps** (addressed):
- `{{}}` GPU crash 버그 발견 (kv_cache.py:103-104) → 계획에 추가
- Dead stub `determine_num_available_blocks` 발견 (worker_v1.py:458-459) → 계획에 추가
- profile_run 로그 순서 문제 발견 → 계획에 추가
- GPU assertion (`assert not is_openvino_cpu()`)은 버그가 아님 — 정상 동작 확인
- PlatformEnum 변경은 연구 필요 — 즉시 변경 금지
- kv_cache 스텁 메소드 (clone/grow/get_slot_kv_cache) 인터페이스 필수 여부 미확인
- Vault 노트는 코드 변경 전에 작성 (현재 상태 캡처)

---

## Work Objectives

### Core Objective
vllm-openvino 프로젝트의 코드 품질을 개선하고, 분석 과정에서 축적한 지식을 Obsidian vault에 체계적으로 정리하여 향후 재사용 가능하게 한다.

### Concrete Deliverables
1. Obsidian vault 노트 5개 (한국어 중심)
2. GPU crash 버그 수정 (kv_cache.py)
3. Dead stub 수정 (worker_v1.py)
4. 로그 메시지 오타/순서 수정 (platform.py, worker_v1.py)
5. 미사용 import 제거, .bak 파일 삭제, .gitignore 개선
6. Containerfile non-root user 추가

### Definition of Done
- [ ] 5개 vault 노트가 Obsidian에 작성되고 읽기 가능
- [ ] `grep -n '{{}}' vllm_openvino/kv_cache.py` → 0 matches
- [ ] `python3 -m py_compile` 모든 수정 파일 PASS
- [ ] `ls *.bak 2>/dev/null` → 결과 없음

### Must Have
- GPU crash 버그(`{{}}`) 수정
- Dead stub 수정
- 로그 오타 수정 ("bp16", "f16")
- Vault 노트 5개 작성
- .bak 파일 삭제

### Must NOT Have (Guardrails)
- ❌ PlatformEnum 값 변경 (연구 필요 — 이 계획에서 변경 금지)
- ❌ `is_openvino_version("<", "2026.0.0")` version guard 제거
- ❌ InputBatch 아키텍처 리팩토링 (ModelRunner → Worker 이동)
- ❌ `assert not is_openvino_cpu()` in `profile_run` 제거 (정상 코드)
- ❌ kv_cache.py clone/grow/get_slot_kv_cache 스텁 수정 (인터페이스 요구사항 미확인)
- ❌ openvino._offline_transformations 교체 (public API 연구 필요)
- ❌ vLLM 버전 업그레이드 (0.13.0에서 변경 금지)
- ❌ 테스트 인프라 추가 (별도 계획 필요)
- ❌ CI/CD 파이프라인 추가 (별도 계획 필요)
- ❌ Vault 노트를 45개 이슈 모두 "확인된 버그"로 기술 (일부는 개발자의 의문 코멘트)
- ❌ 버그 수정과 오타 수정을 같은 커밋에 묶기

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO
- **Automated tests**: None (테스트 인프라 추가는 이 계획 범위 밖)
- **Framework**: none

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Code fixes**: Use Bash (grep, py_compile) — 패턴 검증, 구문 검증
- **Vault notes**: Use obsidian_read_note — 노트 존재 및 내용 확인
- **Container**: Use Bash (grep) — Containerfile 내용 검증

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — Vault 지식 정리, 현재 상태 캡처):
├── V1: 코드베이스 분석 vault 노트 [writing]
├── V2: 개발 운영 runbook vault 노트 [writing]
├── V3: vLLM 플러그인 아키텍처 가이드 vault 노트 [writing]
├── V4: OpenVINO LLM 최적화 가이드 vault 노트 [writing]
└── V5: 업스트림 호환성 가이드 vault 노트 [writing]

Wave 2 (After Wave 1 — Critical 버그 수정):
├── T1: {{}} GPU crash 수정 [quick]
├── T2: Dead stub 수정 [quick]
├── T3: profile_run 로그 순서 수정 [quick]
└── T4: 로그 오타 수정 (bp16, f16) [quick]

Wave 3 (After Wave 2 — Cosmetic 개선):
├── T5: 미사용 import 제거 [quick]
├── T6: KV cache precision 코드 단순화 [quick]
├── T7: .bak 파일 삭제 + .gitignore 업데이트 [quick]
├── T8: 불확실 주석 정리 [quick]
└── T9: format() double-wrapping 수정 [quick]

Wave 4 (After Wave 3 — Container 보안):
└── T10: Containerfile non-root user + EXPOSE [quick]

Wave FINAL (After ALL — 검증):
├── F1: Plan compliance audit [oracle]
├── F2: Code quality review [unspecified-high]
├── F3: Vault notes QA [unspecified-high]
└── F4: Scope fidelity check [deep]

Critical Path: V1-V5 → T1 → T2 → T4 → T7 → F1-F4
Parallel Speedup: ~60% faster than sequential
Max Concurrent: 5 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|-----------|--------|
| V1-V5 | — | T1-T10 |
| T1-T4 | V1-V5 | T5-T9 |
| T5-T9 | T1-T4 | T10 |
| T10 | T5-T9 | F1-F4 |
| F1-F4 | T10 | — |

### Agent Dispatch Summary

| Wave | Tasks | Categories |
|------|-------|-----------|
| 1 | 5 | V1-V5 → `writing` |
| 2 | 4 | T1-T4 → `quick` |
| 3 | 5 | T5-T9 → `quick` |
| 4 | 1 | T10 → `quick` |
| FINAL | 4 | F1→`oracle`, F2→`unspecified-high`, F3→`unspecified-high`, F4→`deep` |

---

## TODOs

> Implementation + QA = ONE Task. Every task has Agent-Executed QA Scenarios.

---

### Wave 1: Vault 지식 정리 (5 tasks, parallel)

- [x] V1. 코드베이스 분석 Vault 노트

  **What to do**:
  - Obsidian vault에 `20_AREAS/AI-Infrastructure/vllm-openvino-codebase-analysis.md` 작성
  - 내용: 프로젝트 구조, 파일별 역할, 발견된 45+ 개선점 (카테고리별), 각 이슈의 file:line 참조
  - 프론트매터: title, date, tags(vllm, openvino, code-analysis), status: published
  - 기존 노트 `vllm-openvino-containerfile-guide.md` 링크 포함
  - 언어: 한국어 중심, 코드/기술용어는 영어

  **Must NOT do**:
  - 모든 이슈를 "확인된 버그"로 기술하지 말 것 — 일부는 개발자의 의문 코멘트
  - 300줄 초과 금지

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with V2, V3, V4, V5)
  - **Blocks**: T1-T10
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `vllm_openvino/platform.py` — 프로젝트의 핵심 플랫폼 등록 모듈, 불확실 주석 5개 위치
  - `vllm_openvino/kv_cache.py` — KV 캐시 엔진, GPU crash 버그 위치 (L103-104)
  - `vllm_openvino/worker_v1/openvino_worker_v1.py` — V1 워커, dead stub 위치 (L458-459)
  - `vllm_openvino/model_executor/model_loader/openvino.py` — 모델 로더, TODO (L226)

  **External References**:
  - vLLM Plugin System: https://docs.vllm.ai/en/v0.15.0/design/plugin_system/
  - OpenVINO Weight Compression: https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Vault 노트 존재 확인
    Tool: Bash (obsidian_read_note via MCP)
    Steps:
      1. obsidian_read_note path="20_AREAS/AI-Infrastructure/vllm-openvino-codebase-analysis.md"
      2. 응답에 "vllm-openvino" 텍스트 존재 확인
      3. 프론트매터에 tags 배열 존재 확인
    Expected Result: 노트 내용이 반환되고, 제목/태그/본문 존재
    Evidence: .sisyphus/evidence/V1-vault-note-exists.txt

  Scenario: 필수 섹션 존재 확인
    Tool: Bash (obsidian_read_note via MCP)
    Steps:
      1. 노트 내용에서 "## 프로젝트 구조" 또는 "## Project Structure" 존재 확인
      2. "file:line" 또는 파일 경로 참조 존재 확인
      3. "## 개선점" 또는 "## Improvements" 섹션 존재 확인
    Expected Result: 3개 필수 섹션 모두 존재
    Evidence: .sisyphus/evidence/V1-vault-note-sections.txt
  ```

  **Commit**: NO (vault 노트는 git commit 대상 아님)

---

- [x] V2. 개발 운영 Runbook Vault 노트

  **What to do**:
  - `30_RESOURCES/runbooks/AI/vllm-openvino-development-runbook.md` 작성
  - 내용: 빌드 방법 (pip, container), 핫패치 프로토콜 (podman run/cp/commit), 환경변수 가이드, 디버깅 팁
  - 기존 `vllm-optimizer-runbook.md` 스타일 참고하여 포맷 일치
  - 핫패치 명령어는 `.sisyphus/plans/vllm_upgrade_plan.md`의 "Hot Patching Reference" 섹션에서 참조

  **Must NOT do**:
  - 300줄 초과 금지
  - 전체 API 문서화 금지 — 운영 지식만

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with V1, V3, V4, V5)
  - **Blocks**: T1-T10
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `30_RESOURCES/runbooks/AI/vllm-optimizer-runbook.md` — Vault runbook 스타일/포맷 참조
  - `.sisyphus/plans/vllm_upgrade_plan.md:92-109` — Hot Patching Reference 섹션
  - `Containerfile` — 빌드 설정 참조
  - `README.md` — 환경변수 설명 참조

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Runbook 노트 존재 및 구조 확인
    Tool: Bash (obsidian_read_note via MCP)
    Steps:
      1. obsidian_read_note path="30_RESOURCES/runbooks/AI/vllm-openvino-development-runbook.md"
      2. "## 빌드" 또는 "## Build" 섹션 존재 확인
      3. "podman" 또는 "docker" 명령어 포함 확인
      4. 환경변수 테이블 또는 목록 존재 확인
    Expected Result: 운영 가이드 구조가 완비됨
    Evidence: .sisyphus/evidence/V2-runbook-exists.txt
  ```

  **Commit**: NO

---

- [x] V3. vLLM 플러그인 아키텍처 가이드 Vault 노트

  **What to do**:
  - `40_KNOWLEDGE/RedHat/vllm-plugin-architecture-guide.md` 작성
  - 내용: vLLM 플러그인 시스템 개요, entry_points 등록 패턴, V1 아키텍처 (WorkerBase → ModelRunnerBase), Platform/Worker/ModelRunner/Attention 계층, 커뮤니티 플러그인 비교 (TPU, Neuron)
  - Librarian 리서치 결과 (URL, key findings) 포함
  - 이 프로젝트에 국한되지 않는 **일반 지식** — 다른 vLLM 플러그인 프로젝트에도 재사용 가능

  **Must NOT do**:
  - vllm-openvino 프로젝트 특정 이슈 포함 금지 (그건 V1 노트)
  - 300줄 초과 금지

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with V1, V2, V4, V5)
  - **Blocks**: T1-T10
  - **Blocked By**: None

  **References**:

  **External References**:
  - Plugin System: https://docs.vllm.ai/en/v0.15.0/design/plugin_system/
  - V1 Architecture: https://docs.vllm.ai/en/stable/design/arch_overview.html
  - WorkerBase API: https://docs.vllm.ai/en/v0.15.0/api/vllm/v1/worker/worker_base/
  - GPUModelRunner: https://docs.vllm.ai/en/v0.10.1.1/api/vllm/v1/worker/gpu_model_runner.html

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 일반 지식 노트 존재 확인
    Tool: Bash (obsidian_read_note via MCP)
    Steps:
      1. obsidian_read_note path="40_KNOWLEDGE/RedHat/vllm-plugin-architecture-guide.md"
      2. "entry_points" 또는 "entry-points" 텍스트 존재 확인
      3. "WorkerBase" 텍스트 존재 확인
      4. URL 참조 1개 이상 존재 확인
    Expected Result: 플러그인 아키텍처 가이드 완비
    Evidence: .sisyphus/evidence/V3-plugin-guide-exists.txt
  ```

  **Commit**: NO

---

- [x] V4. OpenVINO LLM 최적화 가이드 Vault 노트

  **What to do**:
  - `40_KNOWLEDGE/RedHat/openvino-llm-optimization-guide.md` 작성
  - 내용: OpenVINO LLM 최적화 패턴 (Weight Compression 4/8bit, PagedAttention, Dynamic Split-Fuse, KV Cache 관리, 모델 캐싱), CPU/GPU 성능 팁, 벤치마크 방법론
  - Librarian 리서치의 OpenVINO 관련 URL/findings 포함
  - vllm-openvino에 국한되지 않는 **일반 지식**

  **Must NOT do**:
  - 프로젝트 특정 코드 참조 금지
  - 300줄 초과 금지

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with V1, V2, V3, V5)
  - **Blocks**: T1-T10
  - **Blocked By**: None

  **References**:

  **External References**:
  - Weight Compression: https://docs.openvino.ai/2025/openvino-workflow/model-optimization-guide/weight-compression.html
  - Efficient LLM Serving: https://docs.openvino.ai/2024/openvino-workflow/model-server/ovms_docs_llm_reference.html
  - Training-time Optimization: https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/compressing-models-during-training.html

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 최적화 가이드 노트 존재 확인
    Tool: Bash (obsidian_read_note via MCP)
    Steps:
      1. obsidian_read_note path="40_KNOWLEDGE/RedHat/openvino-llm-optimization-guide.md"
      2. "Weight Compression" 또는 "양자화" 텍스트 존재 확인
      3. "PagedAttention" 텍스트 존재 확인
    Expected Result: OpenVINO 최적화 가이드 완비
    Evidence: .sisyphus/evidence/V4-optimization-guide-exists.txt
  ```

  **Commit**: NO

---

- [x] V5. 업스트림 호환성 가이드 Vault 노트

  **What to do**:
  - `20_AREAS/AI-Infrastructure/vllm-openvino-upstream-compatibility.md` 작성
  - 내용: vLLM v0.13.0 fragile import 전체 매핑 (17개+ import path, 소스 파일, 라인), 각 import의 fragility 등급, 버전 업그레이드 전략, 깨질 가능성 높은 API 목록, 호환성 테스트 체크리스트
  - Architecture agent 결과의 import 분석 데이터 포함
  - OpenVINO 버전 호환성 (2025.4.1 vs 2026.0.0 version guard) 포함

  **Must NOT do**:
  - 코드 수정 금지 — 분석/문서만
  - 300줄 초과 금지

  **Recommended Agent Profile**:
  - **Category**: `writing`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with V1, V2, V3, V4)
  - **Blocks**: T1-T10
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `vllm_openvino/worker_v1/openvino_worker_v1.py:8-24` — vLLM 내부 import 집중 지점
  - `vllm_openvino/model_executor/model_loader/openvino.py:15-21` — vLLM v1 sample/output imports
  - `vllm_openvino/attention/backends/openvino.py:9-15` — attention 추상화 imports
  - `vllm_openvino/model_executor/model_loader/openvino.py:189` — `is_openvino_version("<", "2026.0.0")` version guard

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 호환성 가이드 노트 존재 확인
    Tool: Bash (obsidian_read_note via MCP)
    Steps:
      1. obsidian_read_note path="20_AREAS/AI-Infrastructure/vllm-openvino-upstream-compatibility.md"
      2. "vllm.v1" 텍스트 존재 확인 (import path 매핑)
      3. "fragile" 또는 "취약" 텍스트 존재 확인
      4. import path 테이블 또는 목록 존재 확인
    Expected Result: 업스트림 호환성 가이드 완비
    Evidence: .sisyphus/evidence/V5-compatibility-guide-exists.txt
  ```

  **Commit**: NO

---

### Wave 2: Critical 버그 수정 (4 tasks, parallel)

- [x] T1. `{{}}` GPU Crash 버그 수정

  **What to do**:
  - `vllm_openvino/kv_cache.py` 라인 103-104에서 `{{}}` → `{}` 수정
  - Python에서 `{{}}` = `set({})` → `TypeError: unhashable type 'dict'` 런타임 crash
  - GPU 경로의 `remote_context.create_tensor()` 호출 시 발생

  **Must NOT do**:
  - GPU 로직 변경 금지 — `{{}}` → `{}` 단일 수정만

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T2, T3, T4)
  - **Blocks**: T5-T9
  - **Blocked By**: V1-V5

  **References**:
  - `vllm_openvino/kv_cache.py:103-104` — GPU crash 위치

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: {{}} 패턴 완전 제거 확인
    Tool: Bash (grep)
    Steps:
      1. grep -n '{{}}' vllm_openvino/kv_cache.py
    Expected Result: grep 결과 없음 (exit code 1)
    Evidence: .sisyphus/evidence/T1-gpu-crash-fixed.txt

  Scenario: 구문 유효성 확인
    Tool: Bash (python3 -m py_compile)
    Steps:
      1. python3 -m py_compile vllm_openvino/kv_cache.py
    Expected Result: 컴파일 성공 (exit code 0)
    Evidence: .sisyphus/evidence/T1-py-compile.txt
  ```

  **Commit**: YES
  - Message: `fix: correct GPU KV cache allocation crash`
  - Files: `vllm_openvino/kv_cache.py`
  - Pre-commit: `python3 -m py_compile vllm_openvino/kv_cache.py`

---

- [x] T2. Dead stub 수정 + profile_run 로그 순서 수정

  **What to do**:
  - `vllm_openvino/worker_v1/openvino_worker_v1.py`:
    - 라인 458-459: `determine_num_available_blocks` — `self.kv_cache_config` 참조를 `raise NotImplementedError("Use determine_available_memory()")` 으로 교체
    - 라인 317-321: "Start profiling run" 로그를 `execute_model()` 호출 **전**으로 이동
    - 라인 367, 369: `format(format_memory_size(...))` → `format_memory_size(...)` (이중 래핑 제거)

  **Must NOT do**:
  - Worker 아키텍처 변경 금지
  - profile_run 로직 변경 금지 — 로그 위치와 format 정리만

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T1, T3, T4)
  - **Blocks**: T5-T9
  - **Blocked By**: V1-V5

  **References**:
  - `vllm_openvino/worker_v1/openvino_worker_v1.py:458-459` — dead stub
  - `vllm_openvino/worker_v1/openvino_worker_v1.py:306-321` — log ordering
  - `vllm_openvino/worker_v1/openvino_worker_v1.py:367,369` — format double-wrap

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Dead 속성 참조 제거
    Tool: Bash (grep)
    Steps:
      1. grep -n 'self.kv_cache_config.num_blocks' vllm_openvino/worker_v1/openvino_worker_v1.py
    Expected Result: 0 matches
    Evidence: .sisyphus/evidence/T2-dead-stub.txt

  Scenario: Double-wrap 제거
    Tool: Bash (grep)
    Steps:
      1. grep -n 'format(format_memory_size' vllm_openvino/worker_v1/openvino_worker_v1.py
    Expected Result: 0 matches
    Evidence: .sisyphus/evidence/T2-double-wrap.txt

  Scenario: 구문 유효성
    Tool: Bash (python3 -m py_compile)
    Steps:
      1. python3 -m py_compile vllm_openvino/worker_v1/openvino_worker_v1.py
    Expected Result: 컴파일 성공
    Evidence: .sisyphus/evidence/T2-py-compile.txt
  ```

  **Commit**: YES
  - Message: `fix: remove dead stub, fix log ordering and format double-wrap in worker`
  - Files: `vllm_openvino/worker_v1/openvino_worker_v1.py`

---

- [x] T3. 로그 메시지 오타 수정

  **What to do**:
  - `vllm_openvino/platform.py`:
    - 라인 108: `"bp16"` → `"bf16"`
    - 라인 113: 로그에서 `"f16"` → `"f32"` (fp32/f32 케이스의 로그 메시지)
    - 라인 66: `"OpenViNO"` → `"OpenVINO"` (대문자 N 오타)

  **Must NOT do**:
  - KV cache 로직 변경 금지 — 문자열만 수정

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T1, T2, T4)
  - **Blocks**: T5-T9
  - **Blocked By**: V1-V5

  **References**:
  - `vllm_openvino/platform.py:66,108,113` — 오타 위치

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 오타 완전 제거
    Tool: Bash (grep)
    Steps:
      1. grep -n '"bp16"' vllm_openvino/platform.py → 0 matches
      2. grep -n 'OpenViNO' vllm_openvino/platform.py → 0 matches
    Expected Result: 모든 오타 제거됨
    Evidence: .sisyphus/evidence/T3-typos-fixed.txt
  ```

  **Commit**: YES
  - Message: `fix: correct log message typos (bp16→bf16, OpenViNO→OpenVINO)`
  - Files: `vllm_openvino/platform.py`

---

- [x] T4. 미사용 import 제거

  **What to do**:
  - `vllm_openvino/platform.py`:
    - 라인 6: `import vllm.envs as vllm_envs` — 파일 어디에서도 `vllm_envs` 미사용 → 삭제
    - 라인 22: `import openvino.properties.hint as hints` — `hints` 미사용 → 삭제
  - 라인 10의 `# not sure if this is a optimal solution!` 주석도 제거 (불확실 주석 정리)

  **Must NOT do**:
  - 로직 변경 금지 — import와 주석만 정리

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with T1, T2, T3)
  - **Blocks**: T5-T9
  - **Blocked By**: V1-V5

  **References**:
  - `vllm_openvino/platform.py:6,10,22` — 미사용 import 위치

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 미사용 import 제거
    Tool: Bash (grep)
    Steps:
      1. grep -n 'vllm_envs' vllm_openvino/platform.py → 0 matches
      2. grep -n 'as hints' vllm_openvino/platform.py → 0 matches
      3. python3 -m py_compile vllm_openvino/platform.py → 성공
    Expected Result: 미사용 import 완전 제거, 컴파일 성공
    Evidence: .sisyphus/evidence/T4-unused-imports.txt
  ```

  **Commit**: YES (groups with T3 — 같은 파일)
  - Message: `refactor: remove unused imports and uncertain comments in platform.py`
  - Files: `vllm_openvino/platform.py`

---

### Wave 3: Cosmetic 개선 (5 tasks, parallel)

- [x] T5. KV cache precision if/elif → dict lookup 단순화

  **What to do**:
  - `vllm_openvino/platform.py` 라인 95-119의 25줄 if/elif/elif... 체인을 dict 매핑으로 교체
  - 예: `precision_map = {"u8": "u8", "i8": "i8", "fp16": "f16", "f16": "f16", ...}`
  - 로그 메시지 패턴도 통합 (동일 포맷 반복 제거)

  **Must NOT do**:
  - KV cache 동작 변경 금지 — 동일한 입력에 동일한 결과 보장

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T6-T9)
  - **Blocks**: T10
  - **Blocked By**: T1-T4

  **References**:
  - `vllm_openvino/platform.py:95-119` — 현재 if/elif 체인

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: if/elif 체인 단순화 확인
    Tool: Bash (grep)
    Steps:
      1. grep -c 'VLLM_OPENVINO_KV_CACHE_PRECISION ==' vllm_openvino/platform.py
    Expected Result: 0 또는 1 (dict lookup으로 교체)
    Evidence: .sisyphus/evidence/T5-dict-lookup.txt
  ```

  **Commit**: YES
  - Message: `refactor: simplify KV cache precision handling with dict lookup`
  - Files: `vllm_openvino/platform.py`

---

- [x] T6. 불확실 주석 정리

  **What to do**:
  - `vllm_openvino/platform.py`에서 불확실/의문 주석을 정리:
    - 라인 29: `# Check! What is the right selection?` → `# PlatformEnum.CPU is used because PlatformEnum.OPENVINO may not exist in vLLM 0.13.0. See upstream-compatibility vault note.`
    - 라인 32: `#dispatch_key: str = "CPU" # Is this still required?` → 삭제 (이미 주석처리된 dead code)
    - 라인 154-155: 주석처리된 assert 2개 삭제

  **Must NOT do**:
  - PlatformEnum 값 변경 금지 (연구 필요)
  - 로직 변경 금지

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T5, T7-T9)
  - **Blocks**: T10
  - **Blocked By**: T1-T4

  **References**:
  - `vllm_openvino/platform.py:29,32,154-155` — 불확실 주석 위치

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 의문 주석 제거 확인
    Tool: Bash (grep)
    Steps:
      1. grep -n 'Check!' vllm_openvino/platform.py → 0 matches
      2. grep -n 'Is this still required' vllm_openvino/platform.py → 0 matches
    Expected Result: 의문형 주석 없음
    Evidence: .sisyphus/evidence/T6-comments-cleaned.txt
  ```

  **Commit**: YES (groups with T5 — 같은 파일)

---

- [x] T7. .bak 파일 삭제 + .gitignore 업데이트

  **What to do**:
  - `Containerfile.bak` 삭제
  - `pyproject.toml.bak` 삭제
  - `.gitignore`에 추가: `*.bak`, `.env`, `.venv/`, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`

  **Must NOT do**:
  - Containerfile (현재 사용 중) 삭제 금지

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T5, T6, T8, T9)
  - **Blocks**: T10
  - **Blocked By**: T1-T4

  **References**:
  - `Containerfile.bak` — 삭제 대상
  - `pyproject.toml.bak` — 삭제 대상
  - `.gitignore` — 업데이트 대상

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: .bak 파일 완전 삭제
    Tool: Bash (ls)
    Steps:
      1. ls *.bak 2>/dev/null
    Expected Result: 파일 없음 (exit code 2)
    Evidence: .sisyphus/evidence/T7-bak-removed.txt

  Scenario: .gitignore 업데이트 확인
    Tool: Bash (grep)
    Steps:
      1. grep '\.bak' .gitignore → match 확인
      2. grep '\.env' .gitignore → match 확인
    Expected Result: 새 항목 포함됨
    Evidence: .sisyphus/evidence/T7-gitignore-updated.txt
  ```

  **Commit**: YES
  - Message: `chore: remove .bak files and update .gitignore`
  - Files: `.gitignore` (삭제: `Containerfile.bak`, `pyproject.toml.bak`)

---

- [x] T8. README 리포 URL 수정

  **What to do**:
  - `README.md` 라인 32: `https://github.com/vllm-project/vllm-openvino.git` → `https://github.com/belonghim/vllm-openvino.git` (실제 리포 URL)

  **Must NOT do**:
  - README 구조 변경 금지 — URL만 수정

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T5-T7, T9)
  - **Blocks**: T10
  - **Blocked By**: T1-T4

  **References**:
  - `README.md:32` — 잘못된 URL
  - `git remote -v` → `origin https://github.com/belonghim/vllm-openvino` (실제 remote)

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: URL 수정 확인
    Tool: Bash (grep)
    Steps:
      1. grep 'belonghim' README.md → match 확인
    Expected Result: 올바른 리포 URL 포함
    Evidence: .sisyphus/evidence/T8-readme-url.txt
  ```

  **Commit**: YES (groups with T7)

---

- [x] T9. pyproject.toml 메타데이터 보강

  **What to do**:
  - `pyproject.toml`에 누락된 메타데이터 추가:
    - `authors = [{name = "belonghim"}]`
    - `classifiers` 추가 (Programming Language :: Python :: 3, License :: OSI Approved :: Apache Software License 등)
    - `[project.urls]` 섹션 추가 (Homepage, Repository)
  - 주석처리된 `setuptools_scm` 섹션 삭제

  **Must NOT do**:
  - 의존성 변경 금지
  - entry_points 변경 금지

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with T5-T8)
  - **Blocks**: T10
  - **Blocked By**: T1-T4

  **References**:
  - `pyproject.toml` — 현재 상태

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: 메타데이터 추가 확인
    Tool: Bash (grep)
    Steps:
      1. grep 'authors' pyproject.toml → match
      2. grep 'project.urls' pyproject.toml → match
    Expected Result: 메타데이터 포함됨
    Evidence: .sisyphus/evidence/T9-metadata.txt
  ```

  **Commit**: YES
  - Message: `chore: add project metadata to pyproject.toml`
  - Files: `pyproject.toml`

---

### Wave 4: Container 보안 (1 task)

- [x] T10. Containerfile non-root user + EXPOSE

  **What to do**:
  - `Containerfile` 런타임 스테이지에:
    - `RUN useradd -r -s /sbin/nologin vllm` 추가 (non-root 사용자 생성)
    - `USER vllm` 추가 (런타임 사용자 전환)
    - `EXPOSE 8000` 추가 (API 서버 포트 문서화)
  - 기존 `chgrp -R 0 . && chmod -R g+rwX .`는 OpenShift 호환성을 위해 유지

  **Must NOT do**:
  - Builder 스테이지 변경 금지
  - 기존 ENV 변수 변경 금지

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (단독)
  - **Blocks**: F1-F4
  - **Blocked By**: T5-T9

  **References**:
  - `Containerfile` — 현재 런타임 스테이지 (라인 15-24)
  - `30_RESOURCES/runbooks/AI/vllm-optimizer-runbook.md` — 기존 컨테이너 운영 참조

  **Acceptance Criteria**:

  **QA Scenarios (MANDATORY):**

  ```
  Scenario: Non-root 사용자 확인
    Tool: Bash (grep)
    Steps:
      1. grep -n 'USER' Containerfile → match 확인
      2. grep -n 'EXPOSE' Containerfile → match 확인
    Expected Result: USER와 EXPOSE directive 존재
    Evidence: .sisyphus/evidence/T10-container-security.txt

  Scenario: 구문 유효성 (Containerfile lint)
    Tool: Bash
    Steps:
      1. python3 -c "open('Containerfile').read()" → 파일 읽기 가능
    Expected Result: 파일 유효
    Evidence: .sisyphus/evidence/T10-containerfile-valid.txt
  ```

  **Commit**: YES
  - Message: `security: add non-root user and EXPOSE directive to Containerfile`
  - Files: `Containerfile`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection → fix → re-run.

- [x] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, grep). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [x] F2. **Code Quality Review** — `unspecified-high`
  Run `python3 -m py_compile` on all modified .py files. Review all changed files for: bare except, empty except blocks, print statements, commented-out code blocks (>3 lines). Verify no new `# type: ignore` added. Check that commit messages follow conventional format.
  Output: `Compile [PASS/FAIL] | Files [N clean/N issues] | VERDICT`

- [x] F3. **Vault Notes QA** — `unspecified-high`
  Read each of 5 vault notes via obsidian_read_note. Verify: frontmatter exists (title, date, tags), content length > 50 lines, required sections present per task spec, cross-references to other vault notes present, no broken links. Verify Korean language for body text.
  Output: `Notes [N/5 valid] | Frontmatter [N/5] | Cross-refs [N/5] | VERDICT`

- [x] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was done, nothing beyond spec was done. Check "Must NOT do" compliance. Detect cross-task contamination. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | VERDICT`

---

## Commit Strategy

| # | Message | Files | Pre-commit |
|---|---------|-------|-----------|
| 1 | `fix: correct GPU KV cache allocation crash` | `kv_cache.py` | `py_compile` |
| 2 | `fix: remove dead stub, fix log ordering and format double-wrap` | `openvino_worker_v1.py` | `py_compile` |
| 3 | `fix: correct log typos + remove unused imports` | `platform.py` | `py_compile` |
| 4 | `refactor: simplify KV cache precision + clean up comments` | `platform.py` | `py_compile` |
| 5 | `chore: remove .bak files, update .gitignore, fix README URL` | `.gitignore`, `README.md` | — |
| 6 | `chore: add project metadata to pyproject.toml` | `pyproject.toml` | — |
| 7 | `security: add non-root user and EXPOSE to Containerfile` | `Containerfile` | — |

---

## Success Criteria

### Verification Commands
```bash
# GPU crash bug fixed
grep -n '{{}}' vllm_openvino/kv_cache.py  # Expected: no matches

# Dead stub fixed
grep -n 'self.kv_cache_config.num_blocks' vllm_openvino/worker_v1/openvino_worker_v1.py  # Expected: no matches

# Typos fixed
grep -n '"bp16"' vllm_openvino/platform.py  # Expected: no matches
grep -n 'OpenViNO' vllm_openvino/platform.py  # Expected: no matches

# Unused imports removed
grep -n 'vllm_envs' vllm_openvino/platform.py  # Expected: no matches
grep -n 'as hints' vllm_openvino/platform.py  # Expected: no matches

# .bak files removed
ls *.bak 2>/dev/null  # Expected: no files

# Container security
grep -n 'USER' Containerfile  # Expected: match
grep -n 'EXPOSE' Containerfile  # Expected: match

# All Python files compile
python3 -m py_compile vllm_openvino/platform.py
python3 -m py_compile vllm_openvino/kv_cache.py
python3 -m py_compile vllm_openvino/worker_v1/openvino_worker_v1.py
```

### Final Checklist
- [ ] 5개 vault 노트 Obsidian에 작성 완료
- [ ] GPU crash 버그 수정됨
- [ ] Dead stub 수정됨
- [ ] 로그 오타 3개 수정됨
- [ ] 미사용 import 2개 제거됨
- [ ] .bak 파일 2개 삭제됨
- [ ] .gitignore 업데이트됨
- [ ] README URL 수정됨
- [ ] pyproject.toml 메타데이터 보강됨
- [ ] Containerfile non-root user 추가됨
- [ ] 모든 수정 파일 py_compile PASS

---

## Deferred Items (이 계획 범위 밖 — 별도 계획 필요)

| Item | 이유 | 우선순위 |
|------|------|---------|
| PlatformEnum.OPENVINO 전환 | vLLM 0.13.0에 존재 여부 연구 필요 | High |
| InputBatch 아키텍처 리팩토링 | Worker↔ModelRunner 책임 분리 — 대규모 변경 | Medium |
| kv_cache.py clone/grow/get_slot_kv_cache 구현 | 인터페이스 요구사항 미확인 | Medium |
| openvino._offline_transformations 대체 | public API 연구 필요 | Medium |
| 테스트 인프라 추가 | pytest + smoke test — 별도 계획 | High |
| CI/CD 파이프라인 | GitHub Actions — 별도 계획 | High |
| docstring 전면 추가 | 모든 공개 메소드 — 별도 계획 | Low |
| vLLM 버전 업그레이드 | 0.13.0 → 최신 — 대규모 마이그레이션 | Low |