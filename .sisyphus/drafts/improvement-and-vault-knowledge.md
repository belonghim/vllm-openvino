# Draft: vllm-openvino 개선점 분석 + Vault 지식 정리 계획

## Requirements (confirmed)
- 프로젝트 개선점을 철저히 찾아내기 (코드 품질, 아키텍처, 빌드, 테스트, 문서)
- 발견한 지식을 Obsidian vault에 재사용 가능하도록 정리할 계획 수립

## 분석 완료 상태
- [x] 전체 소스코드 14개 .py 파일 직접 읽기 완료
- [x] grep/ast-grep 패턴 검색 (TODO, FIXME, NotImplementedError, unused imports 등)
- [x] Librarian: vLLM 플러그인 시스템 + OpenVINO 최적화 리서치
- [x] Explore: Containerfile/빌드 분석
- [x] Vault 구조 분석 (172 notes, PARA 방법론)

---

## 발견된 개선점 요약 (6개 카테고리, 40+ 항목)

### A. 코드 품질 이슈 (14개)
1. platform.py에 불확실한 주석 5개 (L10, L29, L32, L154, L155)
2. platform.py 로그 메시지 오타 2개 (L108 "bp16"→"bf16", L113 "f16"→"f32")
3. platform.py KV cache precision 처리 25줄의 반복 if/elif → dict lookup으로 단순화
4. type: ignore 2개, noqa 4개
5. 하나의 TODO 남아있음 (openvino.py L226)
6. 광범위한 except Exception 1개 (openvino.py L94) — 에러 무시
7. 빈 pass 블록 2개
8. NotImplementedError 스텁 8개 (grow, get_slot_kv_cache, build, forward, LoRA 4개)
9. kv_cache.py clone() 미구현 (리소스 누수 위험 — 새 ov.Core() 생성)
10. str_to_type 매핑 중복 (kv_cache.py + worker_v1.py)
11. envs.py에 사용 여부 불명 import
12. 공개 메소드 대부분 docstring 없음
13. 하드코딩된 값 4곳 (block_size, KV cache 기본값, memory threshold)
14. kv_cache.py L49: assert device_type=="cpu" → GPU 사용 시 실패

### B. 아키텍처/설계 이슈 (6개)
1. PlatformEnum.CPU 사용 — OPENVINO가 아닌 CPU로 등록 (주석에 "Check!" 남김)
2. 17개+ 깊은 vLLM 내부 import — 모든 vLLM 업데이트에서 breakage 위험
3. Model Runner가 Worker의 책임을 가짐 (InputBatch, requests)
4. GPU 지원인데 device_type=="cpu" assertion 존재 (모순)
5. clone()에서 새 ov.Core() 생성 — 리소스 누수
6. Containerfile에서 vllm_openvino 이중 복사

### C. 빌드/패키징 이슈 (8개)
1. CI/CD 없음 — GitHub Actions, GitLab CI 없음
2. 테스트 인프라 제로 — 테스트 파일 없음
3. .bak 파일 커밋됨 (Containerfile.bak, pyproject.toml.bak)
4. Git 의존성 (optimum-intel VCS) — 재현성 문제
5. README 리포 URL 불일치 (vllm-project vs belonghim)
6. pyproject.toml 메타데이터 부족 (authors, classifiers, urls)
7. Containerfile root 사용자 — 보안 취약
8. .gitignore 불완전 (.env, .venv, .mypy_cache 등 누락)

### D. 테스트
- 테스트 파일 0개, 테스트 프레임워크 없음

### E. 문서
- migration_v0_to_v1.md 외 개발자 가이드 없음
- API 문서, 아키텍처 다이어그램, 기여 가이드 없음

### F. 업스트림 호환성 위험
- vLLM == 0.13.0 정확히 고정 — 어떤 업데이트도 플러그인 수정 필요
- 17개+ 내부 vLLM import path — 모두 breakage 포인트
- 호환성 테스트 메커니즘 없음

---

## Vault 지식 정리 (제안 구조)

### 기존 관련 노트
- 20_AREAS/AI-Infrastructure/vllm-openvino-containerfile-guide.md (이미 존재)
- 30_RESOURCES/runbooks/AI/vllm-optimizer-runbook.md (이미 존재)
- 10_PROJECTS/2026-02_vllm-optimizer/ (프로젝트 데이터)

### 제안 노트 5개
1. `20_AREAS/AI-Infrastructure/vllm-openvino-codebase-analysis.md`
   — 코드베이스 전체 분석, 개선점 맵

2. `30_RESOURCES/runbooks/AI/vllm-openvino-development-runbook.md`
   — 빌드/테스트/배포/핫패치 운영 지식

3. `40_KNOWLEDGE/RedHat/vllm-plugin-architecture-guide.md`
   — vLLM 플러그인 시스템 일반 지식 (V1 아키텍처, 등록 패턴)

4. `40_KNOWLEDGE/RedHat/openvino-llm-optimization-guide.md`
   — OpenVINO LLM 최적화 패턴

5. `20_AREAS/AI-Infrastructure/vllm-openvino-upstream-compatibility.md`
   — import 매핑, 호환성 위험, 업그레이드 전략

---

## Agent Research Findings (추가)

### Librarian: vLLM Plugin System
- vLLM 공식 플러그인: entry_points "vllm.platform_plugins" 패턴 사용 → 현재 프로젝트 올바르게 구현
- V1 아키텍처: WorkerBase → ModelRunnerBase 분리 — 현재 프로젝트는 ModelRunnerV1에 InputBatch 직접 관리 (아키텍처 불일치)
- 커뮤니티: TPU, Neuron 등 다른 백엔드도 같은 패턴 사용
- OpenVINO 최적화: Weight Compression (4/8bit), PagedAttention, Dynamic Split-Fuse 
- 벤치마크: vllm.benchmarks.serve 사용 가능 — 현재 프로젝트에 벤치마크 스크립트 없음

### Explore: Architecture Analysis
- 17개+ vLLM 내부 import — 모두 fragile breakage points
- Coupling hotspots 3개: openvino_worker_v1.py, model_loader/openvino.py, attention/backends/openvino.py
- 동적 attribute 접근: envs.py __getattr__ — 런타임 에러 마스킹 가능
- 클래스 계층: OpenVinoPlatform(Platform), OpenVINOWorkerV1(WorkerBase), OpenVINOAttentionBackend(AttentionBackend), OpenVINOAttentionImpl(AttentionImpl)
- 권장: vLLM v1 인터페이스 주위에 adapter/bridge 레이어 추가

### Explore: Build/Deploy Analysis
- Containerfile root 사용자 → 보안 취약점 (HIGH)
- EXPOSE/HEALTHCHECK 없음 (MEDIUM)
- CI/CD 없음 (HIGH)
- 테스트 인프라 없음 (HIGH)
- Git 의존성 → 재현성 문제 (MEDIUM)
- .bak 파일 커밋됨 (LOW)

---

## Open Questions
1. vault 노트 배치 (위 구조 동의 여부)
2. 개선 작업 계획도 필요한지 (코드 수정 work plan도 따로?)
3. 노트 언어 선호 (한국어/영어/혼합)
