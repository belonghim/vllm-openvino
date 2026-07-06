# AGENTS.md — vllm-openvino

## 프로젝트 정체성

**vllm-openvino**는 [vLLM](https://github.com/vllm-project/vllm)의 **플러그인**으로, Intel OpenVINO를 LLM 추론 백엔드로 추가합니다.

- **vLLM 버전**: 0.24.0
- **OpenVINO 버전**: >= 2026.2.1
- **플러그인 등록**: `pyproject.toml`의 `[project.entry-points."vllm.platform_plugins"]`
- **단일 개발자 프로젝트** (belonghim)

### 설계 원칙

1. **upstream 패턴 추종** — vLLM 낱부 구현 패턴을 정확히 따른다. upstream과 다른 아키텍처는 버전 업그레이드 시 호환성을 깨뜨린다.
2. **플러그인 경계 준수** — vLLM core를 수정하거나, core의 낱부 동작을 가정하지 않는다. vLLM이 제공하는 플러그인 인터페이스만 사용한다.
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
├── attention/
│   ├── __init__.py
│   └── backends/
│       ├── __init__.py
│       └── openvino.py                  # OpenVINO Attention 백엔드
├── model_executor/
│   ├── __init__.py
│   └── model_loader/
│       ├── __init__.py
│       └── openvino.py                  # 모델 로딩 + PagedAttention 변환
└── worker_v1/
    ├── __init__.py
    ├── openvino_worker_v1.py            # OpenVINOWorkerV1 — 워커 (KV 캐시 할당, 프로파일링)
    └── openvino_model_runner_v1.py      # ModelRunner — 입력 준비 + 추론 실행
scripts/benchmark.sh                     # 성능 벤치마크 스크립트
scripts/smoke-test.sh                    # 회귀 테스트 스모크 테스트
scripts/guidellm-benchmark.sh            # guidellm 기반 부하 벤치마크 스크립트
```

## 테스트 방법

실제 개발 및 검증은 **빌드 없이 podman 소스 마운트**로 진행합니다. 로컬 `pip install`은 거의 사용되지 않습니다.

### 로컬 검증 우선 원칙 (필수)

- **에이전트는 사용자에게 "배포해주세요"를 요청하기 전, 반드시 로컬 podman에서 직접 검증해야 한다.**
- 코드 수정 → `python3 -m py_compile` → podman 소스 마운트 → API 호출 테스트 → 정상 응답 확인 후에만 사용자에게 클러스터 배포를 요청한다.
- podman 테스트 시 `--enable-auto-tool-choice --tool-call-parser=qwen3_coder` 등 실제 실행 인자를 그대로 사용한다.
- 단일 요청, 연속 요청, 동시 요청 모두 통과해야 한다.
- 상세 가이드 및 반복 디버그 루프: `docs/podman-testing.md`

```bash
# 문법 오류 사전 차단 (반드시 먼저 실행)
python3 -m py_compile <file>

# podman 소스 마운트 테스트
podman run --replace -d --name vllm-server -p 8080:8080 \
  -v /home/user/project/vllm-openvino/vllm_openvino:/opt/app-root/vllm_openvino \
  -v /home/user/hf:/models:Z \
  quay.io/joopark/vllm-openvino \
  --port=8080 --model /models/<model_dir> --max-model-len 4096
```

## ⛔ 하지 말아야 할 것들 (요약)

> 상세 설명은 `docs/decisions.md` 참조. 아래 항목들은 이미 평가되어 **불필요 또는 시기상조**로 판명됨.

| 항목 | 판정 | 핵심 이유 |
|------|------|-----------|
| PlatformEnum.OPENVINO 전환 | 불필요 | enum은 식별자일 뿐 분기 로직 아님. upstream PR 필요 |
| InputBatch 아키텍처 리팩토링 | 해로움 | upstream GPUWorkerV1과 동일 패턴. 미래 호환성 저하 |
| 테스트 인프라 / CI/CD 추가 | 과도 | 단일 개발자, 14파일. OpenVINO 런타임 없이 테스트 불가한 핵심 버그들이 대부분 |
| `str_to_torch_type` / `str_to_ov_type` 통합 | 불필요 | 서로 다른 타입 시스템(PyTorch vs OpenVINO). 단일 모듈 통합 불가 |
| 비동기 추론 파이프라인 | 불가 | vLLM 스케줄러가 순차 동작. 플러그인 레벨 비동기는 구조적으로 불가 |
| Structured outputs (문법 유도 디코딩) | 수요 없음 | outlines 통합 필요. 단순히 `sample_tokens()` 수정으로는 불가 |
| `openvino._offline_transformations` 교체 | 불필요 | `paged_attention_transformation`은 대체 불가. 2026.0.0에서 여전히 정상 동작 확인됨 |

## 기술적 특이사항 (코드 수정 시 참고)

> 전체 목록과 상세 설명은 `docs/compatibility.md` 참조.

- **`TORCH_COMPILE_DISABLE=1` 필수** — torch.compile/Inductor가 OpenVINO와 비호환
- **bf16 → float32 변환** — `_as_numpy_no_copy()`에서 torch bfloat16을 numpy float32로 캐스팅. OpenVINO는 bf16 numpy 미지원
- **KV 캐시 `.fill(0)` 제거 금지** — `_allocate_kv_cache()`에서 `.fill(0)` 제거는 의도된 최적화. 되돌리면 OOM 유발
- **Pin memory 미지원** / **LoRA 미지원** / **단일 소켓만 지원**
- **OpenVINO import 실패 처리** — `platform.py`에서 `import openvino` 실패 시 import 시점에 raise하지 말 것. vLLM 플러그인 디스커버리 메커니즘 때문
- **SSM/Mamba 지원** — hybrid model(Attention + SSM) 지원. `detect_model_type()`으로 모델 타입 탐지, PA 변환은 attention-only 모델에만 적용
- **Gather-before-matmul 변환 주의** — PA-transformed 모델에만 적용. stateful 모델에 적용하면 `seq_len=0` 출력으로 서빙 실패
- **Multi-request batching for stateful models** — `forward()`의 `num_requests` 파라미터로 실제 요청 수만큼 슬라이싱

## Git 및 Commits 정책

- **Co-authored-by 미사용**: AI agent는 commits를 할 때 `Co-authored-by` trailer를 추가하지 않습니다. 모든 commits은 belonghim 계정의 이름으로만 기록됩니다.
- **사유**: GitHub contributor 목록의 명확성을 위해. 실제 코드 개발은 사용자(belonghim)이고, AI agent는 개발 보조 도구입니다.

## 참조 문서

| 문서 | 내용 |
|------|------|
| `docs/compatibility.md` | OpenVINO 2026.x 호환성 변경 이력, 기술적 특이사항 상세 |
| `docs/upgrade-checklist.md` | vLLM 버전 업그레이드 체크리스트, 핵심 의존 관계 상세표 |
| `docs/podman-testing.md` | 소스 마운트 테스트(podman) 상세 가이드, 반복 디버그 루프 |
| `docs/decisions.md` | 평가 및 폐기 결정 상세 기록, Refactoring History |
| `docs/migration_v0_to_v1.md` | V0 → V1 엔진 마이그레이션 노트 |
