# AGENTS.md — vllm-openvino

## 프로젝트 정체성

**vllm-openvino**는 [vLLM](https://github.com/vllm-project/vllm)의 **플러그인**으로, Intel OpenVINO(https://github.com/openvinotoolkit/openvino)를 LLM 추론 백엔드로 추가합니다.

**구현 참조**: [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)의 추론 로직(상태 관리, SDPA 백엔드, 하이브리드 모델 처리)을 참고하여, 동일 OpenVINO IR 모델을 vLLM OpenAI-compatible API로 서빙하는 것이 핵심 목표다.

- **vLLM 버전**: 0.26.0
- **OpenVINO 버전**: >= 2026.3.0
- **플러그인 등록**: `pyproject.toml`의 `[project.entry-points."vllm.platform_plugins"]`

### 설계 원칙

1. **upstream 패턴 추종** — vLLM 구현 패턴을 정확히 따른다. upstream과 다른 아키텍처는 버전 업그레이드 시 호환성을 깨뜨린다.
2. **플러그인 경계 준수** — vLLM core를 수정하거나, core의 동작을 가정하지 않는다. vLLM이 제공하는 플러그인 인터페이스만 사용한다.
3. **규모에 맞는 최소주의** — 현재 규모를 초과하는 복잡도와 추상화를 추가하지 않는다.
4. **현재 문제만 해결** — 실제 사용자 수요가 없거나 vLLM 아키텍처 제약으로 구현 불가능한 기능은 추가하지 않는다.

**원칙 충돌 시 우선순위**: upstream 호환성 > 플러그인 경계 > 최소주의

### 문서 유지 원칙

- 개정 이력과 불필요한 작업 메타데이터는 남기지 않는다.
- 잘못된 정보는 삭제한다. 현재 판단에 필요한 상태, 제약, 검증 결과만 간결하게 기록한다.
- 시작시 git pull 하고, 완료시 git push 까지 진행한다.

## 주요 파일

| 파일 | 역할 |
|------|------|
| `platform.py` | OpenVinoPlatform — vLLM Platform 인터페이스 |
| `envs.py` | 환경변수 정의 (VLLM_OPENVINO_*) |
| `kv_cache.py` | OpenVINOCacheEngine — KV 캐시 관리 |
| `attention/backends/openvino.py` | OpenVINO Attention 백엔드 |
| `model_executor/model_loader/openvino.py` | 모델 로딩 + PagedAttention 변환 |
| `worker_v1/openvino_worker_v1.py` | 워커 — KV 캐시 할당, 프로파일링 |
| `worker_v1/openvino_model_runner_v1.py` | ModelRunner — 입력 준비 + 추론 실행 |

## 검증

- 코드를 수정하면 먼저 `python3 -m py_compile`을 실행한다.
- 런타임 변경은 빌드 없이 podman 소스 마운트로 검증한다.
- 단일·연속·동시 요청을 실제 API로 확인한다.
- 성능 비교는 `--cpus=8 --memory=16g`로 CPU 수와 메모리를 고정한다.
- 상세 절차는 `docs/podman-testing.md`를 따른다.

```bash
# 문법 오류 사전 차단 (반드시 먼저 실행)
python3 -m py_compile <file>

# podman 소스 마운트 테스트 (CPU 8개, 메모리 16GiB로 제한)
podman run --replace -d --name vllm-server -p 8080:8080 --cpus=8 --memory=16g \
  -v ~/prj/vllm-openvino/vllm_openvino:/opt/app-root/vllm_openvino:Z \
  -v ~/hf:/models:Z \
  quay.io/joopark/vllm-openvino \
  --port=8080 --model <model_dir> --max-model-len 4096
```

## 범위 제한

- vLLM core를 수정하지 않는다.
- 현재 요구가 없는 기능, 별도 테스트 인프라, 추상화 레이어를 추가하지 않는다.
- Stateful 경로는 유지보수 대상으로 유지하고, hybrid 모델의 확장은 Hybrid-PA 경로로 분리한다.

## 기술적 특이사항 (코드 수정 시 참고)

> 전체 목록과 상세 설명은 `docs/compatibility.md` 참조.

- **`TORCH_COMPILE_DISABLE=1` 필수** — torch.compile/Inductor가 OpenVINO와 비호환
- **bf16 → float32 변환** — `_as_numpy_no_copy()`에서 torch bfloat16을 numpy float32로 캐스팅. OpenVINO는 bf16 numpy 미지원
- **KV 캐시 `.fill(0)` 복원 금지** — `_allocate_kv_cache()`에 `.fill(0)` 없는 것이 정상(의도된 최적화). 추가하면 OOM 유발. SSM/conv 캐시(`_allocate_state_cache()`)에는 `.fill(0)` 유지됨
- **Pin memory 미지원** / **LoRA 미지원** / **단일 소켓만 지원**
- **OpenVINO import 실패 처리** — `platform.py`에서 `import openvino` 실패 시 import 시점에 raise하지 말 것. vLLM 플러그인 디스커버리 메커니즘 때문
- **서빙 경로** — `ReadValue`가 없는 attention-only 모델은 PagedAttention을 사용한다. `ssm`/`conv` 상태가 있는 hybrid 모델(Qwen3.5, LFM2.5)은 Hybrid-PA를 기본 사용하며 `VLLM_OPENVINO_HYBRID_PA=0`으로 stateful 경로를 강제할 수 있다. 그 외 stateful 모델(Gemma-4)은 OpenVINO 내부 KV cache를 사용하고 `max_num_seqs=1`로 동작한다.
- **Hybrid-PA 후보 판정** — 단순한 `ReadValue`+SDPA 조합이 아니라 IR `variable_id`의 `ssm`/`conv` 상태를 확인한다. Gemma-4의 sliding-window attention은 Hybrid-PA 대상이 아니다.
- **CPU 스레드 자동감지** — `VLLM_OPENVINO_CPU_THREADS_NUM=0`이면 cgroup CPU quota를 고려해 OpenVINO 스레드 수를 제한한다. 명시적인 환경변수 설정이 우선한다.
- **Gather-before-matmul 변환** — PA-transformed 모델에만 적용한다. stateful 모델에는 적용하지 않는다.
- **SSM/conv cache** — stateful 경로는 OpenVINO가 내부 상태를 관리하므로 외부 SSM/conv cache를 할당하지 않는다. Hybrid-PA는 스케줄러 블록과 별도의 물리 slot pool을 사용한다.
- **모델 포맷 필수** — HuggingFace 원본 모델은 직접 로딩하지 않으며 OpenVINO IR로 사전 변환해야 한다. 주요 파일명:
  - 텍스트 모델: `openvino_model.xml`
  - 멀티모달 언어 모델: `openvino_language_model.xml`
  - 텍스트 임베딩: `openvino_text_embeddings_model.xml`
  - 비전 인코더: `openvino_vision_embeddings_model.xml` (입력: **2D** `[num_patches, features]`, 배치 dim 없음 — 필요 시 squeeze)
  - 비전 merger (일부 모델): `openvino_vision_embeddings_merger_model.xml` (어텐션 기반, **dict로 3개 입력 필수**: `hidden_states`, `attention_mask`, `rotary_pos_emb` — 1개만 넘기면 ScaledDotProductAttention shape 불일치로 실패)
- **mm_item 키 차이** — Qwen3.5는 `pixel_values` + `image_grid_thw`, Gemma-4는 `pixel_values`를 사용한다.
