# AGENTS.md — vllm-openvino

## 프로젝트 정체성

**vllm-openvino**는 [vLLM](https://github.com/vllm-project/vllm)의 **플러그인**으로, Intel OpenVINO(https://github.com/openvinotoolkit/openvino)를 LLM 추론 백엔드로 추가합니다.

- **vLLM 버전**: 0.24.0
- **OpenVINO 버전**: >= 2026.2.1
- **플러그인 등록**: `pyproject.toml`의 `[project.entry-points."vllm.platform_plugins"]`
- **단일 개발자 프로젝트** (belonghim)

### 설계 원칙

1. **upstream 패턴 추종** — vLLM 구현 패턴을 정확히 따른다. upstream과 다른 아키텍처는 버전 업그레이드 시 호환성을 깨뜨린다.
2. **플러그인 경계 준수** — vLLM core를 수정하거나, core의 동작을 가정하지 않는다. vLLM이 제공하는 플러그인 인터페이스만 사용한다.
3. **규모에 맞는 최소주의** — 단일 개발자, 14파일 프로젝트다. 현재 규모를 초과하는 복잡도(테스트 인프라, CI/CD, 추상화 레이어)는 유지 비용이 이점을 넘는다.
4. **현재 문제만 해결** — 실제 사용자 수요가 없거나 vLLM 아키텍처 제약으로 구현 불가능한 기능은 추가하지 않는다.

**원칙 충돌 시 우선순위**: upstream 호환성 > 플러그인 경계 > 최소주의

## 디렉토리 구조

| 파일 | 역할 |
|------|------|
| `platform.py` | OpenVinoPlatform — vLLM Platform 인터페이스 |
| `envs.py` | 환경변수 정의 (VLLM_OPENVINO_*) |
| `kv_cache.py` | OpenVINOCacheEngine — KV 캐시 관리 |
| `attention/backends/openvino.py` | OpenVINO Attention 백엔드 |
| `model_executor/model_loader/openvino.py` | 모델 로딩 + PagedAttention 변환 |
| `worker_v1/openvino_worker_v1.py` | 워커 — KV 캐시 할당, 프로파일링 |
| `worker_v1/openvino_model_runner_v1.py` | ModelRunner — 입력 준비 + 추론 실행 |

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
  -v /home/jooan/prj/vllm-openvino/vllm_openvino:/opt/app-root/vllm_openvino:Z \
  -v <hf_models_dir>:/models:Z \
  quay.io/joopark/vllm-openvino \
  --port=8080 --model <model_dir> --max-model-len 4096
```

## 하지 말아야 할 것들 (요약)

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
| stateful path 기반 신규 기능 추가 | 금지 | OpenVINO Model Server(OVMS)에서 stateful model serving deprecated 예정. Runtime ReadValue/Assign ops 자체는 유지되나 장기 방향성 불확실. stateful path는 유지보수 모드로만 유지. 버그 수정은 허용 |

## 기술적 특이사항 (코드 수정 시 참고)

> 전체 목록과 상세 설명은 `docs/compatibility.md` 참조.

- **`TORCH_COMPILE_DISABLE=1` 필수** — torch.compile/Inductor가 OpenVINO와 비호환
- **bf16 → float32 변환** — `_as_numpy_no_copy()`에서 torch bfloat16을 numpy float32로 캐스팅. OpenVINO는 bf16 numpy 미지원
- **KV 캐시 `.fill(0)` 복원 금지** — `_allocate_kv_cache()`에 `.fill(0)` 없는 것이 정상(의도된 최적화). 추가하면 OOM 유발. SSM/conv 캐시(`_allocate_state_cache()`)에는 `.fill(0)` 유지됨
- **Pin memory 미지원** / **LoRA 미지원** / **단일 소켓만 지원**
- **OpenVINO import 실패 처리** — `platform.py`에서 `import openvino` 실패 시 import 시점에 raise하지 말 것. vLLM 플러그인 디스커버리 메커니즘 때문
- **서빙 경로 2가지** — `detect_model_type()`으로 탐지 (기준: `ReadValue` op 유무):
  - **PagedAttention path**: `ReadValue` op 없는 모델(ATTENTION_ONLY) 중 `ScaledDotProductAttention` op도 있는 모델 (Llama 3, Qwen2.5 등). PA 변환 적용, 동시 요청 배칭 가능.
  - **Stateful path**: `ReadValue` op 있는 모델 (Gemma-4) 또는 hybrid Mamba/attention (`ReadValue`에 ssm/conv var_id 포함, Qwen3.5). `max_num_seqs=1`, 순차 처리, OpenVINO 내부 KV 캐시.
- **Gather-before-matmul 변환 주의** — PA-transformed 모델에만 적용. stateful 모델에 적용하면 `seq_len=0` 출력으로 서빙 실패
- **Multi-request batching for stateful models** — `forward()`의 `num_requests` 파라미터로 실제 요청 수만큼 슬라이싱
- **모델 포맷 필수** — OpenVINO IR 포맷 사전 변환 필요. HuggingFace 원본 모델 직접 로딩 불가. 파일명 규칙:
  - 텍스트 모델: `openvino_model.xml`
  - 멀티모달 언어 모델: `openvino_language_model.xml`
  - 텍스트 임베딩: `openvino_text_embeddings_model.xml`
  - 비전 인코더: `openvino_vision_embeddings_model.xml` (입력: **2D** `[num_patches, features]`, 배치 dim 없음 — 필요 시 squeeze)
  - 비전 merger (일부 모델): `openvino_vision_embeddings_merger_model.xml` (어텐션 기반, 3개 입력 필수)
  - 비전 pos (일부 모델): `openvino_vision_embeddings_pos_model.xml` (입력 `[4, ?] i64` → 출력 `[4, ?, 1024]`, 현재 미사용)

### 검증된 모델 및 서빙 경로 (2026-07-17 기준)

| 모델 | 서빙 경로 | 비전 | 비고 |
|------|----------|------|------|
| Qwen2.5-Coder-3B-Instruct-int4-ov | PA | ❌ | 기본 PA 경로 |
| Qwen3.5-2B-int4-ov | Stateful/Hybrid | ✅ merger 필요 | ssm+conv ReadValue |
| gemma-4-E2B-it-int4-ov | Stateful | ✅ | 단순 vision emb |

## 멀티모달(비전) 서빙 파이프라인

### 처리 흐름

```
mm_item["pixel_values"]       → vision encoder → [num_patches, 1024]
mm_item["image_grid_thw"]     → _compute_merger_rotary_pos_emb() → [num_patches, 32]
                                np.ones([1, N, N])                → attention_mask
                              → merger.infer({hidden_states, attention_mask, rotary_pos_emb})
                              → [num_patches, text_embed_dim]  (text dim으로 projection됨)
```

### Merger 호출 규칙

merger(`openvino_vision_embeddings_merger_model.xml`)는 반드시 **dict로 3개 입력** 전달:

```python
self.vision_merger_request.infer({
    "hidden_states": vision_embeds_2d,           # [num_patches, 1024]
    "attention_mask": np.ones((1, N, N), np.float32),  # full attention
    "rotary_pos_emb": rotary_pos_emb,            # [num_patches, 32]
})
```

1개만 넘기면 `ScaledDotProductAttention` 내부에서 shape 불일치로 실패.

### rotary_pos_emb 계산

`image_grid_thw [n_images, 3]` (T, H, W)에서 Qwen2.5-VL `rot_pos_emb`와 동일 로직으로 계산:
- spatial_merge_size=2 기준 permuted h/w pos ids 생성
- rope_dim_per_spatial=16, theta=10000으로 frequency table 조회
- concat(h_rope, w_rope) → [num_patches, 32]

구현: `OpenVINOCausalLM._compute_merger_rotary_pos_emb()`

### mm_item 키 참조

| 모델 | pixel_values | pixel_position_ids | image_grid_thw |
|------|--------------|--------------------|----------------|
| Qwen2.5-Coder (PA, 텍스트 전용) | — | — | — |
| Qwen3.5-2B (Hybrid) | ✅ | ❌ 없음 | ✅ |
| Gemma-4-E2B (Stateful) | ✅ | ❌ 없음 | ❌ |

`pixel_position_ids`가 없을 경우 `image_grid_thw`에서 직접 계산. 두 가지 모두 없으면 rotary_pos_emb를 zeros로 fallback.

### pos model (현재 미사용)

`openvino_vision_embeddings_pos_model.xml`: `[4, num_merged] i64 → [4, num_merged, 1024] f32` (embedding table lookup). merger 품질 개선에 기여할 수 있으나, 현재 skip해도 inference가 정상 동작함. 추가 시 패치 ordering(raster vs spatial-merge) 정렬 필요.

## Git 및 Commits 정책

- **Co-authored-by 미사용**: AI agent는 commits를 할 때 `Co-authored-by` trailer를 추가하지 않습니다. 모든 commits은 belonghim 계정의 이름으로만 기록됩니다.
- **사유**: GitHub contributor 목록의 명확성을 위해. 실제 코드 개발은 사용자(belonghim)이고, AI agent는 개발 보조 도구입니다.
- 완료시 git push 까지 진행한다

## vLLM 버전 호환성

| 버전 | 플러그인 호환성 | 비고 |
|------|---------------|------|
| **v0.24.0** (2026-06-29) | ✅ 현재 타겟 | AGENTS.md 기준 |
| **v0.25.0** (2026-07-11) | ✅ **호환됨** (소스 레벨) | 모든 plugin 인터페이스 동일: WorkerBase, ModelRunnerOutput, SchedulerOutput, AttentionBackend, KVCacheSpec |
| **v0.25.1** (2026-07-14) | ✅ 호환됨 | TorchCodec import, mixed-dtype allreduce RMSNorm 패치만 포함 |

### v0.24.0 → v0.25.0 인터페이스 diff 요약

| 인터페이스 | 파일 | Diff 결과 |
|-----------|------|-----------|
| `WorkerBase` | `worker_base.py` | **변경 없음** — 생성자, `__init__`, `init_device`, `execute_model()`, `sample_tokens()` 모두 동일 |
| `ModelRunnerOutput` | `outputs.py` | **변경 없음** — 모든 dataclass 필드 동일 |
| `SchedulerOutput` / `NewRequestData` / `CachedRequestData` | `core/sched/output.py` | **변경 없음** — 모든 dataclass 필드 동일 |
| `KVCacheSpec` / `FullAttentionSpec` / `AttentionSpec` | `kv_cache_interface.py` | **하위 호환** — `KVQuantMode.INT4_PER_TOKEN_HEAD` (새 값), `RSWASpec` (새 dataclass). 기존 `FullAttentionSpec` 필드 불변 |
| `AttentionBackend` | `attention/backend.py` | **하위 호환** — `rswa_prefix_lens` 필드, `lse_base_on_e` 속성, `token_to_req_indices()` 메서드 추가. 새 추상 메서드 없음 |

**결론**: v0.25.0으로 업그레이드해도 plugin 코드 수정 불필요. 현재 AGENTS.md의 v0.24.0 타겟 유지 또는 v0.25.0으로 업데이트 모두 가능.
