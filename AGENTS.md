# AGENTS.md — vllm-openvino

## 프로젝트 정체성

**vllm-openvino**는 [vLLM](https://github.com/vllm-project/vllm)의 **플러그인**으로, Intel OpenVINO(https://github.com/openvinotoolkit/openvino)를 LLM 추론 백엔드로 추가합니다.

**구현 참조**: [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)의 추론 로직(상태 관리, SDPA 백엔드, 하이브리드 모델 처리)을 참고하여, 동일 OpenVINO IR 모델을 vLLM OpenAI-compatible API로 서빙하는 것이 핵심 목표다.

- **vLLM 버전**: 0.32.0rc0
- **OpenVINO 버전**: >= 2026.4.1
- **플러그인 등록**: `pyproject.toml`의 `[project.entry-points."vllm.platform_plugins"]`

### 설계 원칙

1. **upstream 패턴 추종** — vLLM 구현 패턴을 정확히 따른다. upstream과 다른 아키텍처는 버전 업그레이드 시 호환성을 깨뜨린다.
2. **플러그인 경계 준수** — vLLM core를 수정하거나, core의 동작을 가정하지 않는다. vLLM이 제공하는 플러그인 인터페이스만 사용한다.
3. **규모에 맞는 최소주의** — 현재 규모를 초과하는 복잡도와 추상화, 별도 테스트 인프라를 추가하지 않는다.
4. **현재 문제만 해결** — 실제 사용자 수요가 없거나 vLLM 아키텍처 제약으로 구현 불가능한 기능은 추가하지 않는다.

**원칙 충돌 시 우선순위**: upstream 호환성 > 플러그인 경계 > 최소주의

Stateful 경로는 유지보수 대상으로만 유지하고, hybrid 모델의 확장은 Hybrid-PA 경로에서 한다.

### 문서 유지 원칙

- 개정 이력과 불필요한 작업 메타데이터는 남기지 않는다.
- 잘못된 정보는 삭제한다. 현재 판단에 필요한 상태, 제약, 검증 결과만 간결하게 기록한다.
- 측정값과 구현 세부는 `docs/compatibility.md`와 README에 둔다. 이 파일에는 규칙과 그 이유만 둔다.
- 시작시 git pull 하고, 완료시 git push 까지 진행한다.

## 주요 파일

| 파일 | 역할 |
|------|------|
| `platform.py` | OpenVinoPlatform — vLLM Platform 인터페이스, 설정 검증, 시작 전 KV 풀 추정 |
| `envs.py` | 환경변수 정의 (VLLM_OPENVINO_*) |
| `utils.py` | cgroup CPU/메모리 감지, RSS 측정, 가중치 크기, `image_input_enabled` |
| `kv_cache.py` | OpenVINOCacheEngine — KV/SSM/conv 캐시 할당 |
| `attention/backends/openvino.py` | OpenVINO Attention 백엔드 |
| `model_executor/model_loader/openvino.py` | 모델 로딩, PagedAttention·gather-before-matmul 변환, 비전 IR |
| `worker_v1/openvino_worker_v1.py` | 워커 — warmup, 로드 후 KV 풀 산정, 캐시 할당 |
| `worker_v1/openvino_model_runner_v1.py` | ModelRunner — 입력 준비, 추론, 샘플링 |
| `sampler_patch.py` | CPU용 샘플러 fast path (`VLLM_OPENVINO_FAST_SAMPLER`) |

## 검증

- 코드를 수정하면 먼저 `python3 -m py_compile`을 실행한다.
- 런타임 변경은 빌드 없이 podman 소스 마운트로 검증한다. `pyproject.toml`이나 `Containerfile`을 바꾸면 `podman build`까지 돌린다.
- 단일·연속·동시 요청을 실제 API로 확인한다. 요청은 `localhost`가 아니라 `127.0.0.1`로 보낸다(IPv6로 풀리면 연결이 끊긴다).
- 성능 비교는 `--cpus=8 --memory=16g`로 CPU 수와 메모리를 고정한다.
- 상세 절차는 `docs/podman-testing.md`를 따른다.

```bash
python3 -m py_compile <file>

podman run --replace -d --name vllm-server -p 8080:8080 --cpus=8 --memory=16g \
  -v ~/prj/vllm-openvino/vllm_openvino:/opt/app-root/vllm_openvino:Z \
  -v ~/hf:/models:Z \
  quay.io/joopark/vllm-openvino \
  --port=8080 --model <model_dir> --max-model-len 4096
```

## 깨지기 쉬운 규칙

상세 설명과 측정값은 `docs/compatibility.md` 참조.

- **`TORCH_COMPILE_DISABLE=1` 필수** — torch.compile/Inductor는 OpenVINO와 비호환.
- **`import openvino` 실패를 import 시점에 raise하지 않는다** — vLLM 플러그인 디스커버리가 다른 플랫폼을 검사하다 깨진다.
- **`_allocate_kv_cache()`에 `.fill(0)`을 넣지 않는다** — OpenVINO가 초기화하며, 추가하면 OOM. SSM/conv 캐시(`_allocate_state_cache()`)의 zero-fill은 유지한다.
- **bf16은 numpy 변환 전에 float32로** (`_as_numpy_no_copy()`) — OpenVINO는 bf16 numpy를 받지 않는다.
- **`execute_model`과 `sample_tokens`를 합치지 않는다** — `execute_model`은 logits까지만 계산하고, `sample_tokens(grammar_output)`이 grammar bitmask를 적용한 뒤 샘플링한다(upstream MRV1 패턴). 구조화 출력이 이 경계에 의존한다.
- **PA 전용 변환·입력은 stateful 모델에 주지 않는다** — `paged_attention_transformation()`, gather-before-matmul, PA 입력. gather 때문에 `prompt_logprobs`는 미지원이며 `validate_request`가 400으로 거부한다.
- **서빙 경로는 IR로 판정한다** — `ReadValue`의 `variable_id`에 `ssm`/`conv`가 있으면 Hybrid-PA, 그 외 SDPA 모델은 PA(Gemma-4 포함), SDPA가 없으면 stateful. 단순한 `ReadValue`+SDPA 조합으로 판정하지 않는다.
- **KV 풀은 VmRSS가 아니라 EngineCore anon RSS로 산정한다** — VmRSS에는 회수 가능한 mmap 가중치가 들어 있다. 측정 전 warmup이 실제 최대 prefill(+더미 이미지)을 돌려야 baseline이 맞다. warmup을 줄이면 풀이 커져 가득 찼을 때 OOM이 난다.
- **Gemma-4 PA 입력은 rank를 유지해 넣는다** — `per_layer_inputs` `[tokens, 1, layers, hidden]`, `token_type_ids` `[tokens, 1]`. 빠지면 sliding-window 레이어에서 `Eltwise ... dim index: 0 mismatch`.
- **비전 merger에는 `hidden_states`, `attention_mask`, `rotary_pos_emb` 세 입력을 모두 넘긴다** — 하나만 넘기면 SDPA shape 불일치.
- **HuggingFace 원본은 로딩하지 않는다** — OpenVINO IR만 지원한다. IR 파일명과 모델별 mm 입력 키는 `docs/compatibility.md`의 Model Files 참조.
- **미지원**: Pin memory, LoRA, TP·PP. 멀티소켓은 이득이 작으니 한 소켓 cpuset에 바인딩한다(README CPU Tuning).
