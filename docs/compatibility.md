# OpenVINO Compatibility & Technical Specifics

## OpenVINO 2026.x 호환성 변경 이력

OpenVINO 2026.0.0으로 업그레이드 시 발생한 breaking change 및 대응 내역 (2026-03-09 완료).

| 변경 | 영향 파일 | 대응 |
|---|---|---|
| `ov.runtime.Coordinate` 삭제 (`ov.runtime` 모듈 전체 제거) | `attention/backends/openvino.py` | `ov.Coordinate`로 교체 (커밋 `0b6529a`) |
| `ov.Type.undefined` 삭제 | `model_executor/model_loader/openvino.py` | 해당 코드가 dead code(`_modify_cache_parameters()`)임을 확인 후 함수 전체 제거 (커밋 `49f9587`) |
| `paged_attention_transformation` (`_offline_transformations`) | `model_executor/model_loader/openvino.py` | 변경 없음 — private API 유지됨, 정상 동작 |
| `compile_model` 시 KV 캐시 처리 | — | 2026.0에서 플러그인이 자동 처리 → `_modify_cache_parameters()` 불필요 확인 |
| `enable_mmap` 설정 추가 | `worker_v1/openvino_worker_v1.py` | OpenVINO 2026.0+ 메모리 매핑 활성화를 위해 `ov.Core()`에 `enable_mmap: True` 설정 (커밋 `bffd8cc`) |
| CPU compile properties 추가 (AVX2 튜닝) | `worker_v1/openvino_worker_v1.py` | CPU 디바이스용 AFFINITY, BIND_THREAD, NUM_STREAMS, THREADS_NUM compile properties 추가 (커밋 `e80fea7`) |
| SSM/Mamba 지원 추가 | `model_executor/model_loader/openvino.py`, `worker_v1/openvino_worker_v1.py` | Hybrid model detection 및 selective PagedAttention transform 도입. `MambaSpec` import 및 SSM/conv cache 처리 추가 (커밋 `001a294`, `356b009`, `2701318`) |

> 향후 OpenVINO 버전 업그레이드 시 위 패턴을 참고할 것. 특히 `ov.runtime.*` 같은 하위 모듈 API는 삭제될 가능성이 있음.

---

## 알려진 기술적 특이사항

1. **`TORCH_COMPILE_DISABLE=1` 필수** — vLLM 0.19.1에서 torch.compile/Inductor가 OpenVINO와 비호환. 이 env var 없으면 크래시
2. **Pin memory 미지원** — `is_pin_memory_available()` → False. CPU/OpenVINO 환경에서는 pin memory 불필요
3. **LoRA 미지원** — `check_and_update_config()`에서 assert로 차단
4. **단일 소켓만 지원** — `parallel_config.world_size == 1` 강제. Tensor/Pipeline 병렬 미지원
5. **KV 캐시 블록 크기** — CPU: 32, GPU: 16 (자동 오버라이드)
6. **KServe modelcar 호환성** — modelcar 방식으로 배포 시 `/mnt/models`가 symlink로 제공됨. 로컬 pre-exported IR은 `ov_core.read_model()` 직접 로딩으로 처리. 최적화: 로컬 IR만 지원 (2026-04-21).
7. **OpenVINO import 실패 처리** — `platform.py`에서 `import openvino` 실패 시 `ov = None`으로 설정하고 warning만 출력. 실제 사용 시점인 `check_and_update_config()`에서 `ImportError`를 raise. **import 시점에서 raise하지 않는 이유**: vLLM 플러그인 디스커버리 메커니즘이 모든 플러그인을 import한 뒤 활성 플러그인을 선택하므로, import 시점 raise는 OpenVINO 플러그인이 아닌 다른 플러그인 사용 시에도 크래시를 유발함.
8. **VLLM_OPENVINO_PERFORMANCE_MODE** — 성능 모드 설정 (`LATENCY`/`THROUGHPUT`, 기본값: `LATENCY`). CPU 환경에서 TTFT(Time-To-First-Token) 개선에 유용.
9. **CPU-specific compile properties (AVX2 튜닝)** — CPU 디바이스 사용 시 `inference_num_threads`, `affinity`, `num_streams`, `enable_hyper_threading`, `inference_precision`, `enable_cpu_pinning` compile properties를 적용. `model_executor/model_loader/openvino.py`의 `OpenVINOCausalLM.__init__()` 인라인 처리. (`_get_cpu_compile_properties()` 함수는 존재하지 않음)
10. **Memory-mapped model loading (`enable_mmap`)** — `worker_v1/openvino_worker_v1.py`에서 `ov.Core()` 생성 후 `ov_props.enable_mmap: True` 설정. OpenVINO 2026.0+부터 모델 가중치를 메모리 매핑하여 RAM 사용량 감소.
11. **SSM (State Space Model) / MambaSpec 지원** — hybrid model(Attention + SSM) 지원. `model_executor/model_loader/openvino.py`의 `detect_model_type()`으로 모델 타입 탐지, `apply_selective_paged_attention_transformation()`으로 attention-only 모델에만 PagedAttention 변환 적용. `worker_v1/openvino_worker_v1.py`에서 `MambaSpec` import 및 SSM/conv 상태 캐시 관리.
12. **KV 캐시 `.fill(0)` 제거** — `kv_cache.py`의 `_allocate_kv_cache()`에서 신규 블록 할당 시 `.fill(0)` 제거. OpenVINO가 내부적으로 초기화를 처리하며, 명시적 zero-fill은 대형 KV 캐시에서 불필요한 OOM을 유발했음 (커밋 `e106f19`).
13. **입력 버퍼 pre-allocation 및 `_infer()` wrapper 제거** — `worker_v1/openvino_model_runner_v1.py`에서 매 배치마다 리스트/텐서를 새로 생성하는 대신 고정 스키마의 NumPy 배열을 pre-allocate하여 재사용. `_infer()` wrapper 함수를 제거하고 직접 `ov_request.infer()`를 호출하여 호출 오버헤드 감소 (커밋 `86e4734`, `bffd8cc`, `e86910d`).
14. **InputBuilder Strategy Pattern** — `model_executor/model_loader/openvino.py`에 `OpenVINOInputBuilder` 추상 클래스와 두 구현체(`PAInputBuilder`, `StatefulInputBuilder`)를 도입. `_get_input_builder()`가 컴파일된 모델의 입력 목록에서 KV cache 관련 입력(key_cache.*, value_cache.*) 존재 여부로 자동 분기하여 적절한 빌더를 반환. PA-transformed 모델은 list 기반 입력(기존과 동일)을, stateful 모델(HYBRID_MAMBA 등)은 dict 기반 입력을 사용.
15. **Stateful model 지원 (HYBRID_MAMBA 등)** — `apply_selective_paged_attention_transformation()`이 HYBRID_MAMBA 모델에 대해 PA 변환을 스킵하도록 수정. 이 경우 모델은 남은 ReadValue/Assign stateful ops를 사용하여 나이브하게 추론됨. `StatefulInputBuilder`가 이를 지원하기 위해 컴파일된 모델의 입력 shape을 자동으로 파악하고, `np.tile()`로 단일 요청을 배치 크기만큼 복제하여 OpenVINO 모델에 공급.
16. **Gather-before-matmul 변환 조건 적용** — `apply_gather_before_matmul_transformation()`이 PA-transformed 모델에만 적용되도록 변경. 해당 변환은 `sampled_tokens_indices` 입력 파라미터를 추가하는데, stateful 모델에 이를 적용하면 입력 누락 시 `seq_len=0` 출력이 발생하여 서빙 실패.
17. **Stateful model 단일 요청 제한** — stateful 모델은 동시 요청을 지원하지 않음. `platform.py`의 `check_and_update_config()`에서 stateful 모델 감지 시 `max_num_seqs=1`로 강제 오버라이드. `forward()`에서 `num_requests > 1`이면 `RuntimeError` raise. `num_requests` 파라미터는 PA 모델의 출력 슬라이싱용으로만 사용됨 (`_extract_logits()`에서 `[:num_requests]` 슬라이싱).
18. **bf16 → float32 변환** — `_as_numpy_no_copy()`에서 torch bfloat16 tensor를 numpy로 변환할 때 float32로 캐스팅. OpenVINO는 bf16 numpy array를 지원하지 않음 (커밋 `39faa5d`).
19. **Vision pipeline 완전 지원** — encoder → merger → position embedding 전체 파이프라인 구현 완료. `_prepare_embeddings()`에서 text/vision embedding 통합, `_prepare_vision_inputs()`에서 pixel 입력 처리, `_compute_merger_rotary_pos_emb()`에서 merger용 RoPE 계산. Gemma-4(단순 vision emb), Qwen3.5(vision emb + merger) 모두 검증됨. 모델 파일명 규칙: 비전 인코더 `openvino_vision_embeddings_model.xml`, 비전 merger `openvino_vision_embeddings_merger_model.xml` (dict로 `hidden_states`, `attention_mask`, `rotary_pos_emb` 3개 입력 필수).
20. **Per-layer embeddings model** — `openvino_text_embeddings_per_layer_model.xml` 지원. Gemma-4 계열에서 레이어별 임베딩이 필요한 경우 사용. `StatefulInputBuilder`의 `per_layer_inputs` 처리와 연동. 해당 파일이 모델 디렉토리에 존재하면 자동 로드됨.
21. **Adaptive r-KV / Cache rotation** — `paged_attention_transformation(allow_adaptive_rkv=True, allow_cache_rotation=True)` 활성화 (OpenVINO 2026.3.0). adaptive r-KV는 decode-heavy 워크로드에서 KV 캐시 write bandwidth 감소. cache rotation은 슬라이딩 윈도우 IR 변환이나, 블록 테이블 rotation은 미구현 — PA path에서 슬라이딩 윈도우 모델(Mistral 등) 지원 시 블록 테이블 rotation 추가 구현 필요.
22. **OpenVINO 최소 버전 2026.3.0** — `platform.py`의 `check_and_update_config()`에서 `pip install openvino>=2026.3.0` 요구. `allow_adaptive_rkv`, `allow_cache_rotation` 파라미터가 2026.3.0에서 추가됨.
