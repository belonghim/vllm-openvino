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
11. **SSM (State Space Model) / MambaSpec 지원** — hybrid model(Attention + SSM) 지원. `model_executor/model_loader/openvino.py`의 `detect_model_type()`으로 모델 타입 탐지. 기본적으로 hybrid 모델은 PagedAttention 변환을 스킵하고 stateful로 서빙됨 (`VLLM_OPENVINO_HYBRID_PA=1`로 opt-in PA 변환 가능 — 항목 23 참조). `worker_v1/openvino_worker_v1.py`에서 `MambaSpec` import 및 SSM/conv 상태 캐시 관리 (기본 stateful 경로용; hybrid-PA 경로는 별도의 private slot pool 사용).
12. **KV 캐시 `.fill(0)` 제거** — `kv_cache.py`의 `_allocate_kv_cache()`에서 신규 블록 할당 시 `.fill(0)` 제거. OpenVINO가 내부적으로 초기화를 처리하며, 명시적 zero-fill은 대형 KV 캐시에서 불필요한 OOM을 유발했음 (커밋 `e106f19`).
13. **입력 버퍼 pre-allocation 및 `_infer()` wrapper 제거** — `worker_v1/openvino_model_runner_v1.py`에서 매 배치마다 리스트/텐서를 새로 생성하는 대신 고정 스키마의 NumPy 배열을 pre-allocate하여 재사용. `_infer()` wrapper 함수를 제거하고 직접 `ov_request.infer()`를 호출하여 호출 오버헤드 감소 (커밋 `86e4734`, `bffd8cc`, `e86910d`).
14. **InputBuilder Strategy Pattern** — `model_executor/model_loader/openvino.py`에 `OpenVINOInputBuilder` 추상 클래스와 두 구현체(`PAInputBuilder`, `StatefulInputBuilder`)를 도입. `_get_input_builder()`가 컴파일된 모델의 입력 목록에서 KV cache 관련 입력(key_cache.*, value_cache.*) 존재 여부로 자동 분기하여 적절한 빌더를 반환. PA-transformed 모델은 list 기반 입력(기존과 동일)을, stateful 모델(HYBRID_MAMBA 등)은 dict 기반 입력을 사용.
15. **Stateful model 지원 (HYBRID_MAMBA 등)** — `apply_selective_paged_attention_transformation()`이 기본적으로 HYBRID_MAMBA 모델에 대해 PA 변환을 스킵하도록 함. 이 경우 모델은 남은 ReadValue/Assign stateful ops를 사용하여 나이브하게 추론됨. `StatefulInputBuilder`가 컴파일된 모델의 입력 shape을 자동으로 파악하여 OpenVINO 모델에 공급 — `np.tile()` 배치 복제 코드가 존재하지만 stateful 모델은 항상 `max_num_seqs=1`이 강제되어 실질적으로 도달하지 않는 dead code임.
16. **Gather-before-matmul 변환 조건 적용** — `apply_gather_before_matmul_transformation()`이 PA-transformed 모델에만 적용되도록 변경. 해당 변환은 `sampled_tokens_indices` 입력 파라미터를 추가하는데, stateful 모델에 이를 적용하면 입력 누락 시 `seq_len=0` 출력이 발생하여 서빙 실패.
17. **Stateful model 단일 요청 제한** — stateful 모델은 동시 요청을 지원하지 않음. `platform.py`의 `check_and_update_config()`에서 stateful 모델 감지 시 `max_num_seqs=1`로 강제 오버라이드 (단, `VLLM_OPENVINO_HYBRID_PA=1` + hybrid-PA 적격 모델인 경우 예외 — 항목 23 참조). `forward()`에서 `num_requests > 1`이면 `RuntimeError` raise. `num_requests` 파라미터는 PA 모델의 출력 슬라이싱용으로만 사용됨 (`_extract_logits()`에서 `[:num_requests]` 슬라이싱).
18. **bf16 → float32 변환** — `_as_numpy_no_copy()`에서 torch bfloat16 tensor를 numpy로 변환할 때 float32로 캐스팅. OpenVINO는 bf16 numpy array를 지원하지 않음 (커밋 `39faa5d`).
19. **Vision pipeline 완전 지원** — encoder → merger → position embedding 전체 파이프라인 구현 완료. `_prepare_embeddings()`에서 text/vision embedding 통합, `_prepare_vision_inputs()`에서 pixel 입력 처리, `_compute_merger_rotary_pos_emb()`에서 merger용 RoPE 계산. Gemma-4(단순 vision emb), Qwen3.5(vision emb + merger) 모두 검증됨. 모델 파일명 규칙: 비전 인코더 `openvino_vision_embeddings_model.xml`, 비전 merger `openvino_vision_embeddings_merger_model.xml` (dict로 `hidden_states`, `attention_mask`, `rotary_pos_emb` 3개 입력 필수).
20. **Per-layer embeddings model** — `openvino_text_embeddings_per_layer_model.xml` 지원. Gemma-4 계열에서 레이어별 임베딩이 필요한 경우 사용. `StatefulInputBuilder`의 `per_layer_inputs` 처리와 연동. 해당 파일이 모델 디렉토리에 존재하면 자동 로드됨.
21. **Adaptive r-KV / Cache rotation 비활성화** — `paged_attention_transformation(allow_adaptive_rkv=False, allow_cache_rotation=False)`로 되돌림 (2026-08-18). adaptive r-KV는 KV 캐시 write bandwidth를 줄이는 대신 decode마다 KV 엔트리 스코어링 연산을 추가하는데, 이 연산 자체가 CPU 사용량을 오히려 증가시키는 것으로 확인됨. cache rotation도 실제 사용 중인 슬라이딩 윈도우 모델이 없어(`supports_sliding_window()=False`) 불필요한 오버헤드였음.
22. **OpenVINO 최소 버전 2026.3.0** — `platform.py`의 `check_and_update_config()`에서 `pip install openvino>=2026.3.0` 요구. `allow_adaptive_rkv`, `allow_cache_rotation` 파라미터가 2026.3.0에서 추가됨 (현재는 둘 다 비활성화 상태로 사용, 버전 요구사항 자체는 다른 API에도 필요하여 유지).
23. **Hybrid-PA path (실험적, opt-in, `VLLM_OPENVINO_HYBRID_PA=1`)** — 기존 코드 주석에 있던 "hybrid 모델에 PA 변환 시 SSM Gather/Reshape 노드에서 C++ 패턴 매처가 크래시한다"는 가정은 OpenVINO 2026.3.0에서 재검증한 결과 **더 이상 사실이 아님**으로 확인됨 (2026-08-17). `paged_attention_transformation()`을 LFM2.5(conv-only)와 Qwen3.5(conv+GatedDeltaNet SSM) 양쪽 모두에 적용한 결과 크래시 없이 성공하고, stateful 경로 baseline과 byte-exact 일치하는 출력을 확인함. 이 발견에 따라 `apply_selective_paged_attention_transformation()`에 opt-in 분기 추가:
    - attention 레이어는 표준 PagedAttention(`key_cache.N`/`value_cache.N`)으로 변환
    - conv 레이어는 `la.*` + `conv_state_table.N` paged-state 메커니즘으로, SSM(GatedDeltaNet) 레이어는 동일한 `la.*` 인덱스를 공유하는 `gated_delta_state_table.N`으로 변환됨 (OpenVINO 코어의 `SDPAToPagedAttention` 패스가 자동 생성)
    - `worker_v1/openvino_model_runner_v1.py`에 시퀀스당 전용 slot pool(`_conv_slot_by_req`/`_conv_slot_free`, 크기 `max_num_seqs+1`)을 신설해 conv/SSM state를 관리 — vLLM의 `MambaSpec` 기반 블록 그룹핑은 의도적으로 우회함. 이유: vLLM이 6개 초과 mamba 레이어를 여러 block table로 자동 분할(striping)하는데, 이 모델들은 모든 conv/SSM 레이어가 **하나의 공유** `la.*` 입력을 쓰므로 여러 block table과 근본적으로 호환 불가 (`get_kv_cache_spec()`에서 hybrid-PA 모델은 MambaSpec 등록 자체를 스킵)
    - `platform.py`의 `_is_hybrid_pa_candidate()`가 `check_and_update_config()` 시점에 IR XML을 스캔해 `max_num_seqs=1` 강제를 조건부로 해제. 변환이 실제로 실패하면(`apply_selective_paged_attention_transformation()`에서) 조용히 stateful로 폴백하지 않고 예외를 raise함 — 이미 `max_num_seqs`가 완화된 상태에서 조용한 폴백은 다중 요청 스케줄러가 단일 요청 전용 stateful 경로를 향하게 만들어 위험함
    - Gemma-4(STATEFUL, hybrid 아님)는 이 경로 대상이 아님 — PA 변환 자체는 성공하나 local/global sliding-window 레이어(5개 레이어마다 head_size가 256↔512로 상이)에서 실제 추론 시 shape mismatch 발생, 미해결. 항목 21의 슬라이딩 윈도우 블록 테이블 rotation 미구현과 관련된 것으로 추정
    - 진짜 SSM이 없는 conv-only 모델과 SSM 있는 hybrid 모델 모두 동일하게 처리 시도함 (기존에는 SSM 유무로 분기했었으나, 크래시 가정이 틀렸음이 확인되어 분기 제거)
24. **cgroup CPU quota 자동감지** — `model_loader/openvino.py`의 `_detect_cgroup_cpu_quota()` (2026-08-18). 컨테이너 런타임(podman/docker `--cpus`, Kubernetes CPU limit)은 cgroup CFS quota로 CPU를 스로틀링하지만, 컨테이너 내부의 `os.cpu_count()`/`sched_getaffinity()`는 호스트 전체 코어 수를 그대로 보고한다. `VLLM_OPENVINO_CPU_THREADS_NUM=0`(기본값, auto)일 때 OpenVINO가 이 호스트 전체 코어 수 기준으로 스레드 풀을 잡아 실제 quota 안에서 과다 경쟁(oversubscription)이 발생함. `cgroup v2`(`/sys/fs/cgroup/cpu.max`)와 `v1`(`cpu.cfs_quota_us`/`cpu.cfs_period_us`) 둘 다 지원하며, quota가 가시 코어 수보다 작을 때만 그 값으로 `inference_num_threads`를 캡핑. 실측(8-quota/24-core 호스트, Qwen3.5-0.8B stateful path, 4개 동시 요청×200 토큰): 미적용 시 EngineCore 스레드 91개·45초·~355% CPU, 적용 시 8개·34초·~300% CPU (동일 워크로드 대비 25% 빠르고 CPU 사용량도 낮음). quota가 없거나(`unconstrained`) 가시 코어 수 이상이면 개입하지 않음(기존 동작 유지). `VLLM_OPENVINO_CPU_THREADS_NUM`을 명시적으로 지정하면 자동감지를 건너뛰고 그 값을 그대로 사용.
