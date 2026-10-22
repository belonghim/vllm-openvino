# vllm-openvino

[GitHub](https://github.com/belonghim/vllm-openvino) · OpenVINO plugin for vLLM — run LLM inference on Intel CPUs.

## What is this?

This project provides an OpenVINO backend for vLLM, allowing you to run vLLM's OpenAI-compatible API server on Intel CPUs. It integrates OpenVINO as the inference execution layer, leveraging vLLM's scheduler, PagedAttention, and API server infrastructure. Models must be pre-exported to OpenVINO IR format (openvino_model.xml + openvino_model.bin).

GPU code paths exist (`VLLM_OPENVINO_DEVICE=GPU`) but are not part of the validated configuration; every measurement in this README is CPU-only.

## Requirements

- Python >= 3.10
- Linux (x86-64, AVX2+)

## Installation

### From source

Install vLLM with the OpenVINO backend:

```bash
VLLM_TARGET_DEVICE="empty" PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" pip install .
```

Note: vLLM may install `triton` which is incompatible with OpenVINO. Uninstall it after installation:

```bash
pip uninstall -y triton
```

### Docker

Build the Docker image:

```bash
podman build -f Containerfile -t quay.io/joopark/vllm-openvino .
```

Run the Docker container:

```bash
podman run -d --name vllm-server -p 8080:8080 \
  -e VLLM_OPENVINO_DEVICE=CPU \
  -e TORCH_COMPILE_DISABLE=1 \
  -e VLLM_OPENVINO_KVCACHE_SPACE=8 \
  -e VLLM_OPENVINO_CACHE_DIR=/cache \
  -v ~/models:/models:Z \
  -v ~/ov-cache:/cache:Z \
  quay.io/joopark/vllm-openvino \
  --port=8080 --model /models/<model_dir>
```

`--model` must be a local directory holding the exported IR files, not a HuggingFace
repository id: the image runs with `HF_HUB_OFFLINE=1` and the plugin never downloads
weights. Mounting `VLLM_OPENVINO_CACHE_DIR` keeps the compiled-model blob across
container restarts; without it every start recompiles the model (22 s for a 14 GB
IR, more for larger ones).

## Quick Start

Run the vLLM API server with OpenVINO backend.

```bash
VLLM_OPENVINO_DEVICE=CPU TORCH_COMPILE_DISABLE=1 VLLM_OPENVINO_KVCACHE_SPACE=8 \
  vllm serve --model /models/<model_dir>
```

`<model_dir>` is a local directory with pre-exported OpenVINO IR files
(`openvino_model.xml` for text models, `openvino_language_model.xml` for
multimodal ones). Export a HuggingFace checkpoint with
[optimum-intel](https://github.com/huggingface/optimum-intel) first.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_OPENVINO_DEVICE` | Device selection: CPU, GPU, GPU.1, etc. | `CPU` |
| `VLLM_OPENVINO_KVCACHE_SPACE` | KV cache size in GiB, fractions allowed (0 = auto: CPU sizes the pool from the memory limit after model load, GPU profiles) | `0` |
| `VLLM_OPENVINO_KV_CACHE_PRECISION` | KV cache dtype: `u8`, `i8`, `f16`/`fp16`, `bf16`, `f32`/`fp32` (unset = auto-detected from model). On PagedAttention paths only `u8`/`f16`/`bf16` are supported; `f32`/`i8` fall back to the default. | unset |
| `VLLM_OPENVINO_PERFORMANCE_MODE` | Performance mode: LATENCY or THROUGHPUT | `THROUGHPUT` |
| `VLLM_OPENVINO_CPU_THREADS_NUM` | CPU only. Inference threads (`0` = auto: cgroup CPU quota if constrained, else OpenVINO auto) | `0` |
| `VLLM_OPENVINO_NUM_STREAMS` | CPU only. Inference streams: `AUTO` or integer. vLLM V1 issues a single blocking `infer` per step, so extra streams fragment the thread budget without ever running concurrently — keep at `1` unless the plugin is modified for async inference. | `1` |
| `VLLM_OPENVINO_ENABLE_HYPER_THREADING` | CPU only. Enable/disable hyperthreading: `true`, `false`, or `auto` | `auto` |
| `VLLM_OPENVINO_INFERENCE_PRECISION` | CPU only. Force inference precision: `f32`, `f16`, `bf16` (unset = OpenVINO default) | unset |
| `VLLM_OPENVINO_ENABLE_CPU_PINNING` | CPU only. Enable/disable CPU core pinning: `true`, `false`, or `auto` | `auto` |
| `VLLM_OPENVINO_HYBRID_PA` | Default path for hybrid Mamba/attention models: attempt PagedAttention (concurrent batching) instead of the sequential stateful path. Set to `0` to force the sequential stateful path instead. See [Serving Modes](#serving-modes). | `1` |
| `VLLM_OPENVINO_LA_CACHE_INTERVAL_MULTIPLIER` | Hybrid-PA prefix caching: scheduler block (and conv/SSM checkpoint interval) in 32-token KV blocks (default 8 = 256 tokens). `0` = auto, `max(8, checkpoint bytes / KV block bytes)` as in OpenVINO GenAI (LFM2.5: 256 tokens, Qwen3.5/MiniCPM-V: 1504 tokens). Smaller values give finer prefix hits and use more memory per token; `1` matches the 32-token PA block. See [Serving Modes](#serving-modes). | `8` |
| `VLLM_OPENVINO_STATEFUL_PA` | PagedAttention transformation for plain-attention stateful models (optimum-intel default exports), enabling concurrent batching (sliding-window models such as Gemma-4 included). Models without SDPA ops fall back to the sequential stateful path. Set to `0` to force the sequential stateful path. See [Serving Modes](#serving-modes). | `1` |
| `VLLM_OPENVINO_CACHE_DIR` | Directory for OpenVINO's compiled-model disk cache. Skips recompiling the model on process restart if a matching cached blob exists. When unset, the plugin uses `~/.cache/vllm-openvino` if it can be created, otherwise caching is disabled (an INFO log records which path is in use, if any). | auto |
| `VLLM_OPENVINO_SCHEDULING_CORE_TYPE` | CPU only. Scheduling core type on hybrid P-core/E-core CPUs: `PCORE_ONLY`, `ECORE_ONLY`, `ANY_CORE` | unset |
| `VLLM_OPENVINO_FAST_SAMPLER` | CPU only. Replace vLLM V1's full-vocabulary top-k/top-p sort and dense repetition-penalty routines with partial/sparse equivalents (up to ~2× end-to-end on small models with default `top_k`/`top_p`/`repetition_penalty`). Set to `0` to use stock vLLM. See [Sampling Parameters](#sampling-parameters-large-vocab-models) for seed behavior. | `1` |
| `TORCH_COMPILE_DISABLE` | Must be set to 1; `torch.compile` is incompatible with OpenVINO. | — |

## Performance Tuning

For CPU deployments, especially AVX2-only systems, tuning OpenVINO CPU threading/stream properties can improve sustained tokens/sec.

### Memory Footprint

The KV cache is reserved as one fixed-size mapping and pages become resident only as blocks are used, so RSS climbs toward `baseline + VLLM_OPENVINO_KVCACHE_SPACE` and then plateaus; size the container for that sum. vLLM's prefix caching (on by default) retains freed blocks, so unique-prompt workloads keep touching new pool pages — add `--no-enable-prefix-caching` to hold the resident set at the working set.

Measured baseline, no swap allowed, KV pool fully filled (8 CPUs, `max-model-len 4096`):

| | EngineCore anon RSS | APIServer anon RSS |
|---|---|---|
| Qwen3-1.7B-int4 | weights + 2.5 GiB | 0.84 GiB |
| gemma-4-E2B-int4 | weights + 2.6 GiB | 2.5 GiB (vision) |
| LFM2.5-8B-A1B-int4 | weights + 3.5 GiB | 0.87 GiB |
| Phi-3.5-mini-int4 | weights + 3.0 GiB | 0.55 GiB |
| Qwen3.6-35B-A3B-int4 | weights + 3.6 GiB | 2.1 GiB (vision) |
| gemma-4-26B-A4B-int4 | weights + 7.2 GiB | 2.7 GiB (vision) |

The surplus over the weights is the CPU plugin repacking weights into anonymous memory at the first infer (about 1.5 GiB on the 26B; the pool is sized after that warmup, so the repack is counted), plus prefill/decode buffers that grow with the model and with concurrent prefill. Leave at least the largest figure for your model on top of weights and the KV pool; the 26B needs roughly `weights + 10 GiB` before any KV budget. With the default `VLLM_OPENVINO_KVCACHE_SPACE=0` the CPU pool is sized from the memory ceiling (the cgroup memory limit, or host MemAvailable when no limit is set): at startup from the `.bin` sizes, then after model load and a warmup that prefills `max_num_batched_tokens` tokens (plus one maximum-size dummy image on vision models, whose prompt may be up to 2048 tokens or one step, whichever is larger) from the EngineCore's anonymous RSS (memory-mapped weights excluded — the kernel reclaims them; on the 26B VmRSS reads 33.5 GB where 19.8 GB is anonymous), minus the APIServer overhead (1.5 GiB text, 2.75 GiB vision) and 2.2 GiB headroom. The APIServer and prefill buffers must fit in that headroom because the pool's pages are all touched once unique prompts have cycled through it. Because the pool fills the remaining memory, a container limit must leave room for prefill buffers and the concurrent-prefill surplus; set an explicit `VLLM_OPENVINO_KVCACHE_SPACE` to cap the pool deliberately. An explicit value is only reduced when it does not fit. Measured with that sizing (`max-model-len 4096`): the 26B in a 32 GiB container gets a 7.25 GB pool and served three concurrent image requests, then 56 unique ~2.5k-token prompts four at a time (about 141k tokens, more than the pool holds), all HTTP 200 with no OOM; the cgroup ended at 33.1 of 34.4 GB, so the margin is thin. A 24 GiB container no longer starts: the warmed-up baseline leaves nothing above the headroom, and the 256 MiB floor pool is smaller than one 4096-token request. gemma-4-E2B in a 16 GiB container (7.21 GB pool) served three concurrent images and 224 unique ~2.5k-token prompts four at a time (about 966k tokens) with no OOM. Before the warmup covered a full prefill step and an image, both models were OOM-killed under these loads once the pool was fully resident. Leaving the memory-mapped weights out of the budget does not cost throughput: on the 26B in 32 GiB (8 CPUs, pool filled with 60 unique ~2k-word prompts) the cgroup's file pages fell from 9.8 to 1.2 GB as the pool grew, while file refaults rose by only about 9k pages (35 MB), and 4-way decode stayed at 21.4 tok/s against 21.8 tok/s before the fill, so the evicted weight pages are not read during inference.

### KV Cache Quantization

The KV cache precision can be reduced to significantly lower memory usage:

| Precision | Memory | Notes |
|-----------|--------|-------|
| `u8` | Lowest | 8-bit unsigned integer; smallest footprint, default for the PagedAttention-transformed stateful path |
| `f16` / `bf16` | Medium | Good balance; `bf16` needs AVX-512-BF16 on CPU (see the warning below) |
| `i8` / `f32` | — | Rejected by CPU PagedAttention on the PagedAttention and Hybrid-PA paths; falls back to the default with a warning |

Set via environment variable:
```bash
VLLM_OPENVINO_KV_CACHE_PRECISION=u8 \
  vllm serve --model <model_id>
```

> **Warning**: on CPUs without AVX-512-BF16 (e.g. Xeon E5-2670 v3, Haswell), setting `VLLM_OPENVINO_KV_CACHE_PRECISION=bf16` fails engine startup with `executor_pa.cpp:2938: expect kvcache type f32, current: bf16`. Use `f16` or `u8` instead.

### CPU Tuning (AVX2)

On AVX2-only CPUs, int4 models usually show a larger throughput gap vs AVX-512/VNNI capable CPUs due to lower effective low-precision compute throughput. In practice, CPU scheduling knobs (threads, affinity, streams) are often the main software lever for improving throughput stability.

For older AVX2 systems, fp16 or int8 models are often a better latency/throughput trade-off than int4.

| Variable | Type | Values | Effect |
|----------|------|--------|--------|
| `VLLM_OPENVINO_CPU_THREADS_NUM` | int | `0` (auto), `1..N` | Caps OpenVINO CPU inference threads |
| `VLLM_OPENVINO_NUM_STREAMS` | str/int | `AUTO`, `1..N` | Controls number of parallel CPU inference streams. Default `1`: vLLM V1 is serial (one blocking `infer` per step), so extra streams only fragment the thread budget |
| `VLLM_OPENVINO_ENABLE_HYPER_THREADING` | bool | `true`, `false`, `auto` | Disabling prevents HT oversubscription on 2-socket systems |
| `VLLM_OPENVINO_INFERENCE_PRECISION` | str | `f32`, `f16`, `bf16` (unset = OpenVINO default) | Forces specific precision for matmul operations |
| `VLLM_OPENVINO_ENABLE_CPU_PINNING` | bool | `true`, `false`, `auto` | Controls thread-to-core pinning |
| `VLLM_OPENVINO_FAST_SAMPLER` | int | `1` (default), `0` | `1` uses the partial/sparse CPU sampling routines; `0` restores stock vLLM sampling |

**cgroup-aware auto-detection**: container runtimes (podman/docker `--cpus`, Kubernetes CPU limits) throttle via cgroup CFS quota, but `os.cpu_count()` inside the container still reports the host's full core count. With `VLLM_OPENVINO_CPU_THREADS_NUM=0` (default), the plugin detects the quota and caps both the OpenVINO inference threads and the Torch/OMP thread pools to it when the quota is tighter than the visible core count; an explicit `OMP_NUM_THREADS` is left untouched. On an 8-quota/24-visible-core host, a 32-request burst went from 4.5s wall / 35.6s CPU with 3.1s of CFS throttling to 3.6s wall / 26.6s CPU with none, and the engine process dropped from 147 to 126 threads (earlier OpenVINO-only measurement: 45s/~355% CPU uncapped vs 34s/~300% capped for a 4-request burst). Set `VLLM_OPENVINO_CPU_THREADS_NUM` explicitly to override this detection.

`PERFORMANCE_MODE` defaults to `THROUGHPUT` because vLLM is used for serving. Measured on Qwen2.5-Coder-0.5B-int4-ov (8-CPU quota): THROUGHPUT gave +28–31% single-stream and +10% concurrency-8 aggregate decode throughput on the PagedAttention path over LATENCY (and +28% single-stream on the stateful path), at a higher first-token latency (25 ms → 34 ms). Set `VLLM_OPENVINO_PERFORMANCE_MODE=LATENCY` for interactive first-token response instead.

Other CPU compile properties were measured on Qwen3-1.7B-int4-ov (8-CPU quota, Ryzen 9 7900, OpenVINO 2026.4.1; concurrency 1/8, 128 tokens) and left at OpenVINO's defaults, which are already the best or equal within ~5% noise: `INFERENCE_PRECISION` (default is already bf16 here; `bf16`/`f32` change nothing), `ENABLE_CPU_PINNING` (default already on), and `DYNAMIC_QUANTIZATION_GROUP_SIZE` (default 32; 64/128 equal, disabling it or per-token costs 15%). Forcing `ENABLE_HYPER_THREADING=false` cost 13% at concurrency 8. `KEY/VALUE_CACHE_GROUP_SIZE=64` did not help.

Caveat: on this plugin, `NUM_STREAMS=AUTO` under `THROUGHPUT` fragments threads across streams that vLLM V1 never runs concurrently — the runner holds a single `create_infer_request` and issues a single blocking `infer` per step. Setting `NUM_STREAMS=1` explicitly measured a 3.56× throughput gain on Qwen3-1.7B-int4-ov at concurrency 8 (8.53 → 30.38 tok/s), which is why the default is `1`. The same holds on hybrid-mamba models via the Hybrid-PA path: at concurrency 8 with `VLLM_OPENVINO_CPU_THREADS_NUM=24`, `NUM_STREAMS=1` vs `AUTO` measured 59.40 → 142.71 tok/s (2.40×) on LFM2.5-350M-int8-ov and 25.39 → 67.83 tok/s (2.67×) on Qwen3.5-0.8B-int4-ov (text-only) — vLLM V1 is serial regardless of model architecture, so stream fragmentation hurts dense and hybrid alike.

**Multi-socket placement**: OpenVINO binds threads to every core the process can see, including cores on a second socket. Measured on a 2-socket Xeon E5-2670 v3 (12 physical cores per socket) with Qwen3.5-0.8B-int4-ov, 90 s guidellm runs:

| Threads | Sockets | Output tok/s, 2 streams | Output tok/s, 8 streams |
|---------|---------|-------------------------|-------------------------|
| 8 | node0 only | 14.7 | 22.7 |
| 8 | node0 + node1 (4+4) | 13.0 (−12%) | 18.6 (−18%) |
| 12 | node0 only | 18.5 | 27.6 |
| 24 | node0 + node1 (12+12) | 18.7 (+1%) | 30.0 (+9%) |

Splitting a fixed thread budget across sockets costs 12–18% because the threads on the far socket read the weights over the interconnect, and doubling the thread count across both sockets buys at most ~9% and only under load. `MPOL_INTERLEAVE` across both nodes (`numactl --interleave=all` or the equivalent `set_mempolicy` syscall) does not close the gap — 29.7 vs 30.0 tok/s on the same 24-thread dual-socket run — so the bottleneck is memory bandwidth, not weight locality. Keep inference threads inside one socket by binding the process to that socket's CPU set and setting `VLLM_OPENVINO_CPU_THREADS_NUM` to its core count.

Running several engines instead (`--data-parallel-size` with one engine per socket or per L3/CCD group) is not expected to help either: on one socket every engine shares the same memory bandwidth, and each engine holds its own copy of the weights and its own KV pool. This is inferred from the numbers above, not measured.

Example (latency-optimized for AVX2, 2-socket Xeon):

```bash
VLLM_OPENVINO_DEVICE=CPU \
VLLM_OPENVINO_PERFORMANCE_MODE=LATENCY \
VLLM_OPENVINO_CPU_THREADS_NUM=24 \
VLLM_OPENVINO_NUM_STREAMS=1 \
VLLM_OPENVINO_ENABLE_HYPER_THREADING=false \
TORCH_COMPILE_DISABLE=1 \
vllm serve --model <model_id>
```

### Chunked Prefill Size (`--max-num-batched-tokens`)

The plugin keeps vLLM's default for non-CPU platforms (2048 tokens per step for the API server). On CPU a 2048-token prefill chunk takes about 3 s, and every running decode stalls for that whole step. Measured on Qwen3-1.7B-int4-ov (8-CPU quota): 4 decode streams running while 4 prompts of ~2.7k tokens arrive; decode inter-token latency p99 / prefill TTFT mean:

| `--max-num-batched-tokens` | decode ITL p99 | prefill TTFT mean | total wall |
|---|---|---|---|
| 2048 (default) | ~3.3 s | ~17.5 s | ~33 s |
| 1024 | ~1.75 s | ~16 s | ~33 s |
| 512 | ~0.9 s | ~15.5 s | ~33 s |
| 256 | ~0.7 s | ~19–21 s | ~36–42 s |

Pass `--max-num-batched-tokens 512` for interactive/streaming workloads that mix long prompts with ongoing generations: it cuts worst-case decode stalls about 3.5× at no throughput cost. 256 starts losing throughput. Decode-only or batch workloads are unaffected. Images may be split across steps: the vision encoder runs once, in the step that reaches the image, and the rest of the image is read from vLLM's encoder cache. The gain depends on the model: gemma-4-E2B (about 260 tokens per image) showed no stall from two concurrent images at either size, while on Qwen3.5-0.8B (Hybrid-PA), two concurrent 1024×1024 images (1024 tokens each) joining four running decodes cut the worst stall from 15.3 s to 7.3 s but raised decode ITL p99 from 0.07 s to 6.7 s, because the image prefill then spans more steps; total wall time was unchanged. Runs on this shared host varied by ~±15% in wall time; the p99 trend was consistent across repeats.

### Sampling Parameters (large-vocab models)

Sampling runs on the critical path of every decode step. For small models it costs as much as the OpenVINO inference, so the plugin replaces three of vLLM's CPU sampling routines (see the fast-sampler paragraph below). `temperature=0` (greedy) skips top-k/top-p and all randomness entirely, but still applies the repetition penalty when the model's `generation_config.json` sets one (Qwen2.5-Coder ships `repetition_penalty=1.1`, `top_k=20`, `top_p=0.8`).

**Upstream vLLM V1 `temperature=0` all-greedy fast path**: `Sampler.sample()` (`vllm/v1/sample/sampler.py:257-271`) early-returns after a bare `logits.argmax(dim=-1)` when every request in the batch is greedy (`SamplingType.GREEDY`, which vLLM sets when `temperature<=1e-5`). That skips `apply_temperature`, argmax-invariant logits processors, the full `TopKTopPSampler` pipeline (top-k/top-p filtering and the random draw), and the final `torch.where` merge. Empirical on LFM2.5-350M-int8-ov at concurrency 8, `NUM_STREAMS=1`, `CPU_THREADS_NUM=24`, `max_tokens=128`, 3-run mean: temp=0 **120.26 tok/s** vs temp=0.7 **86.47 tok/s** — temp=0 is **1.39× faster aggregate** and **1.47× faster per-longest-request** (128/wave wall). The greedy fast path is inherited from upstream unchanged.

**Random sampling (`torch.ops._C.fused_gumbel_argmax`)**: vLLM 0.32's `forward_cpu` calls a C++ Gumbel-max kernel that the wheel (built with `VLLM_TARGET_DEVICE=empty`) does not ship, so every non-greedy request failed with HTTP 500 until the plugin registered its own implementation in `platform.py`; it is used regardless of `VLLM_OPENVINO_FAST_SAMPLER`. The kernel draws per-token noise over the whole vocabulary; a categorical draw needs one uniform number per row, so the plugin uses softmax -> cumsum -> `searchsorted(right=True)`, which samples the same distribution and never selects a zero-probability token (20k draws matched the softmax probabilities to within 0.4 points, masked token never chosen; 1.25 ms at batch 8, vocab 151,936). Each row's uniform comes from the per-row seed vLLM passes, which it takes from the request's `torch.Generator` when the request sets `seed`: three `seed=42` requests returned identical text, also when batched with unseeded requests, and unseeded requests differed. Concurrency-8 `temperature=0.7` throughput: LFM2.5-350M-int8-ov 381 tok/s, Qwen3.5-0.8B-int4-ov 139 tok/s.

**Plugin CPU fast-sampler (`VLLM_OPENVINO_FAST_SAMPLER`, default `1`)**: `sampler_patch.py` replaces two upstream routines whose cost scales with the full `(batch, vocab)` tensor even though the result depends on a handful of tokens. (1) `apply_top_k_top_p` sorted the whole vocabulary whenever `top_p` was set; the patch takes a partial `topk` over `max(k)` candidates (1024 when only `top_p` is set) and computes the nucleus on them, falling back to upstream when the candidates hold less than `p` of the mass. (2) `apply_all_penalties` built dense `(batch, vocab)` bin-count and mask tensors; the repetition penalty is applied only at prompt/output token positions, and batches using presence/frequency penalties take the upstream path. Output equivalence against the upstream functions was checked on random and peaked logits: identical kept-token sets and bit-identical values for top-k/top-p and for penalties. Micro-benchmark at batch 8, vocab 151,936: top-k+top-p 17.8 -> 0.9 ms (top-p only on peaked logits 17.9 -> 1.2 ms), repetition penalty 3.3 -> 0.3 ms. End-to-end numbers measured before vLLM 0.32, when the patch also replaced the random-sample routine, on Qwen2.5-Coder-0.5B-int4-ov (PagedAttention, 8-CPU quota, `max_tokens=128`, `ignore_eos`, 3-run mean, model-default `top_k=20`/`top_p=0.8`/`repetition_penalty=1.1`): concurrency 1 greedy 48.0 -> 68.6 tok/s and temp 0.7 39.6 -> 51.0; concurrency 8 greedy 284 -> 366 and temp 0.7 183 -> 345. temp=0 output is unchanged. Set `VLLM_OPENVINO_FAST_SAMPLER=0` to use the upstream top-k/top-p and penalty routines.

**Foreign-script tokens in Korean output (Qwen3.5-0.8B-int4-ov)**: the stock and fast samplers draw from the same distribution. At `temperature=1.0` with no `top_k`/`top_p` a 0.8B int4 model still emits stray CJK/Cyrillic/Arabic characters (about 25–70 per 1k tokens, 16–32 Korean prompts, concurrency 8). The model's `generation_config.json` carries no sampling defaults, so vLLM samples the full vocabulary. Greedy and `temperature=0.7, top_k=20, top_p=0.8` (Qwen's non-thinking recommendation) produced 0 foreign characters over 4k tokens each; pass those parameters per request for small Qwen3.5 models.

**Vision-embedding cache**: vision encoding follows vLLM's encoder scheduling, as in upstream's GPU model runner. The scheduler sends the images to encode in each step (`scheduled_encoder_inputs`) and the cache entries to drop (`free_encoder_mm_hashes`); the runner keeps the merged embeddings keyed by the image's content hash. vLLM sizes that cache (`max(max_num_batched_tokens, tokens of the largest image)` tokens) and keeps entries after their request finishes until it needs the space, so concurrent duplicates and a repeated image behind a different prefix skip the vision tower and merger. Measured on Qwen3.5-0.8B-int4-ov (1024×1024 image, 1024 tokens): first request 4.8 s TTFT, the same image after a different leading sentence 3.55 s. When the image-containing prefix is already in KV blocks, prefix caching skips the image regardless (0.15 s).

**Measured result: in-graph TopK / sampling not pursued**: the host-side partial `topk` above already removes the full-vocabulary sort (0.9 ms at batch 8, vocab 151,936), so changing the compiled graph output contract to emit only top candidates would save at most that. OpenVINO GenAI keeps full-vocabulary logits and samples host-side with a C++ min-heap TopKFilter plus host-side temperature, top-p, and multinomial, which is the same shape as this approach.

### Memory-Mapped Model Loading

OpenVINO automatically memory-maps model weights, reducing RAM usage during model loading by mapping weights directly from disk rather than copying them into memory. No configuration is required.

### Benchmarking

To measure throughput/latency improvements, use the provided benchmark script:

```bash
./scripts/benchmark.sh <model_path> [num_requests]
```

The script runs a warmed-up benchmark against the local OpenAI-compatible endpoint and reports tokens/sec.

`./scripts/socket-experiment.sh <model_name> [seconds]` runs the same guidellm workload against containers bound to different CPU sets (single socket vs split across sockets) and prints the comparison behind the multi-socket numbers above.

## Serving Modes

The plugin supports three serving paths depending on the model architecture:

### PagedAttention (default)

Models with `ScaledDotProductAttention` ops and no state are transformed to use vLLM's PagedAttention mechanism. This enables:
- Concurrent request batching
- External KV cache management
- Full vLLM scheduler features

### Stateful Path

Models without `ScaledDotProductAttention` ops, or any stateful model with `VLLM_OPENVINO_STATEFUL_PA=0`, run via OpenVINO's internal state management (`ReadValue`/`Assign`). Characteristics:
- Sequential request processing (`max_num_seqs=1`)
- Internal KV cache managed by OpenVINO runtime
- Automatic detection and configuration — no manual flags needed

Other `ReadValue`-based stateful models (optimum-intel's default exports such as Qwen2.5-Coder-int4-ov) get the PagedAttention transformation at load time by default (`VLLM_OPENVINO_STATEFUL_PA=1`), keeping `max_num_seqs` at its default 128. Validated on four dense architectures (Qwen2.5-Coder-0.5B, Qwen3-1.7B, TinyLlama-1.1B, Phi-3.5-mini; int4 IR): concurrency-8 aggregate throughput was 4.9–7.0x the stateful path, with a single-stream delta of -7.8% to +4%. Sliding-window models use the same path: Gemma-4-E2B-it-int4-ov and gemma-4-26b-a4b-it-int4-ov (text and image) gave the same greedy output as the stateful path and 6.7x / 3.2x its concurrency-8 throughput (18.7 -> 125.2 and 9.8 -> 30.9 tok/s). The transformed model's KV cache defaults to u8; `VLLM_OPENVINO_KV_CACHE_PRECISION` is honored for `u8`/`f16`/`bf16` (`f32`/`i8` are unsupported by CPU PagedAttention and fall back to the default with a warning). Greedy output was unchanged on TinyLlama-1.1B but can differ from the stateful path on Qwen-family models — a property of the PagedAttention-transformed attention computation, not of cache precision (u8 and bf16 produced identical output). Set `VLLM_OPENVINO_STATEFUL_PA=0` to force the sequential stateful path.

### Hybrid-PA (default for hybrid Mamba/attention models)

For hybrid Mamba/attention models (Qwen3.5, LFM2.5), this converts attention layers to real PagedAttention and conv/SSM layers to a separate linear-attention paged-state mechanism, enabling concurrent request batching (`max_num_seqs > 1`) instead of the sequential stateful path. Verified on LFM2.5 (conv-only) and Qwen3.5 (conv + GatedDeltaNet SSM) with byte-exact output vs. the stateful path. Gemma-4 has no SSM/conv state, so it takes the plain PagedAttention path instead. Set `VLLM_OPENVINO_HYBRID_PA=0` to force the sequential stateful path instead.

Prefix caching works on Hybrid-PA (not with speculative decoding): every scheduler block keeps a conv/SSM state checkpoint, so a hit restores the state at the last hit block. Hits land on the checkpoint interval (see `VLLM_OPENVINO_LA_CACHE_INTERVAL_MULTIPLIER`). Measured hit TTFT: LFM2.5-350M 2.1k-token prompt 0.74 s -> 0.06 s; Qwen3.5-0.8B 2.4k-token prompt 8.2 s -> 0.20 s with the default 256-token interval; its 10.3 MB checkpoints leave about 90k tokens in the 4 GiB pool (280k with the automatic 1504-token interval, `0`, where the same prompt hits in 2.9 s). Details in `docs/compatibility.md`.

## Compatibility

The following vLLM features are compatible with the OpenVINO backend:

- Chunked prefill (`--enable-chunked-prefill`)
- Structured outputs (`structured_outputs`, `response_format`, forced tool calls) via the grammar bitmask
- N-gram speculative decoding (`--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":5,"prompt_lookup_max":8}'`) on PagedAttention and Hybrid-PA models; drafts are verified greedily and only for greedy requests without penalties, logprobs, or structured output. Stateful-path models reject it. It speeds up copy-heavy output (code edits, quoting, RAG); see `docs/compatibility.md`
- Gemma 3 and Gemma 4 text and multimodal (text + image)
- Qwen3.5 and LFM2.5 (hybrid Mamba/attention architecture, via Hybrid-PA)
- MiniCPM-V 4.7 text and image (Hybrid-PA); needs vLLM >= 0.32.0 and Transformers >= 5.18.0, see `docs/compatibility.md`

## Limitations

- LoRA serving is not supported.
- Pin memory is not supported.
- `prompt_logprobs` (and `echo` with `logprobs`) is not supported; such requests are rejected with HTTP 400 because logits are computed only at sampled positions. Output-token `logprobs` work.
- Tensor/pipeline parallelism is not supported. Threads are scheduled across all visible cores, but multi-socket scaling is poor; see [CPU Tuning](#cpu-tuning-avx2).
- vLLM V1 engine only.
- Stateful-path models (no SDPA ops, or `VLLM_OPENVINO_STATEFUL_PA=0`) do not support concurrent request execution (`max_num_seqs=1`).

See `docs/compatibility.md` for the current support matrix.
