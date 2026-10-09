# vllm-openvino

[GitHub](https://github.com/belonghim/vllm-openvino) · OpenVINO plugin for vLLM — run LLM inference on Intel CPUs and GPUs.

## What is this?

This project provides an OpenVINO backend for vLLM, allowing you to run vLLM's OpenAI-compatible API server on Intel CPUs and GPUs. It integrates OpenVINO as the inference execution layer, leveraging vLLM's scheduler, PagedAttention, and API server infrastructure. Models must be pre-exported to OpenVINO IR format (openvino_model.xml + openvino_model.bin).

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
  quay.io/joopark/vllm-openvino \
  --port=8080 --model <model_id>
```

## Quick Start

Run the vLLM API server with OpenVINO backend.

For CPU:

```bash
VLLM_OPENVINO_DEVICE=CPU TORCH_COMPILE_DISABLE=1 VLLM_OPENVINO_KVCACHE_SPACE=8 \
  vllm serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

For GPU:

```bash
VLLM_OPENVINO_DEVICE=GPU TORCH_COMPILE_DISABLE=1 \
  vllm serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Replace `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with a local path to pre-exported OpenVINO IR files (directory containing openvino_model.xml and openvino_model.bin).

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_OPENVINO_DEVICE` | Device selection: CPU, GPU, GPU.1, etc. | `CPU` |
| `VLLM_OPENVINO_KVCACHE_SPACE` | KV cache size in GB (0 = auto: 4 GB on CPU) | `0` |
| `VLLM_OPENVINO_KV_CACHE_PRECISION` | KV cache dtype: `u8`, `i8`, `f16`/`fp16`, `bf16`, `f32`/`fp32` (unset = auto-detected from model). On PagedAttention paths only `u8`/`f16`/`bf16` are supported; `f32`/`i8` fall back to the default. | unset |
| `VLLM_OPENVINO_PERFORMANCE_MODE` | Performance mode: LATENCY or THROUGHPUT | `THROUGHPUT` |
| `VLLM_OPENVINO_CPU_THREADS_NUM` | CPU only. Inference threads (`0` = auto: cgroup CPU quota if constrained, else OpenVINO auto) | `0` |
| `VLLM_OPENVINO_NUM_STREAMS` | CPU only. Inference streams: `AUTO` or integer. vLLM V1 issues a single blocking `infer` per step, so extra streams fragment the thread budget without ever running concurrently — keep at `1` unless the plugin is modified for async inference. | `1` |
| `VLLM_OPENVINO_ENABLE_HYPER_THREADING` | CPU only. Enable/disable hyperthreading: `true`, `false`, or `auto` | `auto` |
| `VLLM_OPENVINO_INFERENCE_PRECISION` | CPU only. Force inference precision: `f32`, `f16`, `bf16` (unset = OpenVINO default) | unset |
| `VLLM_OPENVINO_ENABLE_CPU_PINNING` | CPU only. Enable/disable CPU core pinning: `true`, `false`, or `auto` | `auto` |
| `VLLM_OPENVINO_HYBRID_PA` | Default path for hybrid Mamba/attention models: attempt PagedAttention (concurrent batching) instead of the sequential stateful path. Set to `0` to force the sequential stateful path instead. See [Serving Modes](#serving-modes). | `1` |
| `VLLM_OPENVINO_STATEFUL_PA` | PagedAttention transformation for plain-attention stateful models (optimum-intel default exports), enabling concurrent batching. Models with a sliding window smaller than `max_model_len` and models without SDPA ops fall back to the sequential stateful path. Set to `0` to force the sequential stateful path. See [Serving Modes](#serving-modes). | `1` |
| `VLLM_OPENVINO_CACHE_DIR` | Directory for OpenVINO's compiled-model disk cache. Skips recompiling the model on process restart if a matching cached blob exists. When unset, the plugin uses `~/.cache/vllm-openvino` if it can be created, otherwise caching is disabled (an INFO log records which path is in use, if any). | auto |
| `VLLM_OPENVINO_SCHEDULING_CORE_TYPE` | CPU only. Scheduling core type on hybrid P-core/E-core CPUs: `PCORE_ONLY`, `ECORE_ONLY`, `ANY_CORE` | unset |
| `VLLM_OPENVINO_FAST_SAMPLER` | CPU only. Replace vLLM V1's full-vocabulary random sample, top-k/top-p sort, and dense repetition-penalty routines with partial/sparse equivalents (up to ~2× end-to-end on small models with default `top_k`/`top_p`/`repetition_penalty`). Set to `0` to use stock vLLM. See [Sampling Parameters](#sampling-parameters-large-vocab-models) for seed behavior. | `1` |
| `TORCH_COMPILE_DISABLE` | Must be set to 1; `torch.compile` is incompatible with OpenVINO. | — |

## Performance Tuning

For CPU deployments, especially AVX2-only systems, tuning OpenVINO CPU threading/stream properties can improve sustained tokens/sec.

### Memory Footprint

The KV cache is reserved as one fixed-size mapping and pages become resident only as blocks are used, so RSS climbs toward `baseline + VLLM_OPENVINO_KVCACHE_SPACE` and then plateaus; size the container for that sum (baseline is roughly the model plus ~1.5 GB of runtime for small models). vLLM's prefix caching (on by default) retains freed blocks, so unique-prompt workloads keep touching new pool pages — add `--no-enable-prefix-caching` to hold the resident set at the working set. On CPU the pool is also sized against the container memory limit: if it does not fit alongside the model weights, a 1.5 GB runtime baseline and 512 MB headroom, the plugin reduces it automatically and logs the new size, and it warns when under 10% headroom remains; run without a cgroup memory limit and the check is skipped.

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

Caveat: on this plugin, `NUM_STREAMS=AUTO` under `THROUGHPUT` fragments threads across streams that vLLM V1 never runs concurrently — the runner holds a single `create_infer_request` and issues a single blocking `infer` per step. Setting `NUM_STREAMS=1` explicitly measured a 3.56× throughput gain on Qwen3-1.7B-int4-ov at concurrency 8 (8.53 → 30.38 tok/s), which is why the default is `1`. The same holds on hybrid-mamba models via the Hybrid-PA path: at concurrency 8 with `VLLM_OPENVINO_CPU_THREADS_NUM=24`, `NUM_STREAMS=1` vs `AUTO` measured 59.40 → 142.71 tok/s (2.40×) on LFM2.5-350M-int8-ov and 25.39 → 67.83 tok/s (2.67×) on Qwen3.5-0.8B-int4-ov (text-only) — vLLM V1 is serial regardless of model architecture, so stream fragmentation hurts dense and hybrid alike.

**Multi-socket placement**: OpenVINO binds threads to every core the process can see, including cores on a second socket. Measured on a 2-socket Xeon E5-2670 v3 (12 physical cores per socket) with Qwen3.5-0.8B-int4-ov, 90 s guidellm runs:

| Threads | Sockets | Output tok/s, 2 streams | Output tok/s, 8 streams |
|---------|---------|-------------------------|-------------------------|
| 8 | node0 only | 14.7 | 22.7 |
| 8 | node0 + node1 (4+4) | 13.0 (−12%) | 18.6 (−18%) |
| 12 | node0 only | 18.5 | 27.6 |
| 24 | node0 + node1 (12+12) | 18.7 (+1%) | 30.0 (+9%) |

Splitting a fixed thread budget across sockets costs 12–18% because the threads on the far socket read the weights over the interconnect, and doubling the thread count across both sockets buys at most ~9% and only under load. `MPOL_INTERLEAVE` across both nodes (`numactl --interleave=all` or the equivalent `set_mempolicy` syscall) does not close the gap — 29.7 vs 30.0 tok/s on the same 24-thread dual-socket run — so the bottleneck is memory bandwidth, not weight locality. Keep inference threads inside one socket by binding the process to that socket's CPU set and setting `VLLM_OPENVINO_CPU_THREADS_NUM` to its core count.

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

### Sampling Parameters (large-vocab models)

Sampling runs on the critical path of every decode step. For small models it costs as much as the OpenVINO inference, so the plugin replaces three of vLLM's CPU sampling routines (see the fast-sampler paragraph below). `temperature=0` (greedy) skips top-k/top-p and all randomness entirely, but still applies the repetition penalty when the model's `generation_config.json` sets one (Qwen2.5-Coder ships `repetition_penalty=1.1`, `top_k=20`, `top_p=0.8`).

**Upstream vLLM V1 `temperature=0` all-greedy fast path**: `Sampler.sample()` (`vllm/v1/sample/sampler.py:257-271`) early-returns after a bare `logits.argmax(dim=-1)` when every request in the batch is greedy (`SamplingType.GREEDY`, which vLLM sets when `temperature<=1e-5`). That skips `apply_temperature`, argmax-invariant logits processors, the full `TopKTopPSampler` pipeline (softmax → `exponential_()` Gumbel-max noise → `probs.div(q).argmax`), and the final `torch.where` merge. Empirical on LFM2.5-350M-int8-ov at concurrency 8, `NUM_STREAMS=1`, `CPU_THREADS_NUM=24`, `max_tokens=128`, 3-run mean: temp=0 **120.26 tok/s** vs temp=0.7 **86.47 tok/s** — temp=0 is **1.39× faster aggregate** and **1.47× faster per-longest-request** (128/wave wall). The fast path is inherited from upstream unchanged; the plugin registers no custom sampler.

**Plugin CPU fast-sampler (`VLLM_OPENVINO_FAST_SAMPLER`, default `1`)**: `sampler_patch.py` replaces three upstream routines whose cost scales with the full `(batch, vocab)` tensor even though the result depends on a handful of tokens. (1) `compiled_random_sample` drew a full-vocabulary exponential noise buffer (Gumbel-max); a categorical draw needs one uniform number per row, so the patch uses softmax -> cumsum -> `searchsorted(right=True)`, which never selects a zero-probability token. (2) `apply_top_k_top_p` sorted the whole vocabulary whenever `top_p` was set; the patch takes a partial `topk` over `max(k)` candidates (1024 when only `top_p` is set) and computes the nucleus on them, falling back to upstream when the candidates hold less than `p` of the mass. (3) `apply_all_penalties` built dense `(batch, vocab)` bin-count and mask tensors; the repetition penalty is applied only at prompt/output token positions, and batches using presence/frequency penalties take the upstream path. Output equivalence against the upstream functions was checked on random and peaked logits: identical kept-token sets and bit-identical values for top-k/top-p and for penalties. Micro-benchmark at batch 8, vocab 151,936: sample 7.7 -> 1.1 ms, top-k+top-p 17.8 -> 0.9 ms (top-p only on peaked logits 17.9 -> 1.2 ms), repetition penalty 3.3 -> 0.3 ms. End to end on Qwen2.5-Coder-0.5B-int4-ov (PagedAttention, 8-CPU quota, `max_tokens=128`, `ignore_eos`, 3-run mean, model-default `top_k=20`/`top_p=0.8`/`repetition_penalty=1.1`): concurrency 1 greedy 48.0 -> 68.6 tok/s and temp 0.7 39.6 -> 51.0; concurrency 8 greedy 284 -> 366 and temp 0.7 183 -> 345 (+89%). The step is then ~20 ms of OpenVINO inference plus ~3 ms of sampling. Hybrid-PA Qwen3.5-0.8B-int4-ov gains less because inference dominates: concurrency 8 temp 0.7 118.5 -> 128.1 tok/s, greedy unchanged. Seed: a per-request `seed` creates a `torch.Generator` for that request (verified on Qwen2.5-Coder-0.5B-int4-ov: three `seed=42`, temp=1.0 requests returned identical text, unseeded requests differed). A batch containing any seeded request bypasses the random-sample patch and takes the stock per-generator path; batches with no seeded request use the fast path, whose token stream is not reproducible. temp=0 output is unchanged. Set `VLLM_OPENVINO_FAST_SAMPLER=0` to fall back to stock vLLM.

**Foreign-script tokens in Korean output (Qwen3.5-0.8B-int4-ov)**: the stock and fast samplers draw from the same distribution. At `temperature=1.0` with no `top_k`/`top_p` a 0.8B int4 model still emits stray CJK/Cyrillic/Arabic characters (about 25–70 per 1k tokens, 16–32 Korean prompts, concurrency 8). The model's `generation_config.json` carries no sampling defaults, so vLLM samples the full vocabulary. Greedy and `temperature=0.7, top_k=20, top_p=0.8` (Qwen's non-thinking recommendation) produced 0 foreign characters over 4k tokens each; pass those parameters per request for small Qwen3.5 models.

**Vision-embedding cache**: for multimodal models (Qwen3.5-VL, Gemma-4), the plugin caches merged vision embeddings keyed by `mm_feature.identifier` (a stable blake3 content hash of the image). The cache is an LRU bounded at 8 entries (~32 MB max), lives in `OpenVINOCausalLM._vision_embed_cache`, and is transparent to the caller — no configuration needed. When the same image appears in multiple requests (concurrent duplicates, multi-turn with a different prefix, post-eviction re-prefill), the OpenVINO vision tower + merger are skipped entirely. Measured on Qwen3.5-0.8B-int4-ov (Hybrid-PA, 1024×1024 image, 1024 merged tokens): TTFT dropped from 11.82 s (cache miss) to 6.46 s (cache hit) — 1.83× faster. For small images (64 merged tokens) the gain is smaller (0.72 s → 0.60 s). Sanity check on Gemma-4-E2B-int4-ov stateful path (max_num_seqs=1, 256 merged tokens; varying the prompt text so KV prefix cache misses on the tail but the mm_hash cache still hits — confirmed by `[OV-VISION] Cache hit for mm_hash=...` logs at DEBUG): TTFT stays near 4.2 s across miss and hits, i.e. **~1.0×** speedup. On a small stateful language model with a modest merged-token count, the vision encoder is a tiny share of TTFT compared to the language-model prefill, so caching it saves little wall time — the cache is still correct and free, just not load-bearing here. The cache complements vLLM's KV prefix caching: when the image-containing prefix is already cached in KV blocks, vision is skipped regardless; the embedding cache catches the cases where the prefix differs but the image is the same.

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

Models without `ScaledDotProductAttention` ops, or with sliding-window attention smaller than `max_model_len` (Gemma-4), run via OpenVINO's internal state management (`ReadValue`/`Assign`). Characteristics:
- Sequential request processing (`max_num_seqs=1`)
- Internal KV cache managed by OpenVINO runtime
- Automatic detection and configuration — no manual flags needed

Other `ReadValue`-based stateful models (optimum-intel's default exports such as Qwen2.5-Coder-int4-ov) get the PagedAttention transformation at load time by default (`VLLM_OPENVINO_STATEFUL_PA=1`), keeping `max_num_seqs` at its default 128. Validated on four dense architectures (Qwen2.5-Coder-0.5B, Qwen3-1.7B, TinyLlama-1.1B, Phi-3.5-mini; int4 IR): concurrency-8 aggregate throughput was 4.9–7.0x the stateful path, with a single-stream delta of -7.8% to +4%. A configured `sliding_window` at or above `max_model_len` (Phi-3.5-mini: 262144) never takes effect, so such models are not treated as sliding-window models. The transformed model's KV cache defaults to u8; `VLLM_OPENVINO_KV_CACHE_PRECISION` is honored for `u8`/`f16`/`bf16` (`f32`/`i8` are unsupported by CPU PagedAttention and fall back to the default with a warning). Greedy output was unchanged on TinyLlama-1.1B but can differ from the stateful path on Qwen-family models — a property of the PagedAttention-transformed attention computation, not of cache precision (u8 and bf16 produced identical output). Set `VLLM_OPENVINO_STATEFUL_PA=0` to force the sequential stateful path.

### Hybrid-PA (default for hybrid Mamba/attention models)

For hybrid Mamba/attention models (Qwen3.5, LFM2.5), this converts attention layers to real PagedAttention and conv/SSM layers to a separate linear-attention paged-state mechanism, enabling concurrent request batching (`max_num_seqs > 1`) instead of the sequential stateful path. Verified on LFM2.5 (conv-only) and Qwen3.5 (conv + GatedDeltaNet SSM) with byte-exact output vs. the stateful path. Not currently supported for Gemma-4 (sliding-window attention layers hit a shape mismatch) — Gemma-4 has no SSM/conv state, so it's never selected for this path. Set `VLLM_OPENVINO_HYBRID_PA=0` to force the sequential stateful path instead.

## Compatibility

The following vLLM features are compatible with the OpenVINO backend:

- Chunked prefill (`--enable-chunked-prefill`)
- Structured outputs (`structured_outputs`, `response_format`, forced tool calls) via the grammar bitmask
- Gemma 3 and Gemma 4 text and multimodal (text + image)
- Qwen3.5 and LFM2.5 (hybrid Mamba/attention architecture, via Hybrid-PA)

## Limitations

- LoRA serving is not supported.
- Pin memory is not supported.
- `prompt_logprobs` (and `echo` with `logprobs`) is not supported; such requests are rejected with HTTP 400 because logits are computed only at sampled positions. Output-token `logprobs` work.
- Tensor/pipeline parallelism is not supported. Threads are scheduled across all visible cores, but multi-socket scaling is poor; see [CPU Tuning](#cpu-tuning-avx2).
- vLLM V1 engine only.
- Stateful-path models (e.g. Gemma-4) do not support concurrent request execution (`max_num_seqs=1`).

See `docs/compatibility.md` for the current support matrix.
