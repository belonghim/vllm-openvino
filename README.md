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
| `VLLM_OPENVINO_PERFORMANCE_MODE` | Performance mode: LATENCY or THROUGHPUT | `LATENCY` |
| `VLLM_OPENVINO_CPU_THREADS_NUM` | CPU only. Inference threads (`0` = auto: cgroup CPU quota if constrained, else OpenVINO auto) | `0` |
| `VLLM_OPENVINO_NUM_STREAMS` | CPU only. Inference streams: `AUTO` or integer | `AUTO` |
| `VLLM_OPENVINO_ENABLE_HYPER_THREADING` | CPU only. Enable/disable hyperthreading: `true`, `false`, or `auto` | `auto` |
| `VLLM_OPENVINO_INFERENCE_PRECISION` | CPU only. Force inference precision: `f32`, `f16`, `bf16` (unset = OpenVINO default) | unset |
| `VLLM_OPENVINO_ENABLE_CPU_PINNING` | CPU only. Enable/disable CPU core pinning: `true`, `false`, or `auto` | `auto` |
| `VLLM_OPENVINO_HYBRID_PA` | Default path for hybrid Mamba/attention models: attempt PagedAttention (concurrent batching) instead of the sequential stateful path. Set to `0` to force the sequential stateful path instead. See [Serving Modes](#serving-modes). | `1` |
| `VLLM_OPENVINO_STATEFUL_PA` | PagedAttention transformation for plain-attention stateful models (optimum-intel default exports), enabling concurrent batching. Sliding-window models and models without SDPA ops fall back to the sequential stateful path. Set to `0` to force the sequential stateful path. See [Serving Modes](#serving-modes). | `1` |
| `VLLM_OPENVINO_CACHE_DIR` | Directory for OpenVINO's compiled-model disk cache. When set, skips recompiling the model on process restart if a matching cached blob exists. Unset by default since not every deployment has a writable, persistent path. | unset |
| `VLLM_OPENVINO_SCHEDULING_CORE_TYPE` | CPU only. Scheduling core type on hybrid P-core/E-core CPUs: `PCORE_ONLY`, `ECORE_ONLY`, `ANY_CORE` | unset |
| `TORCH_COMPILE_DISABLE` | Must be set to 1; `torch.compile` is incompatible with OpenVINO. | — |

## Performance Tuning

For CPU deployments, especially AVX2-only systems, tuning OpenVINO CPU threading/stream properties can improve sustained tokens/sec.

### Memory Footprint

The KV cache is reserved as one fixed-size mapping and pages become resident only as blocks are used, so RSS climbs toward `baseline + VLLM_OPENVINO_KVCACHE_SPACE` and then plateaus; size the container for that sum (baseline is roughly the model plus ~1.5 GB of runtime for small models). vLLM's prefix caching (on by default) retains freed blocks, so unique-prompt workloads keep touching new pool pages — add `--no-enable-prefix-caching` to hold the resident set at the working set. On CPU the pool is also sized against the container memory limit: if it does not fit alongside the model weights, a 1.5 GB runtime baseline and 512 MB headroom, the plugin reduces it automatically and logs the new size, and it warns when under 10% headroom remains; run without a cgroup memory limit and the check is skipped.

### KV Cache Quantization

The KV cache precision can be reduced to significantly lower memory usage:

| Precision | Memory | Notes |
|-----------|--------|-------|
| `u8` | Lowest | 8-bit unsigned integer; fastest, smallest footprint |
| `i8` | Low | 8-bit signed integer |
| `f16` / `bf16` | Medium | Default on most GPUs; good balance |
| `f32` | Highest | Best accuracy, highest memory usage |

Set via environment variable:
```bash
VLLM_OPENVINO_KV_CACHE_PRECISION=u8 \
  vllm serve --model <model_id>
```

### CPU Tuning (AVX2)

On AVX2-only CPUs, int4 models usually show a larger throughput gap vs AVX-512/VNNI capable CPUs due to lower effective low-precision compute throughput. In practice, CPU scheduling knobs (threads, affinity, streams) are often the main software lever for improving throughput stability.

For older AVX2 systems, fp16 or int8 models are often a better latency/throughput trade-off than int4.

| Variable | Type | Values | Effect |
|----------|------|--------|--------|
| `VLLM_OPENVINO_CPU_THREADS_NUM` | int | `0` (auto), `1..N` | Caps OpenVINO CPU inference threads |

**cgroup-aware auto-detection**: container runtimes (podman/docker `--cpus`, Kubernetes CPU limits) throttle via cgroup CFS quota, but `os.cpu_count()` inside the container still reports the host's full core count. With `VLLM_OPENVINO_CPU_THREADS_NUM=0` (default), the plugin detects the quota and caps both the OpenVINO inference threads and the Torch/OMP thread pools to it when the quota is tighter than the visible core count; an explicit `OMP_NUM_THREADS` is left untouched. On an 8-quota/24-visible-core host, a 32-request burst went from 4.5s wall / 35.6s CPU with 3.1s of CFS throttling to 3.6s wall / 26.6s CPU with none, and the engine process dropped from 147 to 126 threads (earlier OpenVINO-only measurement: 45s/~355% CPU uncapped vs 34s/~300% capped for a 4-request burst). Set `VLLM_OPENVINO_CPU_THREADS_NUM` explicitly to override this detection.
| `VLLM_OPENVINO_NUM_STREAMS` | str/int | `AUTO`, `1..N` | Controls number of parallel CPU inference streams |
| `VLLM_OPENVINO_ENABLE_HYPER_THREADING` | bool | `true`, `false`, `auto` | Disabling prevents HT oversubscription on 2-socket systems |
| `VLLM_OPENVINO_INFERENCE_PRECISION` | str | `f32`, `f16`, `bf16` (unset = OpenVINO default) | Forces specific precision for matmul operations |
| `VLLM_OPENVINO_ENABLE_CPU_PINNING` | bool | `true`, `false`, `auto` | Controls thread-to-core pinning |

`PERFORMANCE_MODE` measured on Qwen2.5-Coder-0.5B-int4-ov (8-CPU quota): THROUGHPUT gave +28–31% single-stream and +10% concurrency-8 aggregate decode throughput on the PagedAttention path over LATENCY (and +28% single-stream on the stateful path), at a higher first-token latency (25 ms → 34 ms). Prefer `THROUGHPUT` for serving throughput, `LATENCY` for interactive first-token response.

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

On CPU, vLLM's default top-k/top-p sampling sorts the full logits vector every step. For large-vocab models (e.g. Gemma), this sort can cost more CPU time per step than the model's own OpenVINO inference call. `temperature=0` (greedy) skips the sort and all randomness entirely. To keep randomness but still skip the sort, set both `top_k=-1` and `top_p=1.0` explicitly — some models (e.g. Gemma) ship a `generation_config.json` with non-default top_k/top_p, so both must be overridden together.

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

Models without `ScaledDotProductAttention` ops, or with sliding-window attention (Gemma-4, Phi-3.5-mini), run via OpenVINO's internal state management (`ReadValue`/`Assign`). Characteristics:
- Sequential request processing (`max_num_seqs=1`)
- Internal KV cache managed by OpenVINO runtime
- Automatic detection and configuration — no manual flags needed

Other `ReadValue`-based stateful models (optimum-intel's default exports such as Qwen2.5-Coder-int4-ov) get the PagedAttention transformation at load time by default (`VLLM_OPENVINO_STATEFUL_PA=1`), keeping `max_num_seqs` at its default 128. Validated on three dense architectures (Qwen2.5-Coder-0.5B, Qwen3-1.7B, TinyLlama-1.1B; int4 IR): concurrency-8 aggregate throughput was 4.9–7.0x the stateful path, with a single-stream delta of -1.2% to -7.8%. The transformed model's KV cache defaults to u8; `VLLM_OPENVINO_KV_CACHE_PRECISION` is honored for `u8`/`f16`/`bf16` (`f32`/`i8` are unsupported by CPU PagedAttention and fall back to the default with a warning). Greedy output was unchanged on TinyLlama-1.1B but can differ from the stateful path on Qwen-family models — a property of the PagedAttention-transformed attention computation, not of cache precision (u8 and bf16 produced identical output). Set `VLLM_OPENVINO_STATEFUL_PA=0` to force the sequential stateful path.

### Hybrid-PA (default for hybrid Mamba/attention models)

For hybrid Mamba/attention models (Qwen3.5, LFM2.5), this converts attention layers to real PagedAttention and conv/SSM layers to a separate linear-attention paged-state mechanism, enabling concurrent request batching (`max_num_seqs > 1`) instead of the sequential stateful path. Verified on LFM2.5 (conv-only) and Qwen3.5 (conv + GatedDeltaNet SSM) with byte-exact output vs. the stateful path. Not currently supported for Gemma-4 (sliding-window attention layers hit a shape mismatch) — Gemma-4 has no SSM/conv state, so it's never selected for this path. Set `VLLM_OPENVINO_HYBRID_PA=0` to force the sequential stateful path instead.

## Compatibility

The following vLLM features are compatible with the OpenVINO backend:

- Chunked prefill (`--enable-chunked-prefill`)
- Gemma 3 and Gemma 4 text and multimodal (text + image)
- Qwen3.5 and LFM2.5 (hybrid Mamba/attention architecture, via Hybrid-PA)

## Limitations

- LoRA serving is not supported.
- Pin memory is not supported.
- Structured outputs are not supported.
- Tensor/pipeline parallelism is not supported. Threads are scheduled across all visible cores, but multi-socket scaling is poor; see [CPU Tuning](#cpu-tuning-avx2).
- vLLM V1 engine only.
- Stateful-path models (e.g. Gemma-4) do not support concurrent request execution (`max_num_seqs=1`).

See `docs/compatibility.md` for the current support matrix.
