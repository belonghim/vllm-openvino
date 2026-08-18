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
podman run -d --name vllm-server -p 8000:8000 \
  -e VLLM_OPENVINO_DEVICE=CPU \
  -e TORCH_COMPILE_DISABLE=1 \
  -e VLLM_OPENVINO_KVCACHE_SPACE=8 \
  quay.io/joopark/vllm-openvino \
  --model <model_id>
```

## Quick Start

Run the vLLM API server with OpenVINO backend.

For CPU:

```bash
VLLM_OPENVINO_DEVICE=CPU TORCH_COMPILE_DISABLE=1 VLLM_OPENVINO_KVCACHE_SPACE=8 \
  python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

For GPU:

```bash
VLLM_OPENVINO_DEVICE=GPU TORCH_COMPILE_DISABLE=1 \
  python -m vllm.entrypoints.openai.api_server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

Replace `TinyLlama/TinyLlama-1.1B-Chat-v1.0` with a local path to pre-exported OpenVINO IR files (directory containing openvino_model.xml and openvino_model.bin).

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_OPENVINO_DEVICE` | Device selection: CPU, GPU, GPU.1, etc. | `CPU` |
| `VLLM_OPENVINO_KVCACHE_SPACE` | KV cache size in GB (0 = auto: 4 GB on CPU) | `0` |
| `VLLM_OPENVINO_KV_CACHE_PRECISION` | KV cache dtype: u8, i8, f16, bf16, f32 | `auto` |
| `VLLM_OPENVINO_PERFORMANCE_MODE` | Performance mode: LATENCY or THROUGHPUT | `LATENCY` |
| `VLLM_OPENVINO_CPU_THREADS_NUM` | CPU only. Inference threads (`0` = auto: cgroup CPU quota if constrained, else OpenVINO auto) | `0` |
| `VLLM_OPENVINO_CPU_BIND_THREAD` | CPU only. Thread affinity: `CORE`, `NUMA`, `NONE` | unset |
| `VLLM_OPENVINO_NUM_STREAMS` | CPU only. Inference streams: `AUTO` or integer | `AUTO` |
| `VLLM_OPENVINO_ENABLE_HYPER_THREADING` | CPU only. Enable/disable hyperthreading: `true` or `false` | `auto` |
| `VLLM_OPENVINO_INFERENCE_PRECISION` | CPU only. Force inference precision: `f32`, `f16`, `bf16` | `auto` |
| `VLLM_OPENVINO_ENABLE_CPU_PINNING` | CPU only. Enable/disable CPU core pinning: `true` or `false` | `auto` |
| `VLLM_OPENVINO_HYBRID_PA` | Experimental. Attempt PagedAttention (concurrent batching) for hybrid Mamba/attention models instead of the default sequential stateful path. See [Serving Modes](#serving-modes). | `0` |
| `TORCH_COMPILE_DISABLE` | Must be set to 1; `torch.compile` is incompatible with OpenVINO. | — |

## Performance Tuning

For CPU deployments, especially AVX2-only systems, tuning OpenVINO CPU threading/stream properties can improve sustained tokens/sec.

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
  python -m vllm.entrypoints.openai.api_server --model <model_id>
```

### CPU Tuning (AVX2)

On AVX2-only CPUs, int4 models usually show a larger throughput gap vs AVX-512/VNNI capable CPUs due to lower effective low-precision compute throughput. In practice, CPU scheduling knobs (threads, affinity, streams) are often the main software lever for improving throughput stability.

For older AVX2 systems, fp16 or int8 models are often a better latency/throughput trade-off than int4.

| Variable | Type | Values | Effect |
|----------|------|--------|--------|
| `VLLM_OPENVINO_CPU_THREADS_NUM` | int | `0` (auto), `1..N` | Caps OpenVINO CPU inference threads |

**cgroup-aware auto-detection**: container runtimes (podman/docker `--cpus`, Kubernetes CPU limits) throttle via cgroup CFS quota, but `os.cpu_count()` inside the container still reports the host's full core count. With `VLLM_OPENVINO_CPU_THREADS_NUM=0` (default), the plugin now detects the cgroup quota at model-load time and caps OpenVINO's thread pool to it when the quota is tighter than the visible core count — preventing thread oversubscription. Measured on an 8-quota/24-visible-core host: uncapped auto-detection spawned 91 threads and took 45s/~355% CPU for a 4-request burst; quota-aware capping to 8 threads took 34s/~300% CPU for the same workload (25% faster, lower CPU). Set `VLLM_OPENVINO_CPU_THREADS_NUM` explicitly to override this detection.
| `VLLM_OPENVINO_CPU_BIND_THREAD` | str | `CORE`, `NUMA`, `NONE` | Controls CPU thread affinity policy |
| `VLLM_OPENVINO_NUM_STREAMS` | str/int | `AUTO`, `1..N` | Controls number of parallel CPU inference streams |
| `VLLM_OPENVINO_ENABLE_HYPER_THREADING` | bool | `true`, `false`, `auto` | Disabling prevents HT oversubscription on 2-socket systems |
| `VLLM_OPENVINO_INFERENCE_PRECISION` | str | `f32`, `f16`, `bf16`, `auto` | Forces specific precision for matmul operations |
| `VLLM_OPENVINO_ENABLE_CPU_PINNING` | bool | `true`, `false`, `auto` | Controls thread-to-core pinning |

Example (latency-optimized for AVX2, 2-socket Xeon):

```bash
VLLM_OPENVINO_DEVICE=CPU \
VLLM_OPENVINO_PERFORMANCE_MODE=LATENCY \
VLLM_OPENVINO_CPU_THREADS_NUM=24 \
VLLM_OPENVINO_CPU_BIND_THREAD=CORE \
VLLM_OPENVINO_NUM_STREAMS=1 \
VLLM_OPENVINO_ENABLE_HYPER_THREADING=false \
TORCH_COMPILE_DISABLE=1 \
python -m vllm.entrypoints.openai.api_server --model <model_id>
```

### Memory-Mapped Model Loading

OpenVINO automatically memory-maps model weights, reducing RAM usage during model loading by mapping weights directly from disk rather than copying them into memory. No configuration is required.

### Benchmarking

To measure throughput/latency improvements, use the provided benchmark script:

```bash
./scripts/benchmark.sh <model_path> [num_requests]
```

The script runs a warmed-up benchmark against the local OpenAI-compatible endpoint and reports tokens/sec.

## Serving Modes

The plugin supports three serving paths depending on the model architecture:

### PagedAttention (default)

Models with `ScaledDotProductAttention` ops and no state (e.g., Llama 3, Qwen2.5) are transformed to use vLLM's PagedAttention mechanism. This enables:
- Concurrent request batching
- External KV cache management
- Full vLLM scheduler features

### Stateful Path (default for hybrid/Mamba models)

Models without SDPA ops (Gemma-4) or with hybrid Mamba/attention layers (Qwen3.5, LFM2.5) run via OpenVINO's internal state management (`ReadValue`/`Assign`) by default. Characteristics:
- Sequential request processing (`max_num_seqs=1`)
- Internal KV cache managed by OpenVINO runtime
- Automatic detection and configuration — no manual flags needed

### Hybrid-PA (experimental, opt-in via `VLLM_OPENVINO_HYBRID_PA=1`)

For hybrid Mamba/attention models, this converts attention layers to real PagedAttention and conv/SSM layers to a separate linear-attention paged-state mechanism, enabling concurrent request batching (`max_num_seqs > 1`) instead of the sequential stateful path. Verified on LFM2.5 (conv-only) and Qwen3.5 (conv + GatedDeltaNet SSM) with byte-exact output vs. the stateful path. Not currently supported for Gemma-4 (sliding-window attention layers hit a shape mismatch). Off by default — new and less battle-tested than the stateful path.

## Compatibility

The following vLLM features are compatible with the OpenVINO backend:

- Chunked prefill (`--enable-chunked-prefill`)
- Gemma 3 and Gemma 4 text and multimodal (text + image)
- Qwen3.5 (hybrid Mamba/attention architecture)

## Limitations

- LoRA serving is not supported.
- Single socket only; tensor/pipeline parallelism is not supported.
- vLLM V1 engine only.

See `docs/compatibility.md` for the current support matrix.
