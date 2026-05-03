# vllm-openvino

[GitHub](https://github.com/belonghim/vllm-openvino) · OpenVINO plugin for vLLM — run LLM inference on Intel CPUs and GPUs.

## What is this?

This project provides an OpenVINO backend for vLLM, allowing you to run vLLM's OpenAI-compatible API server on Intel CPUs and GPUs. It integrates OpenVINO as the inference execution layer, leveraging vLLM's scheduler, PagedAttention, and API server infrastructure. Models must be pre-exported to OpenVINO IR format (openvino_model.xml + openvino_model.bin).

## Requirements

- Python >= 3.10
- vLLM 0.19.1
- OpenVINO >= 2026.1.0
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
docker build -f Containerfile -t vllm-openvino .
```

Run the Docker container:

```bash
docker run -d --name vllm-server -p 8000:8000 \
  -e VLLM_OPENVINO_DEVICE=CPU \
  -e TORCH_COMPILE_DISABLE=1 \
  -e VLLM_OPENVINO_KVCACHE_SPACE=8 \
  vllm-openvino \
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
| `VLLM_OPENVINO_CPU_THREADS_NUM` | CPU only. Inference threads (`0` = OpenVINO auto) | `0` |
| `VLLM_OPENVINO_CPU_BIND_THREAD` | CPU only. Thread affinity: `CORE`, `NUMA`, `NONE` | unset |
| `VLLM_OPENVINO_NUM_STREAMS` | CPU only. Inference streams: `AUTO` or integer | `AUTO` |
| `VLLM_OPENVINO_ENABLE_HYPER_THREADING` | CPU only. Enable/disable hyperthreading: `true` or `false` | `auto` |
| `VLLM_OPENVINO_INFERENCE_PRECISION` | CPU only. Force inference precision: `f32`, `f16`, `bf16` | `auto` |
| `VLLM_OPENVINO_ENABLE_CPU_PINNING` | CPU only. Enable/disable CPU core pinning: `true` or `false` | `auto` |
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

OpenVINO automatically memory-maps model weights since 2026.0+. This reduces RAM usage during model loading by mapping weights directly from disk rather than copying them into memory. No configuration is required.

### Benchmarking

To measure throughput/latency improvements, use the provided benchmark script:

```bash
./scripts/benchmark.sh <model_path> [num_requests]
```

The script runs a warmed-up benchmark against the local OpenAI-compatible endpoint and reports tokens/sec.

## Compatibility

The following vLLM features are compatible with the OpenVINO backend:

- Chunked prefill (`--enable-chunked-prefill`)
- Gemma 4 text and multimodal (text + image)

## Limitations

- LoRA serving is not supported.
- Stateful models (e.g., Gemma-4 without PA transformation) require `max_num_seqs=1` as they do not support batched inference.
- Single socket only; tensor/pipeline parallelism is not supported.
- vLLM V1 engine only (vLLM 0.19.1).
