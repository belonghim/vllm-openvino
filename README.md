# vllm-openvino

[GitHub](https://github.com/belonghim/vllm-openvino) · OpenVINO plugin for vLLM — run LLM inference on Intel CPUs.

## What is this?

An OpenVINO backend for vLLM: vLLM's scheduler and OpenAI-compatible API server run OpenVINO IR models on Intel CPUs. On the same IR and 8 CPUs it matched OpenVINO Model Server up to 8 concurrent requests and kept batching past 8 (Qwen3-1.7B, 16 concurrent: 254 vs 146-157 tok/s), and single-stream decode was 1.5x faster than vLLM's own CPU backend on a w4a16 checkpoint. See `docs/compatibility.md` ("Comparison with OVMS and vLLM's CPU Backend").

Models must be pre-exported to OpenVINO IR with [optimum-intel](https://github.com/huggingface/optimum-intel). GPU code paths exist (`VLLM_OPENVINO_DEVICE=GPU`) but are not validated; every measurement here is CPU-only.

## Requirements

- Linux x86-64 with AVX2 or newer
- Python >= 3.11, vLLM 0.34.0, OpenVINO >= 2026.4.1

## Installation

### Container

```bash
podman build -f Containerfile -t quay.io/joopark/vllm-openvino .

podman run -d --name vllm-server -p 8080:8080 \
  -v ~/models:/models:Z \
  -v ~/ov-cache:/cache:Z -e VLLM_OPENVINO_CACHE_DIR=/cache \
  quay.io/joopark/vllm-openvino \
  --port=8080 --model /models/<model_dir>
```

`--model` must be a local directory with the IR files (`openvino_model.xml` for text models, `openvino_language_model.xml` for multimodal ones); the image runs with `HF_HUB_OFFLINE=1` and never downloads weights. Mounting `VLLM_OPENVINO_CACHE_DIR` keeps the compiled model across restarts (otherwise every start recompiles, 22 s for a 14 GB IR).

### From source

```bash
VLLM_TARGET_DEVICE="empty" PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" pip install .
pip uninstall -y triton   # vLLM may pull in triton, which is incompatible with OpenVINO
TORCH_COMPILE_DISABLE=1 OMP_WAIT_POLICY=PASSIVE vllm serve --model /models/<model_dir>
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VLLM_OPENVINO_DEVICE` | `CPU`, `GPU`, `GPU.1`, ... | `CPU` |
| `VLLM_OPENVINO_KVCACHE_SPACE` | KV cache size in GiB, fractions allowed. `0` sizes the pool from the memory limit after model load (see [Memory](#memory)) | `0` |
| `VLLM_OPENVINO_KV_CACHE_PRECISION` | `u8`, `f16`, `bf16`; unset = OpenVINO's default (`u8` on CPU). `bf16` fails at startup on CPUs without AVX-512-BF16 (`expect kvcache type f32`); use `f16` there | unset |
| `VLLM_OPENVINO_PERFORMANCE_MODE` | `THROUGHPUT` or `LATENCY` | `THROUGHPUT` |
| `VLLM_OPENVINO_CPU_THREADS_NUM` | Inference threads; `0` = the cgroup CPU quota if constrained, else OpenVINO's choice | `0` |
| `VLLM_OPENVINO_NUM_STREAMS` | Inference streams. Keep `1`: vLLM runs one blocking `infer` per step, so more streams only split the threads | `1` |
| `VLLM_OPENVINO_ENABLE_HYPER_THREADING` | `true`, `false`, `auto` | `auto` |
| `VLLM_OPENVINO_INFERENCE_PRECISION` | `f32`, `f16`, `bf16`; unset = OpenVINO's default | unset |
| `VLLM_OPENVINO_ENABLE_CPU_PINNING` | `true`, `false`, `auto` | `auto` |
| `VLLM_OPENVINO_DYNAMIC_QUANTIZATION_GROUP_SIZE` | Activation quantization group for compressed matmuls: `64`/`128` give +17% / +23-32% prefill at a small, unconfirmed accuracy cost; `0` disables it. Unset = OpenVINO's 32 | unset |
| `VLLM_OPENVINO_SCHEDULING_CORE_TYPE` | Hybrid P/E-core CPUs: `PCORE_ONLY`, `ECORE_ONLY`, `ANY_CORE` | unset |
| `VLLM_OPENVINO_LA_CACHE_INTERVAL_MULTIPLIER` | Hybrid-PA prefix caching block in 32-token KV blocks (8 = 256 tokens). `0` = OpenVINO GenAI's automatic rule. Smaller = finer prefix hits, more memory per token | `8` |
| `VLLM_OPENVINO_MTP_DRAFT_VOCAB` | MTP draft lm_head covers token ids below this value; changes speed, not output. `0` = whole vocabulary | `32768` |
| `VLLM_OPENVINO_FAST_SAMPLER` | Partial top-k/top-p and sparse repetition penalty instead of full-vocabulary routines (same output) | `1` |
| `VLLM_OPENVINO_CACHE_DIR` | Compiled-model disk cache; unset = `~/.cache/vllm-openvino` if writable | auto |
| `TORCH_COMPILE_DISABLE` | Must be `1` (set in the image); torch.compile is incompatible with OpenVINO | — |
| `OMP_WAIT_POLICY` | `PASSIVE` (set in the image) stops idle Torch OpenMP threads from spinning on the CPUs OpenVINO needs: +10-20% at 4 concurrent requests on 4 CPUs. `OMP_NUM_THREADS=1` has the same effect but halves large-batch sampling speed | — |

## Performance

Measured on a Ryzen 9 7900 (no AMX) with an 8-CPU quota and int4 IRs; details in `docs/compatibility.md` ("Performance Profile").

- **Single-stream decode is memory-bandwidth bound** and needs only about 4 CPUs. int4 IRs are the fastest format because they read the fewest bytes.
- **Batch serving is compute bound.** Dense models scale almost linearly to 8-16 concurrent requests and gain little past 32 (Qwen3-1.7B: 29 tok/s at 1, 327 at 16, 532 at 128). Give batch serving more CPUs; MoE models scale worse.
- **Prefill is compute bound** (Qwen3-1.7B ~840 tok/s at 1k tokens, 403 at 17.5k) and 2.2x slower on AVX2-only CPUs, where single-stream decode is unchanged.
- Other OpenVINO CPU properties were best at their defaults, apart from the opt-in dynamic quantization group size. Keep the KV cache at `u8`: it is the fastest and stayed within 0.007-0.016 nats of `bf16`.

### Memory

The KV pool is one fixed mapping whose pages become resident as blocks are used, so RSS climbs toward `baseline + pool` and plateaus; size the container for the sum. Prefix caching (on by default) retains freed blocks, so unique prompts eventually touch the whole pool.

With `VLLM_OPENVINO_KVCACHE_SPACE=0` the pool fills the memory ceiling (cgroup limit, else MemAvailable) minus the EngineCore's anonymous RSS after a full-size warmup, the APIServer (1.5 GiB text, 2.75 GiB vision) and 2.2 GiB headroom. Memory-mapped weights are not counted because the kernel can reclaim them without slowing decode. Measured EngineCore baselines (`max-model-len 4096`):

| Model | EngineCore anon RSS | APIServer |
|---|---|---|
| Qwen3-1.7B-int4 | weights + 2.5 GiB | 0.84 GiB |
| gemma-4-E2B-int4 | weights + 2.6 GiB | 2.5 GiB (vision) |
| LFM2.5-8B-A1B-int4 | weights + 3.5 GiB | 0.87 GiB |
| Phi-3.5-mini-int4 | weights + 3.0 GiB | 0.55 GiB |
| Qwen3.6-35B-A3B-int4 | weights + 3.6 GiB | 2.1 GiB (vision) |
| gemma-4-26B-A4B-int4 | weights + 7.2 GiB | 2.7 GiB (vision) |

gemma-4-26B-A4B needs a 32 GiB container (7.25 GB pool) and does not start in 24 GiB. Set an explicit `VLLM_OPENVINO_KVCACHE_SPACE` to cap the pool; it is only reduced when it does not fit. For text-only serving of a multimodal checkpoint, `--language-model-only` skips the vision encoder and frees its budget for KV. Hybrid-PA models also reserve conv/SSM state slots per request; on large hybrid models lower `--max-num-seqs` (Qwen3.6-35B-A3B: 64 MiB per slot).

### CPU placement

In containers the plugin detects the cgroup CPU quota and caps OpenVINO and Torch/OMP threads to it (an 8-CPU quota on a 24-core host removed 3.1 s of throttling from a 32-request burst). On multi-socket hosts bind the process to one socket's CPUs and set `VLLM_OPENVINO_CPU_THREADS_NUM` to its core count: splitting threads across sockets cost 12-18%, and doubling them across both sockets bought at most 9%.

### Chunked prefill (`--max-num-batched-tokens`)

The default 2048-token chunk takes about 3 s on CPU, and every running decode stalls for it. Qwen3-1.7B, 4 decodes running while 4 ~2.7k-token prompts arrive:

| `--max-num-batched-tokens` | decode ITL p99 | prefill TTFT mean | total wall |
|---|---|---|---|
| 2048 (default) | ~3.3 s | ~17.5 s | ~33 s |
| 1024 | ~1.75 s | ~16 s | ~33 s |
| 512 | ~0.9 s | ~15.5 s | ~33 s |
| 256 | ~0.7 s | ~19–21 s | ~36–42 s |

Use `512` for interactive or streaming workloads that mix long prompts with running generations; `256` starts losing throughput.

### Sampling

Sampling runs in the plugin's CPU sampler, which gives the same tokens and logprobs as vLLM's V1 sampler (`docs/compatibility.md`, "Sampler"). Greedy batches skip sampling entirely, so `temperature=0` is the fastest. `seed` is honored, also in batches with unseeded requests. Small Qwen3.5 models sampled at `temperature=1.0` without `top_k`/`top_p` emit stray foreign-script characters in Korean text; send `temperature=0.7, top_k=20, top_p=0.8` (Qwen's recommendation) per request.

### Benchmarking

`./scripts/benchmark.sh <model_path> [num_requests]` runs a warmed-up benchmark against the local endpoint. `./scripts/socket-experiment.sh <model_name> [seconds]` compares single-socket and split CPU sets.

## Serving Modes

The serving path is chosen from the IR. Both need `ScaledDotProductAttention` ops; an IR without them is rejected at startup.

- **PagedAttention** for attention-only models, including stateful (`ReadValue`) exports and sliding-window models such as Gemma-4. Concurrency-8 throughput was 4.9-7.0x the removed sequential path on dense models and 3.2-6.7x on Gemma-4. Greedy output can differ from the stateful IR on Qwen-family models because of the transformed attention, not the cache precision.
- **Hybrid-PA** for hybrid Mamba/attention models (Qwen3.5, LFM2.5, LFM3, MiniCPM-V 4.7): attention layers use PagedAttention and conv/SSM state is paged separately, so requests batch. Prefix caching works, also with speculative decoding, by keeping a state checkpoint per scheduler block; hits land on the checkpoint interval (`VLLM_OPENVINO_LA_CACHE_INTERVAL_MULTIPLIER`). Qwen3.5-0.8B 2.4k-token prompt: 8.2 s cold, 0.20 s on a hit.

## Compatibility

- Chunked prefill and prefix caching
- Structured outputs (`structured_outputs`, `response_format`, forced tool calls)
- N-gram speculative decoding (`--speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":5,"prompt_lookup_max":8}'`): speeds up copy-heavy output (code edits, quoting, RAG). Drafts only for greedy requests without penalties, logprobs or structured output
- MTP speculative decoding (`--speculative-config '{"method":"mtp","num_speculative_tokens":1}'`) on Qwen3.5/3.6 IRs that ship `openvino_mtp_model.xml`: single-stream Qwen3.5-2B 20.8 -> 27.4 tok/s, Qwen3.6-35B-A3B 10.8 -> 13.8 tok/s; the gain fades by 8 concurrent requests and is small on Korean text. Gemma-4 MTP is not supported
- `thinking_token_budget` (needs `--reasoning-parser`)
- Gemma 3 and Gemma 4, text and image
- Qwen3.5/3.6 text and image, LFM2.5 (Hybrid-PA)
- LFM3 (Hybrid-PA): needs OpenVINO 2026.5 (the image's nightly). It ships `repetition_penalty: 1.1`, which disables n-gram drafts; send `"repetition_penalty": 1.0` to use them
- MiniCPM-V 4.7 text and image (Hybrid-PA); needs vLLM >= 0.32.0 and Transformers >= 5.18.0
- `VLLM_USE_RUST_FRONTEND=1`, with the differences below

## Limitations

- LoRA, pin memory, and tensor/pipeline parallelism are not supported.
- `prompt_logprobs` (and `echo` with `logprobs`) is rejected with HTTP 400 because logits are computed only at sampled positions (HTTP 500 under `VLLM_USE_RUST_FRONTEND=1`). Output-token `logprobs` work.
- With `VLLM_USE_RUST_FRONTEND=1`, MiniCPM-V 4.7 image requests are rejected.
- Custom logits processors (`--logits-processors`) fail at startup; only vLLM's builtin `min_p`, `logit_bias` and `min_tokens` are applied.
- vLLM V1 engine only. Models without `ScaledDotProductAttention` ops are rejected; re-export them with SDPA.
- Qwen3.5 text-only exports (`openvino_model.xml`) are rejected: until openvinotoolkit/openvino#39652 ships, PagedAttention gives them wrong rotary positions. Use the multimodal export (`openvino_language_model.xml`, also with `--language-model-only`).

See `docs/compatibility.md` for implementation details and measurements.
