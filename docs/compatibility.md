# Compatibility

## Requirements

- vLLM `0.26.0`
- OpenVINO `>= 2026.3.0`
- Linux x86-64 with AVX2 or newer
- OpenVINO IR model; HuggingFace source models are not loaded directly
- Single socket only; tensor and pipeline parallelism are unsupported
- vLLM V1 engine only

Set `TORCH_COMPILE_DISABLE=1`. torch.compile/Inductor is incompatible with
the OpenVINO execution path.

## Serving Paths

The plugin selects a path from the model IR:

| Model shape | Path | Behavior |
|---|---|---|
| No `ReadValue`, with SDPA | PagedAttention | External KV cache and concurrent batching |
| `ReadValue` with `ssm`/`conv` variable IDs | Hybrid-PA | External attention/state cache and concurrent batching |
| Other stateful models | Stateful | OpenVINO internal state and `max_num_seqs=1` |

Hybrid-PA is the default for genuine hybrid Mamba/attention models such as
Qwen3.5 and LFM2.5. Set `VLLM_OPENVINO_HYBRID_PA=0` to force the stateful path.
Gemma-4 is not a Hybrid-PA candidate because its state is transformer KV
cache, not SSM/conv state.

Hybrid-PA uses standard PagedAttention for attention layers and a separate
physical slot pool for conv/SSM state. The slot pool is intentionally separate
from vLLM scheduler blocks.

## Runtime Constraints

- CPU KV block size is `32`; GPU KV block size is `16`.
- `VLLM_OPENVINO_CPU_THREADS_NUM=0` uses cgroup CPU quota when available;
  an explicit value takes precedence.
- `VLLM_OPENVINO_KVCACHE_SPACE=0` selects the backend default (4 GiB on CPU).
- KV cache allocation must not add zero-fill to `_allocate_kv_cache()`;
  OpenVINO initializes the cache and extra zero-fill can cause OOM.
- SSM/conv state caches are zero-filled; attention KV caches are not.
- bf16 tensors must be converted to float32 before NumPy conversion because
  OpenVINO does not accept bf16 NumPy arrays.
- Pin memory and LoRA serving are unsupported.
- `paged_attention_transformation()` and gather-before-matmul are applied
  only to PA paths. Stateful models must not receive PA-only inputs.
- OpenVINO import failure must remain deferred until configuration validation
  so plugin discovery can inspect other platforms safely.

## Model Files

- Text model: `openvino_model.xml`
- Language model: `openvino_language_model.xml`
- Text embeddings: `openvino_text_embeddings_model.xml`
- Vision encoder: `openvino_vision_embeddings_model.xml`
- Vision merger: `openvino_vision_embeddings_merger_model.xml`

The vision encoder expects a 2D `[num_patches, features]` input. Vision merger
calls require `hidden_states`, `attention_mask`, and `rotary_pos_emb`.
Qwen3.5 multimodal inputs use `pixel_values` and `image_grid_thw`; Gemma-4
uses `pixel_values`.

## Known Limitations

- Gemma-4 sliding-window attention is supported only on the stateful path;
  Hybrid-PA transformation is not supported for it.
- Stateful models do not support concurrent request execution.
- Structured outputs, LoRA, pin memory, and multi-socket execution are not
  supported.
