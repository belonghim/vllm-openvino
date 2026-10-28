# Compatibility

## Requirements

- vLLM `0.32.0`
- OpenVINO `>= 2026.4.1`
- Linux x86-64 with AVX2 or newer
- OpenVINO IR model; HuggingFace source models are not loaded directly
- Tensor and pipeline parallelism are unsupported; multi-socket scaling is poor (see CPU Tuning in README)
- vLLM V1 engine only

Set `TORCH_COMPILE_DISABLE=1`. torch.compile/Inductor is incompatible with
the OpenVINO execution path.

## Upstream vLLM Compatibility

The plugin ships its own worker and V1-style model runner (MRV1). vLLM 0.32.0
selects MRV2 only when `HAS_TRITON` is true. With the PyPI `vllm` wheel it is
false on CPU hosts even if Triton is installed, because vLLM requires exactly
one active GPU driver, so vLLM falls back to MRV1. The official vLLM CPU build
(`+cpu`) ships triton-cpu
([#58140](https://github.com/vllm-project/vllm/pull/58140)) and would select
MRV2, as would any Triton install with `CUDA_VISIBLE_DEVICES=""`. Set `VLLM_USE_V2_MODEL_RUNNER=0` in
those setups. vLLM marks MRV1 deprecated since 0.29.0; the 0.29.0 notes target
removal for 0.32, but 0.32.0 and main still ship it. vLLM's own CPU runner
(`CPUModelRunner`) still subclasses the MRV1 `GPUModelRunner`, so MRV1 removal
waits on CPU MRV2 without Triton
([#61995](https://github.com/vllm-project/vllm/pull/61995)). When
`gpu_input_batch` goes away, follow what the CPU runner moves to.

Installing the triton-cpu wheel into a `VLLM_TARGET_DEVICE=empty` install does
not change this: `vllm.triton_utils.importing` recognizes triton-cpu only when
`"cpu" in version("vllm")`, so the probe still reports no active driver,
`HAS_TRITON` stays false and MRV1 is selected. With the version string faked to
`+cpu` (MRV2 selected by config), the plugin's own worker still served
Qwen3-1.7B on the PagedAttention path. Hybrid-PA, stateful, multimodal, n-gram speculation and structured output were not tested under
that configuration.

Tracked upstream:

- [vllm-project/vllm#57684](https://github.com/vllm-project/vllm/issues/57684) —
  support-path policy question for platform plugins that ship their own
  worker/model runner and cannot install Triton.
- [vllm-project/vllm#58703](https://github.com/vllm-project/vllm/pull/58703) —
  adds `Platform.supports_v2_model_runner`; the plugin overrides it to `False`
  to opt out of the Triton probe (harmless no-op on vLLM builds without the
  hook). Parked: the reviewer prefers to wait for the MRV1 removal plan.
- [vllm-project/vllm#62576](https://github.com/vllm-project/vllm/pull/62576)
  (merged after 0.32.0) moves the MRV2 Triton check from config validation to
  `gpu_worker.Worker`. It keeps the MRV1 fallback for every platform except
  the unspecified one, and the plugin's worker does not derive from
  `gpu_worker.Worker`, so it does not affect the plugin.
- [vllm-project/vllm#61995](https://github.com/vllm-project/vllm/pull/61995)
  (open) makes MRV2 the default for CPU backends without Triton (Arm, POWER,
  s390x, RISC-V) by overriding MRV2 components (`sampler_cls`,
  `block_tables_cls`, `structured_outputs_worker_cls`, ...) in
  `CPUModelRunner`. It keeps MRV1 selectable and does not apply to x86. Once
  merged it is the pattern to follow when MRV1 is removed.
- [vllm-project/vllm#63373](https://github.com/vllm-project/vllm/pull/63373)
  (open draft, no milestone) removes MRV1: `gpu_input_batch.py`,
  `cpu_model_runner.py`, the V1 `Sampler` and `use_v2_model_runner`, and drops
  "vLLM on CPU without Triton". The plugin breaks on import: it uses
  `InputBatch`/`CachedRequestState`, the V1 `Sampler`/`SamplingMetadata`,
  `NgramProposer` (n-gram speculation) and `apply_grammar_bitmask` (structured
  output), all removed.
  The MRV2 replacements (`vllm/v1/worker/gpu/input_batch.py`, `block_table.py`,
  `sample/gumbel.py`, `penalties.py`, ...) are Triton kernels, so the plugin
  cannot switch to them. If #61995's torch components survive, follow them;
  otherwise keep trimmed copies of the classes in the plugin (the runner uses
  about ten `InputBatch` members). Before any vLLM upgrade, check whether the
  release contains #63373.
  Asked on the PR (2026-10-27) whether a non-Triton MRV2 path stays supported.

Open questions on #57684 (MRV1 removal timing, and whether a replacement-runner
path exists for plugins) are unanswered after two follow-ups.

Since vLLM 0.32 the `Sampler` calls `torch.ops._C.greedy_argmax` whenever the
platform reports CPU, and the PyPI-style wheel has no CPU kernels. `platform.py`
registers an argmax implementation for that op when it is missing.

## Serving Paths

The plugin selects a path from the model IR:

| Model shape | Path | Behavior |
|---|---|---|
| No `ReadValue`, with SDPA | PagedAttention | External KV cache and concurrent batching |
| `ReadValue` with `ssm`/`conv` variable IDs | Hybrid-PA | External attention/state cache and concurrent batching |
| Other `ReadValue` models with SDPA (incl. Gemma-4) | PagedAttention (`VLLM_OPENVINO_STATEFUL_PA=1`) | External KV cache and concurrent batching |
| Stateful models without SDPA, or `VLLM_OPENVINO_STATEFUL_PA=0` | Stateful | OpenVINO internal state and `max_num_seqs=1` |

Stateful plain-attention models (optimum-intel default exports, e.g.
Qwen2.5-Coder-int4-ov) get the PagedAttention transformation at load time
by default (`VLLM_OPENVINO_STATEFUL_PA=1`): `max_num_seqs` keeps its default
128 and the KV cache moves to the external paged cache (u8 default, which
is the OpenVINO CPU plugin's own default on x86_64/ARM64 - the variable is
unset by default and the loader then passes no `KV_CACHE_PRECISION`;
verified on Qwen3-1.7B, Gemma-4-E2B, LFM2.5-350M (Hybrid-PA) and
Qwen3.5-0.8B (hybrid), all `cache_type=u8`. `VLLM_OPENVINO_KV_CACHE_PRECISION`
selects `u8`/`f16`/`bf16` through the
OpenVINO `KV_CACHE_PRECISION` compile property; on a CPU whose runtime
precision is bf16 the plugin replaces `f16` with `bf16` (`scaled_attn.cpp`
`side_cache_precision`: f16 is kept only if rtPrecision is not bf16), seen
as `key_cache.0` port type and worker `cache_dtype` both `bf16` — `f32`/`i8` are rejected by
CPU PagedAttention and fall back to the default with a warning). Validated
on Qwen2.5-Coder-0.5B, Qwen3-1.7B, TinyLlama-1.1B and Phi-3.5-mini (int4
IR): concurrency-8 aggregate throughput was 4.9-7.0x the stateful path,
single-stream -7.8% to +4%. Sliding-window models take the same path: the
PagedAttention op applies the window itself. Stateful models without SDPA
ops stay on the stateful path.

Gemma-4 E2B (sliding window 512, 15 KV pairs after KV sharing, per-layer
embeddings) was verified on the PA path with OpenVINO 2026.4.1: greedy output
on 2.3k-token prompts and on image prompts was byte-identical to the stateful
path, concurrent and prefix-cache-hit requests matched single requests, and
n-gram speculation produced the same output. 128-token decode (8 CPUs, u8 KV):
concurrency 1 18.6 -> 19.5 tok/s, concurrency 8 18.7 -> 125.2 tok/s (6.7x);
a repeated 2.3k-token prompt went from 7.5 s to 1.2 s through prefix caching.
Gemma-4 26B-A4B (MoE, sliding window 1024, 30 KV pairs, `token_type_ids`
input, 14.3 GB weights) gave the same text answer as the stateful path and the
same extraction of a text-heavy image; concurrency 1 9.8 -> 10.2 tok/s,
concurrency 8 9.8 -> 30.9 tok/s (3.2x). With the automatic pool the PA path
needs a 32 GiB container: post-warmup anonymous RSS is 19.8 GB, the pool
7.25 GB, and three concurrent images followed by about 141k tokens of unique
~2.5k-token prompts four at a time passed with no OOM. At 24 GiB the pool
falls to the 256 MiB floor and startup fails for `max-model-len 4096`.
When concurrent requests split long prompts at different chunk boundaries,
greedy output can change at near-tie tokens (seen on E2B with and without
prefix caching, both answers correct).

The PA graph keeps `per_layer_inputs` rank 4, so it is fed as
`[tokens, 1, layers, hidden]` (as in OpenVINO GenAI's `ModelRunner`); 26B-A4B
has hidden 0, so reshapes name the token count instead of `-1`.
`token_type_ids` stays rank 2 and is fed as `[tokens, 1]` per the
PagedAttention op, carrying the step's real per-token types (1 on image
embedding positions, 0 elsewhere) the way OpenVINO GenAI's `ModelRunner` feeds
them. The PA kernel reads this input directly (input 25). A synthetic run of
the PA op from this IR showed that tokens inside a contiguous type-1 run attend
bidirectionally, text stays causal, and two runs separated by text are
independent. Adjacent runs with no separating token merge into one block. The
E2B IR has no `token_type_ids` input, so the feed is dropped there and its image
tokens stay causal. Without `per_layer_inputs` the first infer failed in a
sliding-window layer's residual add (`Eltwise shape infer ... dim index: 0
mismatch`).

Greedy output matched the stateful path on TinyLlama-1.1B and
differed on Qwen-family models; u8 and bf16 caches produced
byte-identical output on those short checks, so the difference comes from
the transformed attention computation, not cache precision (long-context
`u8` vs `bf16` differences are in the Performance Profile). Set
`VLLM_OPENVINO_STATEFUL_PA=0` to force the stateful path.

Hybrid-PA is the default for genuine hybrid Mamba/attention models such as
Qwen3.5 and LFM2.5. Hybrid models without SDPA ops keep the stateful path.
Gemma-4 is not a Hybrid-PA candidate because its state is transformer KV
cache, not SSM/conv state.

Hybrid-PA uses standard PagedAttention for attention layers. Conv/SSM state
storage depends on prefix caching:

- Prefix caching on (default, no speculative decoding): each vLLM scheduler
  block owns one conv/SSM checkpoint slot with the same index. The
  linear-attention inputs follow OpenVINO GenAI's prefix layout: read the
  checkpoint of the block holding token `num_computed - 1`, write one
  checkpoint per touched interval (`la.cache_interval` = scheduler block
  size). A prefix hit is always block-aligned, so it restores the state
  from the last hit block. A scheduler block spans `N` OpenVINO KV blocks of
  32 tokens; the runner expands each block id to its `N` KV block ids. `N` is
  `VLLM_OPENVINO_LA_CACHE_INTERVAL_MULTIPLIER`, default 8 (256 tokens) for
  every model. `0` selects GenAI's automatic rule
  `max(8, checkpoint bytes / KV block bytes)`: 8 for LFM2.5-350M (61 KB
  checkpoint) and 47 (1504 tokens) for Qwen3.5-0.8B and MiniCPM-V 4.7
  (10.3 MB checkpoint). Checkpoint bytes are part of the block page size, so
  the 4 GiB pool holds 570k tokens for LFM2.5 (619k without prefix caching)
  and 90k for Qwen3.5 (280k with the automatic 1504-token interval). With
  multiplier 1 (32 tokens, the plain PA block) LFM2.5 holds 466k tokens and
  hits like a pure-attention model; Qwen3.5 would hold about 13k.
- Checkpoint bytes follow the CPU plugin's state precision: bf16 on CPUs with
  AVX-512-BF16, f32 otherwise (or an explicit
  `VLLM_OPENVINO_INFERENCE_PRECISION`). The same IR and flags therefore give a
  10.3 MB (bf16) or 20.6 MB (f32) checkpoint on Qwen3.5-0.8B and a block page
  of 12.1 or 22.4 MB. A memory limit that fits the bf16 pool can leave the f32
  pool at the 256 MiB floor, which holds 11 blocks where one
  `--max-model-len 4096` request needs 16; the worker logs the applicable
  remedies before vLLM's own ValueError.
- vLLM caches a block when it allocates it, so a request can hit a block that
  another request fills in the same step. The runner then splits the batch
  into waves (writer first) so the checkpoint read follows its write. Without
  waves, 3 of 4 concurrent identical prompts had different logprobs.
- Prefix caching off, or speculative decoding configured: a separate
  physical slot pool, described below.

Verification (8 CPUs, greedy): a hit run's first-token logprobs were
bit-identical to a no-prefix-caching run whose chunk boundary is at the same
interval (`--max-num-batched-tokens 1504`), and to a cold run with the same
chunking, including MiniCPM-V and Qwen3.5 image prompts whose hit boundary falls
inside the image tokens. A cold run with a different chunk boundary can differ
slightly because the bf16 checkpoint is stored at the boundary, the same
effect chunked prefill already has. Hit TTFT: LFM2.5 2.1k-token prompt 0.74 s
-> 0.06 s; Qwen3.5 2.4k/4.0k-token prompts 8.2/13.9 s -> 0.20/0.56 s
(multiplier 8) or 2.9/3.4 s (1504-token hits). Concurrency-8 decode was
unchanged (LFM2.5 410 vs 403 tok/s, Qwen3.5 133 vs 134 tok/s).

The slot pool (prefix caching off) is intentionally separate
from vLLM scheduler blocks. Slots are released when a request finishes and are
reset on a fresh or re-prefilled start (`num_computed == 0`). A preempted
request keeps its slot, so the pool of `max_num_seqs + 1` is never exhausted
under FCFS scheduling. This was verified on Qwen3.5-0.8B with forced
preemption (`--num-gpu-blocks-override`, 7 preemptions at `max_num_seqs=2`).
A recomputed request can diverge from its uninterrupted output late in the
text because of prefill vs decode numerics.

## Runtime Constraints

- CPU KV block size is `32`; GPU KV block size is `16`. Hybrid-PA with prefix
  caching uses a larger scheduler block (a multiple of the KV block size).
- `VLLM_OPENVINO_CPU_THREADS_NUM=0` uses cgroup CPU quota when available;
  an explicit value takes precedence. The same quota caps the Torch/OMP
  thread pools at engine config time unless `OMP_NUM_THREADS` is set
  externally.
- `VLLM_OPENVINO_KVCACHE_SPACE=0` sizes the CPU pool automatically from the
  memory ceiling (see the sizing rules below); GPU profiles the device.
- KV cache allocation must not add zero-fill to `_allocate_kv_cache()`;
  OpenVINO initializes the cache and extra zero-fill can cause OOM.
- The KV cache is one fixed-size mapping whose pages become resident only as
  blocks are used, so RSS grows toward `baseline + KV pool` and then plateaus
  (measured with a 1 GiB pool: one 1049 MiB mapping, Rss 36 -> 110 MiB over
  10 rounds while its size stayed fixed; cgroup total 1.6 -> 2.5 GB).
  Size containers for `baseline + VLLM_OPENVINO_KVCACHE_SPACE`, not for the
  pool alone.
- On CPU the pool is sized from a memory ceiling: the cgroup memory limit when
  one is set, otherwise host MemAvailable. Before model load (estimate):
  ceiling - model weights (`openvino_*.bin`) - 1.5 GiB runtime baseline -
  APIServer overhead - 2.2 GiB headroom. After model load: the PA warmup runs
  first, then the pool is set to ceiling - EngineCore anonymous RSS -
  APIServer overhead - 2.2 GiB headroom (never below 256 MiB). The APIServer
  overhead is 1.5 GiB for text models and 2.75 GiB for checkpoints with a vision
  encoder (measured APIServer anon RSS 0.55-0.87 GiB text, 1.23-1.26 GiB
  Gemma-4 with images disabled, 2.1-2.7 GiB vision). Anonymous RSS is the right baseline because the kernel
  reclaims the memory-mapped weights under pressure (26B: VmRSS 33.5 GB of which
  19.8 GB is anonymous).
- With `VLLM_OPENVINO_KVCACHE_SPACE=0` (default) the post-load budget sets the
  pool in both directions. With an explicit value it only reduces the pool to
  that budget, with a warning; a pool above 90% of the pre-load budget logs a
  headroom warning.
- vLLM's prefix caching (default on) retains freed blocks, so workloads with
  unique prompts keep touching new pool pages over time;
  `--no-enable-prefix-caching` keeps the resident set at the working set
  (measured: anon flat at ~477 MiB after warm-up vs +62 MiB over 8 rounds
  with prefix caching on).
- Prefix caching is disabled automatically for stateful models, and for
  Hybrid-PA models with speculative decoding: their recurrent/internal state
  is not checkpointed per vLLM block, so a prefix hit would skip tokens
  without restoring that state (repeated long prompts gave different, wrong
  answers). Pure-attention PA models and Hybrid-PA models without
  speculative decoding keep it enabled.
- Multimodal requests are encoded per image (own grid and merger attention)
  when vLLM's scheduler lists the image in `scheduled_encoder_inputs`; the
  merged embeddings stay in the runner's encoder cache until the scheduler
  frees them (`free_encoder_mm_hashes`), so an image split across steps is
  encoded once. Embeddings are scattered by explicit
  `(batch_start, batch_end, patch_offset, image_index)` rows (one per contiguous
  `is_embed` run of the placeholder), so concurrent requests, differently sized images, multi-image prompts and
  images split by chunked prefill are handled. For text-only serving of a
  multimodal checkpoint, `--language-model-only` or
  `--limit-mm-per-prompt '{"image":0,"video":0}'` skips the multimodal warmup
  (startup 64 s to 50 s on Qwen3.5-0.8B-int4), does not compile the vision
  encoder, and budgets the text APIServer overhead. gemma-4-E2B at 16 GiB:
  post-load anon RSS 3.84 -> 2.71 GB, pool 7.21 -> 9.59 GB (1.23 GB APIServer observed, 1.5 GiB budgeted); image requests
  get HTTP 400 from vLLM.
- PA models run one prefill and one decode step on the reserved null block 0 at
  startup to absorb the CPU kernel JIT cost; the first request dropped from
  1.51 s to 0.79 s (Qwen3-1.7B-int4) and 0.87 s to 0.49 s (Qwen3.5-0.8B-int4)
  with no visible startup cost. On CPU the warmup runs inside
  `determine_available_memory`, against a throwaway cache, *before* the
  post-load memory measurement, so the CPU plugin's first-infer weight repack
  and prefill buffers are part of the measured baseline;
  `compile_or_warm_up_model` skips its own copy. The warmup prefill is
  `max_num_batched_tokens` long, and multimodal models also encode one
  maximum-size dummy image from vLLM's dummy-input processor and prefill the
  first step of its prompt (skipped when the image limit is 0 or the dummy
  prompt exceeds both 2048 tokens and one step, e.g. Qwen3.5's 16k-token dummy
  image). At `--max-num-batched-tokens 256`, leaving out the image would put
  the baseline 0.52 GB low on gemma-4-E2B (259-token prompt) and 0.58 GB low
  on MiniCPM-V 4.7 (679 tokens), and the pool that much too large. With a
  16-token warmup the 26B baseline was 16.9 GB
  and the pool OOM-killed under full residency; with the full-step text and
  image warmup it is 19.8 GB and the same loads pass. The startup cost is
  26 s on the 26B and 11 s on E2B. Hybrid-PA prefix caching needs one
  conv/SSM checkpoint per warmup block in the throwaway cache. Stateful models keep their own
  warm-up in `load_model`.
- SSM/conv state caches are zero-filled; attention KV caches are not.
- The worker zero-fills scheduler-reported `new_block_ids_to_zero` blocks each
  step. Removing it changed nothing measurable (Qwen3-1.7B-int4, 8-CPU quota,
  2 runs each: prefill 16 x ~1.4k tokens at concurrency 8 about 690 tok/s,
  decode concurrency 8 about 125 tok/s, RSS 6.5 GB, decode output identical),
  so it is kept to stay aligned with upstream.
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

### MiniCPM-V 4.7

The vision IR takes one packed crop per infer: `pixel_values [1,3,14,h*w*14]`
plus host-built `position_ids`, `window_index` and `merge_index` (same algorithms
as OpenVINO GenAI's `VisionEncoderMiniCPMv4_7`) and returns merged tokens
`[tokens, hidden]`. vLLM's zero-padded `pixel_values [n_slices,3,14,L_max]` is
cut by `tgt_sizes` (passed through the `image_grid_thw` slot). The slice
placeholders interleave `<slice>`/newline tokens with `<|image_pad|>` runs, so
embeddings go to the `is_embed` runs only. The LM takes `[3, n]` canvas M-RoPE
positions computed by vLLM's `MiniCPMV4_7ForConditionalGeneration`
(`get_mrope_input_positions`); decode positions are `index + rope_delta`.

Requires vLLM >= 0.32.0 (contains vllm-project/vllm #58674 and #56372) and
Transformers >= 5.18.0; vLLM 0.31.0 and Transformers 5.17.0 lack them. Verified
on the 0.32.0 image: single and multi-slice images and concurrent image requests
describe the image correctly; chunked prefill was last checked on 0.32.0rc0.

## N-gram Speculative Decoding

Only `method=ngram` is accepted, on PagedAttention and Hybrid-PA models
(`check_and_update_config` raises otherwise; a model that falls back to the
stateful path fails at load). `execute_model` computes logits for the 1+k query
positions of each draft request, and `sample_tokens` accepts drafts while they
match the argmax (the stock rejection sampler needs Triton). Drafts are
proposed only for greedy requests without penalties, `logit_bias`,
`allowed_token_ids`, `bad_words`, `min_tokens`, logprobs, or structured
output; other requests in the same batch use the normal sampler. vLLM disables
async scheduling for `ngram` automatically.

Hybrid-PA: a draft request runs with `la.cache_interval=1` and
`la.block_indices = [committed slot, 1+d temporary slots]`, so token t's
conv/SSM state lands in temporary slot t. After verification the slot of the
last accepted token becomes the committed slot and the rest return to the pool.
With speculation the pool holds `max_num_seqs * (k + 2) + 1` slots, so lower
`--max-num-seqs` when enabling it on a large-state model.

Measured (8 CPUs, k=4, `prompt_lookup_min=5`, `prompt_lookup_max=8`,
tok/s, baseline -> spec; copy-heavy / free-form output):

| Model | 1 request | 4 concurrent | 8 concurrent | 16 concurrent |
|---|---|---|---|---|
| Qwen3-1.7B (PA) | 23 -> 57 / 23 -> 23 | 82 -> 148 / 82 -> 85 | 154 -> 282 / 160 -> 149 | 268 -> 358 / 267 -> 274 |
| Qwen3.5-2B (Hybrid-PA) | 21 -> 43 / 22 -> 21 | 56 -> 92 / 64 -> 64 | 86 -> 114 / 102 -> 90 | - -> 124 / - |
| LFM2.5-8B-A1B (Hybrid-PA, MoE, conv-only state) | 23 -> 42 / 25 -> 50 | 44 -> 60 / 49 -> 78 | 56 -> 69 / 66 -> 88 | - |

LFM2.5-8B-A1B was run with `--generation-config vllm`; its "free-form" prompt
degenerates into loops under greedy decoding without a repetition penalty, so
that column is not representative of free text (acceptance ~99%).

Model `generation_config.json` defaults count as request parameters. LFM2.5
ships `repetition_penalty: 1.05`, which makes every request ineligible for
drafts (0 drafts proposed) unless the request sets `repetition_penalty=1.0` or
the server runs with `--generation-config vllm`.

Free-form output is neutral up to 4 concurrent requests and loses up to ~10%
on Hybrid-PA at 8, where verification is not free. Keep `prompt_lookup_min`
high: with 2, spurious matches cost 7-11% on free-form output for Qwen3-1.7B.
Output can diverge from non-spec runs at near-tie tokens because kernel
numerics vary with query length.

Stateful-path models are not supported. Rolling back rejected tokens
by trimming every KV state on the sequence axis (`get_state`/`set_state`) is
exact, sliding-window layers included (Gemma-4 E2B, 700-token context,
logits identical to the untrimmed run), but the trim copies all states at
10-33 ms for 700-4000 tokens of context against a 53-130 ms decode step, and
the stateful path serves one request at a time and is maintenance-only.
Gemma-4 runs n-gram speculation on the PA path.

## Performance Profile

Measured on Ryzen 9 7900 (no AMX), 8-CPU quota, OpenVINO 2026.4.1, greedy
decode. Where the time goes, so tuning effort is not spent in the wrong place:

- **Runner overhead is negligible.** Qwen2.5-Coder-0.5B-int4: client inter-token
  latency 13.7 ms (1 request) / 19.8 ms (8) vs step time 13.9 / 20.1 ms; `infer`
  is ~95% of the step, input preparation 0.1 ms, sampling 0.5-1.1 ms. The same IR
  run as a plain stateful model in OpenVINO decodes in 13.3 ms (LATENCY) /
  14.5 ms (THROUGHPUT), so the PagedAttention transformation costs nothing.
  `VLLM_OPENVINO_PERFORMANCE_MODE=LATENCY` made no difference in the server.
- **KV precision: the `u8` default is the fastest.** Qwen3-1.7B-int4 decode at
  1.2k / 4.5k / 8.7k / 17.5k context tokens: `u8` 36 / 43 / 51 / 70 ms,
  `f16` 37 / 47 / 59 / 89 ms, `bf16` 37 / 48 / 61 / 88 ms; prefill speed is the
  same for all three. The `f16` and `bf16` rows are the same configuration on
  this CPU (the plugin stores `bf16` for an `f16` request), so their small
  differences are run-to-run noise. Greedy outputs diverge after 40-480
  characters between `u8` and the 2-byte cache, so the two were compared on
  next-token distributions. 20 prompts of 500 / 2k / 5k / 12k tokens (slices of
  this repository's sources), each scored at 4 positions along the `bf16`
  greedy continuation (teacher-forced, `logprobs=20`, `u8` vs `bf16`). Top-1
  token agreement was 79/80 on Qwen3-1.7B-int4 and 79/80 on
  Qwen2.5-Coder-0.5B-int4; mean KL over the top-20 was 0.007 and 0.016 nats,
  mean |logprob change| of the top-1 token 0.019 and 0.024. The one flip per
  model was a near tie (`bf16` top-1/top-2 gap 0.0 and 0.25 nats). KL did not
  grow with context (Qwen3: 0.006 / 0.012 / 0.007 / 0.003 at 500 / 2k / 5k /
  12k), and rerunning `bf16` gave bit-identical logprobs. 80 positions per
  model is a small sample: this bounds the `u8` effect, it does not replace a
  task benchmark.
- **Long-context prefill slows down.** Qwen3-1.7B-int4 `u8`: 845 / 732 / 607 /
  403 tok/s at the same four context lengths, consistent with attention
  becoming about half of the time at 17k tokens. The OpenVINO node profile
  (`PERF_COUNT`) did not sum to the step time, so it was not used for shares.
- **MoE decode is expert matmul.** `GatherMatmulCompressed` is ~62%
  (LFM2.5-8B-A1B) to ~78% (Gemma-4 26B-A4B) of the node time and attention
  2-4%. Decode is 34 ms/token on LFM2.5-8B-A1B and ~90 ms/token on Gemma-4
  26B-A4B. This is OpenVINO CPU kernel time, not plugin code, and it is not
  slow for what it reads. Weight bytes per token (constants in the IR, routed
  experts scaled by top-k / experts, everything else read in full, tied
  `lm_head` included) over the step time: LFM2.5-8B-A1B 1.0 GB in 34 ms
  (29.5 GB/s), Gemma-4 26B-A4B 2.4 GB in 90 ms (26.9 GB/s), Qwen3-1.7B-int4
  1.18 GB in 36 ms (~32 GB/s). Decode is memory-bandwidth bound at ~27-32 GB/s
  for dense and MoE alike, so only reading fewer bytes (smaller weights, fewer
  active experts) makes it faster. Evidence: a read-only AVX2 sum over 2 GB
  reaches 43 GB/s on 1 thread and 45-48 GB/s on 2-24 threads on this machine,
  and plain OpenVINO stateful decode of Qwen3-1.7B-int4 is flat from 4 to 24
  threads (33.5 / 33.8 / 34.2 / 34.5 / 34.5 ms at 4 / 8 / 12 / 16 / 24), about
  35 GB/s or ~75% of that ceiling. Decode needs only ~4 CPUs; extra CPUs help
  prefill and batched decode, not single-stream decode.
- **Prefill speed:** Qwen3-1.7B-int4 ~840 tok/s at 1k tokens; LFM2.5-8B-A1B
  ~516 tok/s at 2.3k (patched plugin); Gemma-4 26B-A4B ~170-250 tok/s at 2.4k
  (first requests are slower while mmapped weights fault in).
- **Prefix caching works on the PA path:** an 8.6k-token prompt took 14.4 s
  the first time and 0.19 s when repeated.
- **Prefill is at the matmul ceiling.** Qwen3-1.7B has 1.41 B non-embedding
  parameters (2.8 GFLOP/token), so 888 tok/s at 1k tokens is ~2.5 TFLOP/s. A
  standalone OpenVINO matmul with Qwen3 MLP shapes (1024 x 2048 x 6144, 8-CPU
  quota, `bf16` inference precision) reached 1.8-1.9 TFLOP/s with `bf16`
  weights and 2.1-2.2 with int4 weights at the default
  `DYNAMIC_QUANTIZATION_GROUP_SIZE` 32 (256 tokens: 2.7), so prefill has no
  headroom left at the default beyond kernel work. The one lever found is
  `DYNAMIC_QUANTIZATION_GROUP_SIZE=128` (exposed as
  `VLLM_OPENVINO_DYNAMIC_QUANTIZATION_GROUP_SIZE`, unset by default; measured
  with a scratch patch before the variable existed): matmul 2.7-2.8 TFLOP/s and server
  prefill 888 -> 1148 tok/s at 1k tokens and 752 -> 935 at 4k (+29% / +24%),
  decode unchanged (28.6-29.3 tok/s). Accuracy was compared with a
  larger sample (56 first-token distributions of 300 / 1k / 3k / 6k-token
  prompts per model, top-20 KL against no dynamic quantization, `0`):

  | | prefill 1k / 4k (tok/s) | mean KL (nats) | top-1 agreement |
  |---|---|---|---|
  | Qwen3-1.7B, 0 | 627 / 560 | - | - |
  | Qwen3-1.7B, 32 (default) | 877 / 755 | 0.032 | 55/56 |
  | Qwen3-1.7B, 64 | 1031 / 863 | 0.031 | 55/56 |
  | Qwen3-1.7B, 128 | 1155 / 940 | 0.040 | 56/56 |
  | Qwen2.5-Coder-0.5B, 0 | 2241 / 1918 | - | - |
  | Qwen2.5-Coder-0.5B, 32 | 3124 / 2450 | 0.033 | 54/56 |
  | Qwen2.5-Coder-0.5B, 64 | 3610 / 2767 | 0.051 | 55/56 |
  | Qwen2.5-Coder-0.5B, 128 | 3857 / 2911 | 0.042 | 53/56 |

  Larger groups buy +17-18% (64) and +23-32% (128) prefill, but the default
  already deviates from unquantized activations by ~0.03 nats and the extra
  deviation of 64/128 (0.00-0.02) is within the noise of this sample (it is
  not monotonic on Qwen2.5-Coder; 48-token greedy output matched the
  unquantized run on only 1-3 of 8 prompts for every setting, including the
  default). An earlier 24-prompt run suggested 2.4x the KL at 128; the larger
  sample does not support that. KL grows with prompt length for all settings
  (Qwen3: 0.01-0.03 at 300 tokens, 0.09-0.11 at 6k). The default was kept
  because the cost, though small, could not be ruled out; the variable is opt-in.
- **Concurrency scaling (decode-heavy, 8 CPUs).** N simultaneous requests,
  ~15-token prompts, 256 output tokens each, `ignore_eos`; aggregate output
  tok/s (default `max_num_seqs` 128):

  | N | 1 | 4 | 16 | 32 | 64 | 128 |
  |---|---|---|---|---|---|---|
  | Qwen3-1.7B-int4 | 29 | 101 | 327 | 469 | 508 | 532 |
  | Qwen2.5-Coder-0.5B-int4 | 86 | 236 | 757 | 1125 | 1228 | 1340 |
  | LFM2.5-8B-A1B-int4 (MoE) | 32 | 59 | 121 | 157 | 228 | 296 |

  Dense models scale almost linearly to N=8-16 (weights are read once per
  step), reach ~85-90% of their ceiling at N=32 and gain only 5-15% more
  from N=32 to 128 while per-request speed halves each doubling, so N=16-32
  is the useful range. The MoE model scales poorly because different tokens
  select different experts, so a larger batch reads more expert weights per
  step; it is still gaining at N=128. With 4 CPUs Qwen3-1.7B matches 8 CPUs
  up to N=4 (29 / 99 tok/s) but flattens at 318 tok/s against 532 once the
  batch is large enough to become compute-bound (N=16: 253 vs 327). Use 4
  CPUs for a single-user or few-user deployment and more for batch serving.

## Known Limitations

- Stateful models do not support concurrent request execution.
- LoRA, pin memory, and multi-socket execution are not supported.
- `prompt_logprobs` is not supported: the gather-before-matmul transformation
  computes logits only at sampled positions. `OpenVinoPlatform.validate_request`
  rejects such requests with HTTP 400. Output-token `logprobs` work.
- Gemma-4 image prompts on the stateful path pass `token_type_ids` as zeros, so
  image tokens do not attend to each other bidirectionally there; the
  PagedAttention path feeds the sequence's real token types. On a text-heavy
  test image the two arrangements returned the same flag list and the same text
  answer, and one-sentence descriptions that differ in wording (both plausible);
  no quality delta beyond that was demonstrated.
- MoE prefill is slow on CPUs without AMX: LFM2.5-8B-A1B ~108 tok/s,
  Qwen3.6-35B-A3B ~74 tok/s (8 CPUs, Zen 4), with `GatherMatmulCompressed`
  taking 80-90% of the time. The OpenVINO CPU plugin batches routed rows per
  expert into a GEMM only on bf16+AMX; elsewhere it runs one GEMV per routed
  row. Tracked upstream in openvinotoolkit/openvino#38517, fixed by #39086
  (merged into master, milestone 2026.5). Until a release contains it, the image ships a
  patched CPU plugin with per-expert GEMM (`patches/README.md`): 425 and
  204 tok/s at 1024 tokens with unchanged NLL.
- `VLLM_USE_RUST_FRONTEND=1` starts with the bundled `vllm-rs` binary but gave
  no throughput change (Qwen3-1.7B, 1/8/16 concurrent, within run-to-run
  noise): decode runs at 20-300 tok/s, so the frontend is not the bottleneck.
  Its multimodal support was not tested.
