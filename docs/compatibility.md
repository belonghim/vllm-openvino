# Compatibility

## Requirements

- vLLM `0.34.0`
- OpenVINO `>= 2026.4.1`; the image pins nightly `2026.5.0.dev20261030`, which
  has the MoE prefill fix (#39086) and reads IRs exported with 2026.5 (LFM3)
- Linux x86-64 with AVX2 or newer
- OpenVINO IR model; HuggingFace source models are not loaded directly
- Tensor and pipeline parallelism are unsupported; multi-socket scaling is poor
  (see Performance Profile)
- vLLM V1 engine only

Set `TORCH_COMPILE_DISABLE=1`. torch.compile/Inductor is incompatible with
the OpenVINO execution path.

## Upstream vLLM Compatibility

The plugin ships its own worker and model runner and keeps trimmed copies of
the MRV1 pieces it used, which give the same output as vLLM 0.32's:
`sampler.py` (V1 `Sampler`, `SamplingMetadata`), `worker_v1/input_batch.py`
(`InputBatch`, `CachedRequestState`), `worker_v1/ngram_proposer.py`, and
`apply_grammar_bitmask` in the runner (about 3x faster than vLLM's xgrammar
call on CPU). It imports none of the modules upstream removes. The sampler
needs no `torch.ops._C` kernel; vLLM's CPU path calls `greedy_argmax` and
`fused_gumbel_argmax`, which the `VLLM_TARGET_DEVICE=empty` wheel lacks.

vLLM 0.34.0 has no MRV1 and no `VLLM_USE_V2_MODEL_RUNNER` switch (#63373);
leave the variable unset. The plugin's platform and runner do not depend on
`HAS_TRITON`, which is false on CPU hosts with the PyPI wheel.

Tracked upstream:

- [#61995](https://github.com/vllm-project/vllm/pull/61995) (in 0.34.0) adds `StandaloneCPUModelRunner`, an MRV2 runner
  with no `vllm/v1/worker/gpu` dependency. It is selected only when
  `not HAS_TRITON and cpu_architecture != X86`, so nothing changes on x86. It
  needs ten `torch.ops._C.*_cpu` ops missing from the `empty` wheel and does
  not support speculative decoding.
- [#63373](https://github.com/vllm-project/vllm/pull/63373) (in 0.34.0) removes MRV1 (`gpu_input_batch.py`,
  `cpu_model_runner.py`, the V1 `Sampler`, `VLLM_USE_V2_MODEL_RUNNER`) and
  makes `CpuPlatform` refuse x86 without Triton (the plugin's platform is not
  `CpuPlatform`). It also moves the scheduler-worker interface to MRV2's, which
  the runner and worker accept alongside 0.33.0's: `CachedRequestData` loses
  `resumed_req_ids` and `all_token_ids`, `ModelRunnerOutput` loses
  `req_id_to_index`, `NewRequestData.prefill_token_ids` becomes required, and
  a preempted request resumes through `scheduled_new_reqs` with
  `prefill_token_ids` holding prompt plus generated tokens. The runner
  restores the generated tokens into `output_token_ids` (so penalties and
  n-gram lookups see them) and keeps the request's `torch.Generator` (so a
  seeded request continues its sequence). The platform's `validate_environ`
  rejects `VLLM_USE_V2_MODEL_RUNNER` values other than `1`, as upstream does.
  Verified with the merge-commit wheel (`aac1b2a4f`): under the same
  preemptions Qwen3-1.7B greedy and seeded output matched 0.33.0 (dropping the
  restore or the generator changed every preempted request), and the smoke
  test passed at 0.33.0's throughput. `encoder_input_updates` only manage the
  worker's copy of multimodal payloads; the runner keeps its own
  `mm_features`.
- [#61452](https://github.com/vllm-project/vllm/pull/61452) (merged after
  v0.34.0rc3, expected in 0.35) moves allocation policy from `KVCacheSpec`
  onto the KV cache managers: `FullAttentionSpec` is removed and
  `get_kv_cache_spec()` becomes `get_kv_cache_layer_configs()`. The worker
  supports both: it imports `AttentionSpec` when `FullAttentionSpec` is gone
  and wraps each spec in `FullAttentionManager.layer_config()`.
  `page_size_padded` stays on `AttentionSpec`. On main `5ec99956a`
  Qwen3-1.7B and Qwen3.5-0.8B planned the same blocks as on 0.33.0 with
  identical greedy output, MTP drafted, and the smoke test passed on both
  frontends. That main pins `torchvision==0.29.2`.
- [#63830](https://github.com/vllm-project/vllm/pull/63830) (main, after
  0.34.0) removes `mamba_cache_mode`; prefix caching alone now selects Mamba
  state caching. The platform's `mamba_cache_mode = "none"` override is
  guarded by `hasattr` and becomes a no-op. Hybrid-PA is unaffected because
  it exposes only attention specs: `has_mamba_layers` stays false, so the
  scheduler never reads `mamba_block_size`, and the core's Mamba block-size
  resolution (#67994) finds no Mamba spec. Async scheduling depth (#64145)
  leaves `async_scheduling=False` in force. On main `6abe321bd` the smoke test
  passed (preemption, prefix hit next to MTP drafts) at or above 0.34.0's
  throughput, and Qwen3.5-0.8B and Qwen3-1.7B planned the same blocks with
  identical greedy output, sequential and concurrent.
- [#53558](https://github.com/vllm-project/vllm/pull/53558) (in 0.33.0) lets a
  platform supply a `KVCacheConfigBuilder`. The plugin keeps the default:
  Hybrid-PA exposes only attention specs and allocates conv/SSM state itself.

vLLM 0.34.0 (same tree as rc4, still `torch==2.14.0`/`torchvision==0.29.0`):
the smoke test passed on both frontends at or above 0.33.0's throughput,
Qwen3.5 and Gemma-4 image requests worked on both, and `prompt_logprobs` is
still rejected (400, or 500 under the Rust frontend). Qwen3.5-2B MTP with
prefix caching hit a repeated 2.3k-token prompt in 0.29 s (14.2 s cold) with
76% acceptance on rc4.

The copies track vLLM-side formats the plugin consumes. Since 0.33.0 the
grammar bitmask gives each structured-output request a fixed block of
`num_speculative_tokens + 1` rows, and `apply_grammar_bitmask` indexes it that
way; the 0.32 packed layout returned HTTP 500 on 0.33.0 when n-gram
speculation was on and several structured-output requests were batched.

### Why the plugin keeps its own runner

vLLM offers no upstream runner for an x86 plugin that cannot use Triton
(#61995 is non-x86 only); #57684 is closed on that basis. A prototype on
branch `mrv2-prototype` (vLLM 0.34.0rc4, `VLLM_OPENVINO_MRV2_PROTOTYPE=1`)
subclasses `StandaloneCPUModelRunner` and keeps only the model, the KV cache
and the forward call. On Qwen3-1.7B, greedy output, penalties, JSON schema,
`min_tokens` and `logit_bias` matched the plugin's runner and logprobs were
within 4e-5; single-stream decode was 11% slower (21.5 vs 24.2 tok/s) and 8
concurrent equal (153 tok/s). It needed:

- PyTorch replacements for eight of ten `vllm._cpu_ops`
  (`gather_block_table_cpu`, `post_update_cpu`, `gumbel_sample_cpu`, ...) that
  exist only in CPU builds of `_C` (seeded samples then differ from the C++ op);
- unwrapping the `torch.compile(fullgraph=True)` helpers
  (`compute_token_logprobs`, `_compute_token_ranks`, `_apply_min_p`), which
  raise under `TORCH_COMPILE_DISABLE=1`;
- overriding `execute_model`, `sample`, `load_model` and
  `initialize_kv_cache`, because there is no forward seam.

It loses drafts (`decode_query_len != 1` raises), so MTP and n-gram, and
hybrid models select `MambaHybridModelState`, which Hybrid-PA's state slots do
not fit.

`/v1/decisions` (on by default since 0.33.0) reads label probabilities from one
greedy token with `logprob_token_ids`, not `prompt_logprobs`, so it works;
concurrent and sequential requests returned the same probabilities.

## Serving Paths

The plugin selects a path from the model IR:

| Model shape | Path | Behavior |
|---|---|---|
| `ReadValue` with `ssm`/`conv` variable IDs, with SDPA | Hybrid-PA | External attention/state cache and concurrent batching |
| Any other model with SDPA (stateful KV exports and Gemma-4 included) | PagedAttention | External KV cache and concurrent batching |
| No SDPA | Rejected at startup | — |

The sequential stateful path that served IRs without SDPA was removed because
no tested model needed it. `VLLM_OPENVINO_STATEFUL_PA=0` is rejected at
startup and `=1` only logs that the variable is redundant.

### PagedAttention

Stateful plain-attention exports (optimum-intel defaults such as
Qwen2.5-Coder-int4-ov) are transformed at load time; `max_num_seqs` keeps its
default 128. The KV cache defaults to u8, the CPU plugin's own default on
x86_64/ARM64: with the variable unset the loader passes no
`KV_CACHE_PRECISION`. `VLLM_OPENVINO_KV_CACHE_PRECISION` selects
`u8`/`f16`/`bf16`; on a CPU whose runtime precision is bf16 the CPU plugin
stores `bf16` for an `f16` request (`scaled_attn.cpp`
`side_cache_precision`), and `f32`/`i8` are rejected by CPU PagedAttention and
fall back to the default with a warning. vLLM sees `cache_dtype="auto"` for
float caches: 0.34 rejects a `float16`/`bfloat16` cache dtype that differs from
the model dtype (bf16 here), which made `f16` fail at startup.
On Qwen2.5-Coder-0.5B, Qwen3-1.7B,
TinyLlama-1.1B and Phi-3.5-mini, concurrency-8 throughput was 4.9-7.0x the
removed sequential path and single-stream -7.8% to +4%. Greedy output matched
the stateful IR on TinyLlama and differed on Qwen-family models with u8 and
bf16 caches alike, so the difference comes from the transformed attention,
not cache precision.

Sliding-window models take the same path; the PagedAttention op applies the
window. Gemma-4 E2B and 26B-A4B gave the same greedy output as their stateful
IR on text and image prompts, with 6.7x / 3.2x its concurrency-8 throughput
(18.7 -> 125.2 and 9.8 -> 30.9 tok/s). When concurrent requests split long
prompts at different chunk boundaries, greedy output can change at near-tie
tokens.

Gemma-4 inputs keep their rank. `per_layer_inputs` is fed as
`[tokens, 1, layers, hidden]` (as in OpenVINO GenAI's `ModelRunner`); 26B-A4B
has hidden 0, so reshapes name the token count instead of `-1`. Without it the
first infer failed in a sliding-window layer (`Eltwise shape infer ... dim
index: 0 mismatch`). `token_type_ids` is fed as `[tokens, 1]` with the step's
real per-token types (1 on image embedding positions); the PA kernel reads it
(input 25), so tokens inside a contiguous type-1 run attend bidirectionally
and text stays causal. Adjacent runs with no separating token merge into one
block. The E2B IR has no `token_type_ids` input, so its image tokens stay
causal.

### Hybrid-PA

Hybrid-PA serves hybrid Mamba/attention models (Qwen3.5, LFM2.5, LFM3,
MiniCPM-V 4.7, Nemotron-H) with standard PagedAttention for attention layers
and paged conv/SSM state. Gemma-4 is not a candidate: its state is transformer
KV. The PA transformation names the SSM state input after the op:
`gated_delta_state_table.N` for GatedDeltaNet (Qwen3.5),
`selective_ssm_state_table.N` for Mamba2 (Nemotron-H); both page through the
same `la.*` slots, as in GenAI's `LinearAttentionCacheManager`.

Nemotron-H (NVIDIA-Nemotron-3-Nano-4B: 21 Mamba2, 4 attention layers, no
RoPE) was checked on a bf16 IR: greedy output matched the stateful IR token
for token on 7 prompts of 5-868 tokens, sequentially, as 7 concurrent
requests and on a prefix-cache hit. Export it with optimum-intel >= 0ee909b
(#2085), which caps Transformers below 5.16, and without
`--trust-remote-code`: the checkpoint's remote code imports the CUDA-only
`mamba-ssm`, while Transformers' built-in `nemotron_h` exports. The stateful
IR's own single-shot prefill of 869 tokens exceeded 16 GiB; the plugin's PA
path serves it within the same limit.

LFM3 (conv width 7, full and 256-token sliding-window attention) needs vLLM
0.34, whose `Lfm3ForCausalLM` and config validate the IR's `config.json`
without `--trust-remote-code` (0.33 needed a plugin-registered arch and remote
code, which failed under an arbitrary UID). Its IR is exported with OpenVINO
2026.5. A 1746-token
prompt gave byte-identical 400-token greedy output against the stateful IR.

Conv/SSM state storage depends on prefix caching.

**Prefix caching on (default).** Each scheduler block owns one conv/SSM
checkpoint slot with the same index, following OpenVINO GenAI's prefix
layout: read the checkpoint of the block holding token `num_computed - 1`,
write one per touched interval (`la.cache_interval` = scheduler block size).
Hits are block-aligned, so a hit restores the state at the last hit block.

- A scheduler block spans `N` 32-token KV blocks and the runner expands each
  block id to its `N` KV block ids. `N` is
  `VLLM_OPENVINO_LA_CACHE_INTERVAL_MULTIPLIER`, default 8 (256 tokens). `0`
  selects GenAI's `max(8, checkpoint bytes / KV block bytes)`: 8 for
  LFM2.5-350M (61 KB checkpoint), 47 (1504 tokens) for Qwen3.5-0.8B and
  MiniCPM-V 4.7 (10.3 MB). Checkpoints are part of the block page, so a 4 GiB
  pool holds 570k tokens for LFM2.5 and 90k for Qwen3.5 (280k at 1504 tokens).
- Checkpoints follow the CPU plugin's state precision: bf16 with AVX-512-BF16,
  f32 otherwise. Qwen3.5-0.8B's checkpoint is then 20.6 MB instead of 10.3 MB,
  and a limit that fits the bf16 pool can leave the f32 pool at the 256 MiB
  floor (11 blocks where one 4096-token request needs 16); the worker logs the
  remedies before vLLM's own ValueError.
- vLLM caches a block when it allocates it, so a request can hit a block that
  another request fills in the same step. The runner splits such a batch into
  waves (writer first) so the checkpoint read follows its write; without waves
  3 of 4 concurrent identical prompts had different logprobs. A drafting
  request has `1 + drafts` logits rows and MTP reads the main model's hidden
  states and embeddings at the request's token offset, so the runner
  concatenates every wave's rows and MTP inputs in batch order. With one row
  per request, a same-step hit next to a drafting request killed the engine
  (`IndexError` in `_sample_with_drafts`); `scripts/smoke-test.sh` covers it.
- With speculative decoding, a draft step reads the checkpoint (or the
  request's private slot) and writes per-token states (`la.cache_interval` 1)
  into temporary slots. A token that ends a block writes that block's
  checkpoint directly: if it is rejected the block is not full of known tokens,
  so vLLM has not cached it. The accepted state becomes the request's private
  slot, so nothing is copied. The temporary and private slots (`2 + k` per
  drafting request) come out of the KV pool.
- A prefix hit can change greedy output against a cold run whose chunk
  boundaries differ: restoring the bf16 checkpoint and reading past tokens
  from the u8 cache moved first-token logprobs by up to 0.07. With chunks on
  checkpoint boundaries (`--max-num-batched-tokens 256`, or 1504 at the
  automatic interval) hit and cold runs were bit-identical, image prompts whose
  hit boundary falls inside the image included.
- Hit TTFT: LFM2.5 2.1k-token prompt 0.74 -> 0.06 s; Qwen3.5 2.4k / 4.0k-token
  prompts 8.2 / 13.9 -> 0.20 / 0.56 s (2.9 / 3.4 s at 1504-token hits).
  Concurrency-8 decode was unchanged.
- Mixed-load stress test (8-16 concurrent greedy, sampled, prefix-hit, image,
  logprobs, JSON and aborted requests, with a per-step check that free +
  private + temporary state slots equal the pool): Qwen3.5-2B MTP for 60 min
  and for 20 min with a 7.9k-token pool (20 preemptions), Qwen3.5-0.8B n-gram
  k=4 for 20 min (9 preemptions). No errors and no slot drift.

**Prefix caching off.** Conv/SSM state lives in a separate slot pool of
`max_num_seqs + 1` slots, taken out of the KV pool first. Slots are released
when a request finishes or is preempted (`preempted_req_ids`, as MRV2 frees a
preempted request) and reset on a fresh or re-prefilled start; holding only
running requests, the pool cannot be exhausted. A recomputed request can
diverge late in the text from its uninterrupted output (prefill vs decode
numerics).

## Runtime Constraints

- CPU KV block size is `32`; GPU KV block size is `16`. Hybrid-PA with prefix
  caching uses a larger scheduler block (a multiple of the KV block size).
- `VLLM_OPENVINO_CPU_THREADS_NUM=0` uses the cgroup CPU quota when it is
  tighter than the visible cores and also caps the Torch/OMP pools unless
  `OMP_NUM_THREADS` is set. An explicit value takes precedence. On an
  8-quota/24-core host a 32-request burst went from 4.5 s wall with 3.1 s of
  CFS throttling to 3.6 s with none.
- The image sets `OMP_WAIT_POLICY=PASSIVE`. With libgomp's default spin-wait,
  the 3 extra Torch workers spun after every sampler call (even greedy
  `argmax`) and took 6.5 CPU-seconds per 10 s from OpenVINO's 4 threads under
  a 4-CPU quota. Passive waiting raised aggregate decode at 4 CPUs (LFM2.5-350M
  c4 / c16 208-214 / 611 -> 233-265 / 763 tok/s, Qwen3.5-0.8B 73 / 110 -> 82 /
  117) and at 8 CPUs by 3-9% (Qwen3-1.7B, Qwen3.5-2B c4-c16); sampling stayed
  as fast as before (64 x 248k-vocab top-k/top-p 52 ms; one thread takes
  102 ms, which is why the plugin does not set `OMP_NUM_THREADS=1` itself).
  libgomp reads the variable when `import torch` loads it, before the plugin
  runs, so it has to be in the process environment.
- Do not add zero-fill to `_allocate_kv_cache()`; OpenVINO initializes the
  cache and extra zero-fill can cause OOM. SSM/conv state caches are
  zero-filled. The worker zero-fills scheduler-reported
  `new_block_ids_to_zero` blocks as upstream does (no measurable cost).
- The KV cache is one fixed-size mapping whose pages become resident only as
  blocks are used, so RSS grows toward `baseline + pool` and then plateaus.
  Prefix caching retains freed blocks, so unique prompts keep touching new
  pages; `--no-enable-prefix-caching` holds the resident set at the working
  set.
- vLLM plans the block count from each layer's spec page, so that page must
  be the OpenVINO block's real size. u8 keys keep per-channel scale and zero
  point as extra rows (`[heads, 32 + 8, head]`); `get_kv_cache_spec` pads such
  layers (`page_size_padded`). Without the padding vLLM planned 8.8% more
  blocks than the budget on Qwen3-1.7B and the pool grew that much over it.
- CPU pool sizing (`VLLM_OPENVINO_KVCACHE_SPACE=0`), from a memory ceiling
  (cgroup limit, else host MemAvailable):
  - before model load (estimate): ceiling - `openvino_*.bin` sizes - 1.5 GiB
    runtime - APIServer overhead - 2.2 GiB headroom;
  - after model load and warmup: ceiling - EngineCore anonymous RSS -
    APIServer overhead - 2.2 GiB headroom, never below 256 MiB. Anonymous RSS
    excludes the memory-mapped weights, which the kernel reclaims (26B: VmRSS
    33.5 GB, 19.8 GB anonymous); evicting them did not slow decode;
  - APIServer overhead is 1.5 GiB for text models and 2.75 GiB with a vision
    encoder (measured 0.55-0.87 GiB text, 2.1-2.7 GiB vision);
  - an explicit value is only reduced when it does not fit, with a warning.
- The warmup must run the real maximum prefill. PA models run one prefill of
  `max_num_batched_tokens` and one decode on the null block 0, against a
  throwaway cache, inside `determine_available_memory` before the post-load
  measurement, so the first-infer weight repack and prefill buffers are in the
  baseline. Multimodal models also encode one maximum-size dummy image and
  prefill the first step of its prompt (skipped when the image limit is 0 or
  the prompt exceeds both 2048 tokens and one step). With a 16-token warmup
  the 26B baseline read 16.9 GB instead of 19.8 GB and the pool was OOM-killed
  once fully resident; leaving out the image put the baseline 0.5-0.6 GB low
  on gemma-4-E2B and MiniCPM-V. The warmup also halves first-request latency
  (Qwen3-1.7B 1.51 -> 0.79 s). Startup cost: 26 s on the 26B, 11 s on E2B.
- Hybrid-PA conv/SSM slots come out of the KV pool first: without prefix
  caching `max_num_seqs + 1` plus `1 + k` per drafting request; with prefix
  caching `2 + k` per drafting request. At most `min(max_num_seqs, 8)`
  requests draft per step. A Qwen3.6-35B-A3B slot is about 64 MiB, so the
  default 128 requests reserve 8 GiB; lower `--max-num-seqs`. The startup
  warning reports the reservation.
- Images are encoded when vLLM lists them in `scheduled_encoder_inputs`; the
  merged embeddings stay in the runner's encoder cache until
  `free_encoder_mm_hashes`, so an image split across steps is encoded once and
  a repeated image behind a different prefix skips the vision tower.
  Embeddings are scattered by `(batch_start, batch_end, patch_offset,
  image_index)` rows, one per contiguous `is_embed` run. For text-only serving
  of a multimodal checkpoint, `--language-model-only` or
  `--limit-mm-per-prompt '{"image":0,"video":0}'` skips the vision encoder and
  its warmup and budgets the text APIServer overhead (gemma-4-E2B at 16 GiB:
  pool 7.21 -> 9.59 GB).
- bf16 tensors must be converted to float32 before NumPy conversion because
  OpenVINO does not accept bf16 NumPy arrays.
- `paged_attention_transformation()` and gather-before-matmul are applied to
  every model.
- OpenVINO import failure must stay deferred until configuration validation
  so plugin discovery can inspect other platforms safely.

## Model Files

- Text model: `openvino_model.xml`
- Language model: `openvino_language_model.xml`
- Text embeddings: `openvino_text_embeddings_model.xml`
- Vision encoder: `openvino_vision_embeddings_model.xml`
- Vision merger: `openvino_vision_embeddings_merger_model.xml`
- Vision position embeddings (Qwen3.5/3.6): `openvino_vision_embeddings_pos_model.xml`
- MTP head (Qwen3.5/3.6): `openvino_mtp_model.xml`

The vision encoder expects a 2D `[num_patches, features]` input. Vision merger
calls require `hidden_states`, `attention_mask`, and `rotary_pos_emb`.
Qwen3.5 multimodal inputs use `pixel_values` and `image_grid_thw`; Gemma-4
uses `pixel_values`.

### Qwen3.5 / Qwen3.6 images

Same steps as OpenVINO GenAI's `qwen3_vl` and `qwen3_5`; Qwen3.5-0.8B gives
the same answer as GenAI 2026.4.1 on the same IR, also with chunked prefill,
cached images and concurrent requests:

- vLLM enables `mm_device_do_normalize` for this family, so `pixel_values`
  arrive as raw uint8. The loader applies vLLM's `build_mm_input_norm`.
- The position IR takes the four bilinear corner indices `[4, n]` of the 48x48
  learned grid; the weighted sum is permuted to merge order and added to the
  patch embeddings before the merger.
- `rotary_pos_emb` width comes from the merger IR (32 on Qwen3.5-0.8B, 36 on
  Qwen3.6-35B-A3B), with `dim/2` frequencies per axis.
- The LM's `position_ids [4, n]` carry text positions in row 0 and M-RoPE
  (t, h, w) in rows 1-3, computed by vLLM's `Qwen3VLForConditionalGeneration`;
  decode positions are `index + rope_delta`.

### MiniCPM-V 4.7

The vision IR takes one packed crop per infer: `pixel_values [1,3,14,h*w*14]`
plus host-built `position_ids`, `window_index` and `merge_index` (same
algorithms as OpenVINO GenAI's `VisionEncoderMiniCPMv4_7`) and returns merged
tokens `[tokens, hidden]`. vLLM's zero-padded `pixel_values
[n_slices,3,14,L_max]` is cut by `tgt_sizes` (passed through the
`image_grid_thw` slot). Slice placeholders interleave `<slice>`/newline tokens
with `<|image_pad|>` runs, so embeddings go to the `is_embed` runs only. The LM
takes `[3, n]` M-RoPE positions from vLLM's
`MiniCPMV4_7ForConditionalGeneration`; decode positions are
`index + rope_delta`.

Requires vLLM >= 0.32.0 (#58674, #56372) and Transformers >= 5.18.0.
MiniCPM-V-4.7-1B-int4 misreads ~10 px text ("HELLO 2026" as "HELLO 2028")
exactly as OpenVINO GenAI does on the same IR, so this is the model.

## Sampler

`sampler.py` follows vLLM 0.32's V1 `Sampler` and its builtin `min_p`,
`logit_bias` and `min_tokens` processors step for step. An all-greedy batch
returns after a bare `argmax`, as upstream does.

- Random draws: vLLM's CPU path uses a C++ Gumbel-max kernel the wheel does
  not ship. The plugin samples softmax -> cumsum -> `searchsorted(right=True)`
  with one uniform per row from the per-row seed vLLM passes (from the
  request's `torch.Generator` when it sets `seed`). 20k draws matched the
  softmax probabilities within 0.4 points and never chose a masked token;
  `seed=42` requests were identical alone and batched with unseeded ones.
- `VLLM_OPENVINO_FAST_SAMPLER=1` (default) replaces two routines whose cost
  scales with `(batch, vocab)`: top-k/top-p takes a partial `topk` over
  `max(k)` candidates (1024 when only `top_p` is set) and falls back to the
  full sort when they hold less than `p` of the mass; the repetition penalty
  is applied only at prompt/output token positions (presence/frequency
  penalties take the dense path). At batch 8, vocab 151,936: top-k+top-p
  17.8 -> 0.9 ms, repetition penalty 3.3 -> 0.3 ms.
- Equivalence: 56 cases each (greedy, seeded and unseeded random,
  top-k/top-p, penalties, logprobs, logprob token ids, allowed tokens, bad
  words) gave bit-identical tokens and logprobs against vLLM 0.32's `Sampler`,
  with `FAST_SAMPLER` on and off.
- An in-graph TopK was not pursued: the host-side partial `topk` leaves at
  most 0.9 ms to save, and OpenVINO GenAI also samples host-side over full
  logits.

### Thinking Token Budget

`thinking_token_budget` follows MRV2's CPU implementation
(`CPUThinkingBudgetState`, `vllm/v1/worker/cpu/sampler.py`): a request is
reasoning while its last reasoning start marker follows its last natural end
marker, and once the tokens after that start marker (prompt included) reach
the budget, the next token of `reasoning_end_str` gets logit `1e9` after every
other logits stage and before temperature. Marker positions are cached per
row, so each step scans only new tokens. vLLM rejects the parameter unless the
server runs with `--reasoning-parser` (or `--reasoning-config`). Requests with
a budget get no drafts. Qwen3-1.7B (greedy, 600 tokens): reasoning of 1926
characters unbudgeted, 0 / 122 / 452 at budgets 0 / 32 / 128; concurrent
requests returned the same text as single ones, also with n-gram on.

## N-gram Speculative Decoding

`method=ngram` is accepted on PagedAttention and Hybrid-PA models, `mtp` on
Hybrid-PA IRs that ship an MTP model (next section); `check_and_update_config`
raises for any other method. `execute_model` computes logits for the 1+k query
positions of each draft request, and `sample_tokens` verifies them (the stock
rejection sampler needs Triton). Greedy requests accept drafts while they
match the argmax. Sampled requests use speculative sampling as OpenVINO GenAI
does since openvino.genai#4769: accept draft d with probability
min(1, p(d)/q(d)), else emit a sample of norm(max(0, p - q)) and stop, where p
is the sampler's distribution after temperature, min_p, top-k and top-p and q
is the one the draft was drawn from (one-hot for n-gram), so the output
follows p alone. Drafts are proposed only for requests without penalties,
`logit_bias`, `allowed_token_ids`, `bad_words`, `min_tokens`, logprobs, or
structured output; other requests in the batch use the normal sampler.
Drafts return through `take_draft_token_ids`, so the platform turns async
scheduling off for both methods: vLLM 0.34 enables it for ngram, and the
scheduler then sends `-1` placeholders instead of the drafts, which accepted
no draft and changed greedy output. Model
`generation_config.json` defaults count: LFM2.5 ships `repetition_penalty:
1.05`, so no request drafts unless it sets `repetition_penalty=1.0` or the
server runs with `--generation-config vllm`.

Hybrid-PA: a draft request runs with `la.cache_interval=1` and
`la.block_indices = [committed slot, 1+d temporary slots]`, so token t's
state lands in temporary slot t; after verification the last accepted slot
becomes the committed slot. Drafts go to at most 8 requests per step
(`MAX_DRAFTING_REQS`), so the pool holds
`max_num_seqs + 1 + min(max_num_seqs, 8) * (k + 1)` slots; past 8 concurrent
requests verification is no longer free on CPU (table below). Without the cap
Qwen3.5-0.8B with k=4 and the default 128 requests reserved 14.8 GiB and did
not start in 16 GiB.

Measured (8 CPUs, k=4, `prompt_lookup_min=5`, `prompt_lookup_max=8`,
tok/s, baseline -> spec; copy-heavy / free-form output):

| Model | 1 request | 4 concurrent | 8 concurrent | 16 concurrent |
|---|---|---|---|---|
| Qwen3-1.7B (PA) | 23 -> 57 / 23 -> 23 | 82 -> 148 / 82 -> 85 | 154 -> 282 / 160 -> 149 | 268 -> 358 / 267 -> 274 |
| Qwen3.5-2B (Hybrid-PA) | 21 -> 43 / 22 -> 21 | 56 -> 92 / 64 -> 64 | 86 -> 114 / 102 -> 90 | - -> 124 / - |
| LFM2.5-8B-A1B (Hybrid-PA, MoE) | 23 -> 42 / 25 -> 50 | 44 -> 60 / 49 -> 78 | 56 -> 69 / 66 -> 88 | - |

LFM2.5-8B-A1B ran with `--generation-config vllm`; its free-form prompt loops
under greedy decoding without a repetition penalty, so that column is not
representative. Free-form output is neutral up to 4 concurrent requests and
loses up to ~10% on Hybrid-PA at 8. Keep `prompt_lookup_min` high: with 2,
spurious matches cost 7-11% on free-form output. Output can diverge from
non-spec runs at near-tie tokens because kernel numerics vary with query
length. On gemma-4-E2B copying 300 tokens of code ran at 28 tok/s including
prefill (17.5 without speculation), 138 of 152 drafts accepted.

## MTP Speculative Decoding

Qwen3.5/3.6 exports ship `openvino_mtp_model.xml`, a one-layer
multi-token-prediction head, and `--speculative-config
'{"method":"mtp","num_speculative_tokens":1}'` drafts with it, following
OpenVINO GenAI's MTP strategy (`mtp_model_transforms.cpp`):

- The loader adds a `last_hidden_state` output to the main model (the lm_head
  input of every token, before gather-before-matmul) and grafts the main
  lm_head onto the PagedAttention-transformed MTP model, gathered at the
  sampled rows.
- The MTP layer at position i reads the main hidden state h_i and the
  embedding of token i+1 (the main step's own input embedding when that token
  was in the step, so image positions keep their image embedding) and
  predicts token i+2, at position i (the vLLM EAGLE convention).
- Its KV cache is one more attention layer behind the main layers on the same
  block table, so vLLM budgets it with them; the main model gets
  `kv_caches[:num_main_kv_layers]`.
- After each step's verification, the MTP layer catches up on the positions
  the main model committed (prefill chunks included), which gives the first
  draft. Each further draft feeds the MTP hidden state and the previous draft
  one position later; that draft-only KV is overwritten by the next catch-up
  and stays within the k lookahead slots the scheduler reserves.
- Requests that cannot take drafts (same rules as n-gram) skip the MTP layer.
  Sampled requests draw each draft from the sampler's distribution over the
  draft logits (top-k clamped to the draft vocabulary) and keep it as q for
  verification.

The full lm_head (248,320 x 2,048 u8, 508 MB on Qwen3.5-2B) made a k=1 draft
step 18 ms against 49 ms for the main model. The draft lm_head keeps token
ids below `VLLM_OPENVINO_MTP_DRAFT_VOCAB` (default 32768) plus every token
whose text contains a `VLLM_OPENVINO_MTP_DRAFT_CHARS` character (default
`가-힣`, Hangul syllables), copied at load time from the rows of the
compressed weight and its per-row scale and zero point, so it stays a
compressed matmul (6 ms at 32k). The drafter maps draft logits back to token
ids and spreads q over them for verification. Drafts are verified, so this
changes speed, not output. A `Gather` on the compressed weight was slower
than the full lm_head.

The low ids alone cover English and code but not Korean: byte-level BPE
orders tokens by merge rank, and only 36 of Qwen3.5's 6,807 Hangul tokens are
below 32k (median id 189k). On Qwen3.5-2B's own outputs the first 32k ids
covered 45% of Korean tokens and 90% of English ones; adding the Hangul
tokens (39,525 rows) covers 96% of Korean and leaves English unchanged.
Tokenizing the candidate tokens adds about 1.7 s to an MTP start.

Measured (8 CPUs, greedy, 256 tokens, mostly English/code prompts; tok/s for
1 request / 4 / 8 concurrent):

| Qwen3.5-2B-int4 | 1 | 4 | 8 | acceptance |
|---|---|---|---|---|
| no speculation | 20.8 | 61.7 | 94.0 | - |
| k=1, whole vocabulary | 24.0 | 67.6 | 95.5 | 0.75 |
| k=1, 32k | 27.4 | 69.2 | 94.7 | 0.73 |
| k=2, 32k | 29.1 | 65.9 | 88.5 | 0.65, 0.41 |
| k=1, 16k | 27.7 | 68.4 | 94.5 | 0.66 |

| Qwen3.6-35B-A3B-int4 (32 GiB, `--max-num-seqs 4`) | 1 | 4 | acceptance |
|---|---|---|---|
| no speculation | 10.8 | 18.1 | - |
| k=1, whole vocabulary | 12.7 | 19.7 | 0.86 |
| k=1, 32k | 13.8 | 19.4 | 0.81 |
| k=2, 32k | 14.7 | 19.4 | 0.73, 0.62 |

Sampled (`temperature=0.7, top_k=20, top_p=0.8`, 256 tokens, one request):
Qwen3.5-2B 20.2-20.7 -> 25.5 tok/s at k=1 (acceptance 0.59) and 24.3 at k=2
(0.40 per draft). The verifier reproduced p on synthetic distributions
(total variation 0.002-0.005 over 200k draws, acceptance equal to
sum(min(p, q))), and 600 seeded `temperature=1.0` requests gave per-position
output distributions as far from two non-speculative runs as those runs are
from each other.

Korean prompts (Qwen3.5-2B k=1, 256 tokens, thinking off; tok/s for 1
request greedy / 1 sampled / 4 / 8 concurrent greedy, acceptance greedy,
sampled):

| draft vocabulary | 1 greedy | 1 sampled | 4 | 8 | acceptance |
|---|---|---|---|---|---|
| no speculation | 21.1 | 20.6 | 58.6 | 77.2 | - |
| first 32k ids | 23.5-24.3 | 20.9-21.1 | 56.4 | 71.9 | 0.34, 0.32 |
| first 32k + Hangul | 27.2-27.9 | 24.9-25.0 | 63.8 | 80.9 | 0.58, 0.62 |

English on the same runs was within noise between the two vocabularies
(29.3-30.7 tok/s single, 68.9-69.8 at 4, 88.2-89.5 at 8, acceptance 0.71
on both). Keeping 32k rows in total (first 26k + Hangul) lowered Korean
sampled acceptance to 0.54. Verifying two tokens costs 1.43x a decode
step on the MoE model against ~1.05x on the dense one, so concurrency erases
the gain sooner there. Use k=1 for mixed load; k=2 helps a single stream and
loses at 8 concurrent requests.

Gemma-4 MTP is not supported. Its drafter is a separate 4-layer model
(`google/gemma-4-*-it-assistant`) whose attention layers have only Q
projections and read K/V from the target's last full-attention and last
sliding-window layers. No OpenVINO IR of it exists (optimum-intel #1763 and
#1911 are unmerged), and OpenVINO GenAI 2026.4 runs it only in its stateful
SDPA pipeline at batch 1, on a target exported with extra outputs
(`mtp_last_hidden_state`, `mtp_{full,sliding}_attention_{key,value}`). A PA
version would need that export plus a drafter whose attention reads the
target layers' paged u8 cache. Use n-gram on Gemma-4.

## Performance Profile

Measured on Ryzen 9 7900 (no AMX), 8-CPU quota, OpenVINO 2026.4.1, greedy
decode. Where the time goes, so tuning effort is not spent in the wrong place:

- **Runner overhead is negligible.** Qwen2.5-Coder-0.5B: client inter-token
  latency 13.7 ms (1 request) / 19.8 ms (8) vs step time 13.9 / 20.1 ms;
  `infer` is ~95% of the step, input preparation 0.1 ms, sampling 0.5-1.1 ms.
  The same IR as a plain stateful model decodes in 13.3-14.5 ms, so the
  PagedAttention transformation costs nothing.
- **Decode is memory-bandwidth bound** at ~27-32 GB/s for dense and MoE alike
  (weight bytes per token over step time: Qwen3-1.7B 1.18 GB in 36 ms,
  LFM2.5-8B-A1B 1.0 GB in 34 ms, Gemma-4 26B-A4B 2.4 GB in 90 ms), against a
  45-48 GB/s read ceiling on this machine. Plain OpenVINO decode of Qwen3-1.7B
  is flat from 4 to 24 threads (33.5-34.5 ms), so a single stream needs ~4
  CPUs; only reading fewer bytes makes it faster. On MoE models
  `GatherMatmulCompressed` is 62-78% of node time.
- **KV precision: the `u8` default is the fastest.** Qwen3-1.7B decode at
  1.2k / 4.5k / 8.7k / 17.5k context: `u8` 36 / 43 / 51 / 70 ms, `bf16`
  37 / 48 / 61 / 88 ms; prefill is the same. On 80 teacher-forced positions
  per model (500-12k-token prompts) `u8` vs `bf16` top-1 agreement was 79/80
  on Qwen3-1.7B and Qwen2.5-Coder-0.5B, mean top-20 KL 0.007 / 0.016 nats,
  not growing with context; the flips were near ties. This bounds the effect;
  it does not replace a task benchmark.
- **Prefill is at the matmul ceiling** (~2.5 TFLOP/s; Qwen3-1.7B ~840-890
  tok/s at 1k tokens) and slows with context (845 / 732 / 607 / 403 tok/s at
  the four lengths above) as attention grows to about half the time. The PA
  op's XAttention sparse prefill is not used: the CPU kernel applies it only
  to single-sequence steps and it approximates attention.
- **Dynamic activation quantization** is the one prefill lever found.
  `VLLM_OPENVINO_DYNAMIC_QUANTIZATION_GROUP_SIZE` (unset = OpenVINO's 32);
  56 first-token distributions per model, top-20 KL against `0`
  (unquantized activations):

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

  Decode is unchanged. The extra deviation of 64/128 is within this sample's
  noise, but the default was kept because the cost could not be ruled out.
- **Other CPU compile properties** (Qwen3-1.7B, concurrency 1/8) are best at
  OpenVINO's defaults: `INFERENCE_PRECISION` (already bf16), `ENABLE_CPU_PINNING`
  (already on). `ENABLE_HYPER_THREADING=false` cost 13% at concurrency 8 and
  `KEY/VALUE_CACHE_GROUP_SIZE=64` did not help. `PERFORMANCE_MODE=THROUGHPUT`
  gave +28-31% single-stream and +10% at concurrency 8 over LATENCY on
  Qwen2.5-Coder-0.5B, at 25 -> 34 ms first-token latency.
  `NUM_STREAMS=AUTO` fragments threads across streams vLLM never runs
  concurrently (one blocking `infer` per step): `NUM_STREAMS=1` was 3.56x
  faster at concurrency 8 on Qwen3-1.7B and 2.4-2.7x on Hybrid-PA models.
- **Concurrency scaling** (decode-heavy, 256 output tokens, aggregate tok/s):

  | N | 1 | 4 | 16 | 32 | 64 | 128 |
  |---|---|---|---|---|---|---|
  | Qwen3-1.7B-int4 | 29 | 101 | 327 | 469 | 508 | 532 |
  | Qwen2.5-Coder-0.5B-int4 | 86 | 236 | 757 | 1125 | 1228 | 1340 |
  | LFM2.5-8B-A1B-int4 (MoE) | 32 | 59 | 121 | 157 | 228 | 296 |

  Dense models scale almost linearly to N=8-16 and gain only 5-15% from 32 to
  128. MoE scales worse because a larger batch reads more experts. With 4
  CPUs Qwen3-1.7B matches 8 CPUs up to N=4 but flattens at 318 tok/s.
- **Multi-socket** (2-socket Xeon E5-2670 v3, 12 cores per socket,
  Qwen3.5-0.8B, guidellm output tok/s at 2 / 8 streams): 8 threads on one
  node 14.7 / 22.7, split 4+4 across nodes 13.0 / 18.6; 12 threads on one
  node 18.5 / 27.6, 24 across both 18.7 / 30.0. `numactl --interleave=all`
  did not close the gap (29.7), so the limit is bandwidth, not locality.
- **AVX2-only CPUs (emulated).** `OV_CPU_MAX_ISA=AVX2` drops inference
  precision to `f32` while keeping Zen 4's caches and bandwidth, so it
  approximates the ISA of Haswell/Zen 2-3, not their speed. Qwen3-1.7B, host
  shared with VMs (compare rows with each other):

  | | prefill 0.9k / 3.7k (tok/s) | decode c=1 (ms/tok) | c=8 / c=32 (tok/s) |
  |---|---|---|---|
  | int4, native (bf16) | 719 / 627 | 42.7 | 153 / 402 |
  | int4, AVX2 | 307-312 / 285 | 41.7-42.5 | 96-99 / 198 |
  | int8, AVX2 | 304 / 279 | 56.6 | 86 / 189 |
  | fp16, AVX2 | 295 / 273 | 111.3 | 58 / 154 |

  Single-stream decode is unchanged; prefill is 2.2-2.3x and batched decode
  1.6-2x slower, all from f32 matmul throughput. int4 is the fastest format
  because it reads the fewest bytes. Dynamic quantization group size has no
  effect there. N-gram (k=4) still cut copy-heavy single-stream decode from
  44.4 to 31.1 ms/tok.

## Comparison with OVMS and vLLM's CPU Backend

Same host, `--cpus=8 --memory=16g`, same client (chat completions, greedy,
256 tokens with `ignore_eos`, streaming). Decode is single-stream tok/s after
the first token; c4-c16 are aggregate tok/s over one wave of concurrent
requests; prefill is a 2.1k-token prompt. Plugin `113e404` on vLLM 0.33.0 and
OpenVINO 2026.5 nightly; OVMS 2026.4.1 (`--task text_generation`, defaults
plus `--rest_workers 64`); `vllm/vllm-openai-cpu` 0.33.0
(`VLLM_CPU_KVCACHE_SPACE=4`, threads bound to 8 CPUs). Two alternating runs
each; they agreed within 5%.

| Qwen3-1.7B | decode | c4 | c8 | c16 | c16 TTFT | prefill tok/s | peak memory |
|---|---|---|---|---|---|---|---|
| plugin, int4 IR | 22.3 | 78 | 147 | 254 | 0.55 s | 604-641 | 4.8-5.1 GiB |
| OVMS, same IR | 21.5-22.6 | 83-86 | 151-155 | 146-157 | 7.0-7.4 s | 606-611 | 1.6 GiB |
| vLLM CPU, w4a16 | 15.1 | 51 | 83 | 115 | 1.2 s | 263 | 9.4 GiB |
| vLLM CPU, bf16 | 8.0 | 32 | 60 | 112 | 0.7 s | 520 | 11.3 GiB |

| Qwen3.5-2B (hybrid) | decode | c4 | c8 | c16 | c16 TTFT | prefill tok/s | peak memory |
|---|---|---|---|---|---|---|---|
| plugin, int4 IR | 21.0-21.2 | 59-60 | 96 | 131-133 | 2.0-2.2 s | 213-220 | 7.3-9.2 GiB |
| OVMS, same IR | 21.3 | 64 | 100 | 96 | 11.7 s | 198 | 1.8 GiB |

- On the same IR the plugin and OVMS run at the same speed up to 8
  concurrent requests; both execute the same OpenVINO PagedAttention kernels.
  The plugin rows above predate `OMP_WAIT_POLICY=PASSIVE`, which was the
  5-8% c4-c8 gap: with it Qwen3-1.7B gave 82 / 153 / 261 and Qwen3.5-2B
  64 / 97 / 133 tok/s at c4 / c8 / c16. At 4 CPUs / 8 GiB
  (`--max-num-batched-tokens 512`) the plugin gave LFM2.5-350M-int8 63-69 /
  233-265 / 763 tok/s at c1 / c4 / c16 against OVMS's 69-71 / 259 / 270, and
  Qwen3.5-0.8B-int4 35 / 82 / 117 against 32 / 78 / 78.
- Accuracy matches OVMS on the same IR (4 CPUs, greedy, OVMS 2026.4.1, first
  300 GSM8K test questions, every 28th of MMLU's 14k test questions with the
  letter answer):

  | | GSM8K | MMLU (500) |
  |---|---|---|
  | LFM2.5-350M-int8, OVMS / plugin / plugin DQ 128 | 38.3 / 38.3 / 36.7% | 32.6 / 32.4 / 33.2% |
  | Qwen3.5-0.8B-int4 (no thinking), OVMS / plugin / plugin DQ 128 | 37.7 / 37.7 / 43.0% | 33.8 / 33.0 / 31.4% |

  The plugin's text matched OVMS's on 291/300 and 495/500 (LFM2.5) and 275/300
  and 483/500 (Qwen3.5) after stripping a trailing newline OVMS emits; no
  GSM8K question was solved by only one of them. Dynamic quantization group
  128 changes outputs at near-ties (57/300 identical on LFM2.5) without a
  consistent direction: Qwen3.5 gained 16 GSM8K questions and lost 8 MMLU
  ones.
- OVMS admitted only 8 requests at a time here: its executor log showed at
  most 8 scheduled requests, so c16 ran in two waves (unary and streaming
  alike). Its default `rest_workers` (the CPU count) capped it at 4 until set
  to 64, and no documented option lifted the 8. The plugin batches all 16.
- vLLM's CPU backend is 1.5x (w4a16) to 2.8x (bf16) slower single-stream and
  its w4a16 prefill is 2.3x slower; it reserves its KV space up front.
- Memory: OVMS grows its KV cache on demand (`cache_size 0`); the plugin's
  default pool is sized to the container limit and its pages become resident
  as blocks are used, so its peak reflects the pool touched by the run, not a
  floor. vLLM CPU's figure includes its 4 GiB KV space.

## Known Limitations

- Models without SDPA ops are not supported.
- Qwen3.5 text-only exports (`openvino_model.xml`, `Qwen3_5ForCausalLM`) are
  rejected at startup. They build RoPE positions from the sequence length and
  leave `position_ids` unused, so after the PagedAttention transformation
  every token of a step gets position `max_context_len - num_tokens`: all
  prompt tokens land on position 0 and batched requests share one position.
  Output still reads well because only 6 of 24 layers are attention, but
  greedy tokens drift from the SDPA reference (an 869-token prompt diverged at
  token 4 at a 1.24-logit margin). openvinotoolkit/openvino#39652 fixes the
  transformation on master, after 2026.5.1. The loader rejects an IR that
  declares `position_ids` but has no consumer for it after the transformation;
  every other local IR, LFM2.5/LFM3 and the MTP heads included, has one.
  NoPE models (Nemotron-H) declare no `position_ids`, and the transformation
  adds an unused one, so they pass. Multimodal exports
  (`openvino_language_model.xml`) take `position_ids` and are not affected.
- LoRA, pin memory, and multi-socket execution are not supported.
- `prompt_logprobs` is not supported: the gather-before-matmul transformation
  computes logits only at sampled positions. `OpenVinoPlatform.validate_request`
  rejects such requests with HTTP 400. Output-token `logprobs` work.
- On OpenVINO 2026.4.x, MoE prefill is slow on CPUs without AMX
  (LFM2.5-8B-A1B ~108 tok/s, Qwen3.6-35B-A3B ~74 tok/s) because the CPU plugin
  batches routed rows per expert into a GEMM only on bf16+AMX.
  openvinotoolkit/openvino#39086 fixes it in 2026.5; the image's nightly has
  it.
- `VLLM_USE_RUST_FRONTEND=1` replaces the Python API server with `vllm-rs`
  over the same engine boundary, so the plugin is unchanged; the smoke test
  passes on every path at the same throughput, and image (Qwen3.5, Gemma-4)
  and JSON-schema requests give the same answers. Two differences:
  - `validate_request` is called only by the Python frontend, so
    `prompt_logprobs` fails with HTTP 500 instead of 400.
    [#67623](https://github.com/vllm-project/vllm/pull/67623) (open) runs the
    check in the engine core; with it both frontends return 400.
  - The Rust frontend has no multimodal spec for MiniCPM-V 4.7: image
    requests get HTTP 400. Serve it with the Python frontend.
