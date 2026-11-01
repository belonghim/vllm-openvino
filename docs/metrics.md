# Metrics

The API server exposes vLLM's Prometheus metrics at `/metrics` on the serving
port (`curl -s http://127.0.0.1:8080/metrics`). Every `vllm:*` metric comes
from vLLM itself: the scheduler in the EngineCore process and the frontend in
the APIServer process. The plugin adds no metric. Its own per-step timing is
the `[OV-RUNNER]` log line (below).

Checked on vLLM 0.32.0 with Qwen3-1.7B-int4-ov (PagedAttention, n-gram
speculation) and Qwen3.5-0.8B-int4-ov (Hybrid-PA, `--enable-mfu-metrics`),
8 CPUs. The stateful path was not checked.

## What Is Populated

| Metrics | Status on this plugin |
|---|---|
| `num_requests_running`, `num_requests_waiting`, `num_requests_waiting_by_reason` | Populated |
| `kv_cache_usage_perc` | Populated; fraction of CPU pool blocks in use (see below) |
| `prefix_cache_queries_total`, `prefix_cache_hits_total`, `prompt_tokens_cached*`, `prompt_tokens_by_source_total` | Populated where prefix caching is on |
| `prompt_tokens_total`, `generation_tokens_total`, `iteration_tokens_total`, `request_*_tokens`, `request_params_*` | Populated |
| `time_to_first_token_seconds`, `inter_token_latency_seconds`, `request_time_per_output_token_seconds`, `e2e_request_latency_seconds` | Populated |
| `request_queue_time_seconds`, `request_inference_time_seconds`, `request_prefill_time_seconds`, `request_decode_time_seconds` | Populated |
| `request_success_total{finished_reason}` | Populated; rejected requests (HTTP 400, e.g. `prompt_logprobs`) are not counted |
| `num_preemptions_total`, `request_num_preemptions` | Populated |
| `spec_decode_*` | Populated with `--speculative-config` (`method=ngram`) |
| `estimated_flops_per_gpu_total`, `estimated_read_bytes_per_gpu_total`, `estimated_write_bytes_per_gpu_total` | 0 unless `--enable-mfu-metrics` |
| `mm_cache_queries_total`, `mm_cache_hits_total` | Multimodal requests only |
| `cache_config_info` | Populated; includes plugin fields (see below) |
| `external_prefix_cache_*`, `prompt_tokens_cached_by_source{source!="device"}` | Always 0: no KV connector or offloading |
| `admission_rejections_total` | 0 unless queue limits are configured |
| `process_*`, `python_gc_*`, `http_*` | APIServer process only |

## Reading Them on CPU

**Gauges update once per engine step.** The scheduler attaches its stats to
every step's output, so `num_requests_running` and `kv_cache_usage_perc` keep
the previous step's value until the current step finishes. CPU prefill steps
can take seconds: on Qwen3.5-0.8B four ~2.4k-token prompts arriving together
gave a 16.5 s step, and scrapes kept reading the idle values (0 running, 0%)
for about 21 s. A Prometheus scrape inside a long prefill sees stale gauges;
counters and histograms catch up when the step ends.
`--max-num-batched-tokens 512` shortens the steps (README, Chunked Prefill
Size).

**`kv_cache_usage_perc` counts blocks, not memory.** It is used blocks over
`num_gpu_blocks`, which on CPU is the size of the plugin's KV pool. The pool
is one fixed mapping whose pages become resident only as blocks are first
used, and prefix caching keeps freed blocks, so this gauge can drop to 0%
while the container's resident memory stays at its peak. Watch the cgroup
memory (container metrics) for memory; this gauge only tells how close the
scheduler is to preempting. On Hybrid-PA with prefix caching a block is a
scheduler block (256 tokens by default, `block_size="256"` in
`cache_config_info`), and its page includes the conv/SSM checkpoint, so the
same pool holds fewer tokens than on a pure-attention model.

**`cache_config_info` labels.**

| Label | Meaning on this plugin |
|---|---|
| `block_size` | 32 (CPU KV block), or the Hybrid-PA scheduler block |
| `num_gpu_blocks` | Blocks in the CPU KV pool |
| `kv_cache_size_tokens`, `kv_cache_max_concurrency` | `num_gpu_blocks x block_size`, and that over `max-model-len` |
| `openvino_kvcache_space_bytes` | Final pool size in bytes, after the post-load sizing |
| `openvino_kv_dtype` | `VLLM_OPENVINO_KV_CACHE_PRECISION`; `None` when unset, in which case OpenVINO picks the precision (u8 on CPU) |
| `cache_dtype` | Stays `auto`; it does not show the OpenVINO KV precision |
| `gpu_memory_utilization` | Ignored on CPU; the pool is sized from `VLLM_OPENVINO_KVCACHE_SPACE` or the memory limit |

**Memory is not in `/metrics`.** `process_resident_memory_bytes` and the other
`process_*` metrics describe the APIServer process. The weights, the KV pool
and the inference buffers live in the EngineCore process, which `/metrics`
does not report; use the cgroup's `memory.current` / `memory.stat` (container
or node exporter). The sizing rules are in the README (Memory Footprint).

**Prefix cache counters.** Prefix caching is turned off for the stateful path
and for Hybrid-PA with speculative decoding, so no hits are recorded there. On
Hybrid-PA hits are block-aligned, so `prefix_cache_hits_total` grows in
multiples of the scheduler block (256 tokens by default).

**Speculative decoding.** Drafts are proposed only for greedy requests without
penalties, logprobs or structured output (README, Compatibility), so
`spec_decode_num_drafts_total` counts only those requests. The acceptance rate
is `spec_decode_num_accepted_tokens_total / spec_decode_num_draft_tokens_total`;
a copy prompt on Qwen3-1.7B accepted 104 of 104 draft tokens. A model whose
`generation_config.json` sets `repetition_penalty` gets no drafts at all, which
shows up as a flat `spec_decode_num_drafts_total`.

**MFU estimates.** With `--enable-mfu-metrics` vLLM computes FLOPs and bytes
per step analytically from the model config; the plugin does not measure them.
`per_gpu` means per engine here. Compare
`rate(vllm:estimated_flops_per_gpu_total[1m])` with the ~2.5 TFLOP/s prefill
ceiling and `rate(vllm:estimated_read_bytes_per_gpu_total[1m])` with the
45-48 GB/s read ceiling measured on the Ryzen 9 7900 (compatibility.md,
Performance Profile). The estimates were not checked against measured counts,
including on Hybrid-PA layers.

**Multimodal cache.** `mm_cache_*` counts vLLM's multimodal processor cache in
the APIServer. The runner's vision-embedding cache (README, Vision-embedding
cache) has no metric; a hit there shows only as a shorter TTFT.

## Log Lines

Both lines are written every `VLLM_LOG_STATS_INTERVAL` seconds (default 10);
the `[OV-RUNNER]` line skips windows with no steps.

APIServer, from vLLM:

```
Engine 000: Avg prompt throughput: 378.4 tokens/s, Avg generation throughput: 38.3 tokens/s, Running: 4 reqs, Waiting: 0 reqs, CPU KV cache usage: 3.3%, Prefix cache hit rate: 10.5%
```

EngineCore, from the plugin runner:

```
[OV-RUNNER] 10.0s: 217 steps, 86.8 scheduled tokens/s, step avg 45.6 ms (prep 0.2, infer 42.2, other forward 0.7, sample 2.5), max 77.2 ms
```

| Field | Meaning |
|---|---|
| `steps`, `scheduled tokens/s` | Engine steps and scheduled tokens (prefill + decode + drafts) in the window |
| `prep` | Input preparation in the runner |
| `infer` | The OpenVINO `infer` call |
| `other forward` | The rest of the forward: logits handling and the vision encoder |
| `sample` | Grammar bitmask, sampling, draft verification and proposal |
| `max` | Longest step in the window; a long prefill step shows up here |

`/metrics` has no infer/sample split; this line is where to look when the step
time, not the scheduler, is the question.
