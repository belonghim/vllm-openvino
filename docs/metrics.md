# vLLM Metrics on GPU (KServe)

How to collect and read vLLM's Prometheus metrics for a model served by stock
vLLM on an NVIDIA GPU under KServe (OpenShift AI). Collection stays inside the
cluster, so it works the same on a closed network.

Checked on a reference deployment:

- Red Hat AI Inference Server image, vLLM `0.24.0+rhaiv.9`, `vllm-cuda-runtime`
- Qwen3.5-0.8B (`Qwen3_5ForConditionalGeneration`, hybrid attention/Mamba),
  one GeForce RTX 3060 12 GB
- `--max-model-len=8192 --max-num-seqs=32 --max-num-batched-tokens=1280
  --gpu-memory-utilization=0.93`

The metric set changes between vLLM releases (0.32 adds
`admission_rejections_total`, `prompt_tokens_cached_by_source_total` and
`request_num_preemptions`). Check the `# HELP` lines of the target server
before building dashboards on a name.

## Collection Path

| Piece | Reference deployment |
|---|---|
| Endpoint | vLLM API server, container port 8080, `/metrics` (no auth on the pod port) |
| Service | `<isvc>-metrics`, port `vllm-cuda-runtime-metrics` (8080) |
| ServiceMonitor | `<isvc>-metrics`, path `/metrics`, http |
| Prometheus | User workload monitoring (`enableUserWorkload: true`), scrape interval 30 s; query through `thanos-querier` |
| Series labels | `namespace`, `pod`, `job`/`service` (`<isvc>-metrics`), `model_name` (served model name, the isvc name), `engine` |
| GPU metrics | NVIDIA GPU Operator's DCGM exporter; the serving pod is in `exported_namespace` / `exported_pod` |

`up{namespace="<ns>", job="<isvc>-metrics"}` is 1 when scraping works. To
read the raw endpoint:

```bash
oc exec -n <ns> <predictor-pod> -c kserve-container -- curl -s http://127.0.0.1:8080/metrics
```

vLLM does not export GPU utilization or GPU memory. Use DCGM:
`DCGM_FI_DEV_GPU_UTIL`, `DCGM_FI_DEV_FB_USED` and `DCGM_FI_DEV_FB_FREE` (MiB).
The `process_*` metrics in `/metrics` cover only the API server process.

## Metrics (vLLM 0.24)

| Metrics | Notes |
|---|---|
| `num_requests_running`, `num_requests_waiting` | Gauges, updated every engine step |
| `num_requests_waiting_by_reason{reason}` | Stayed 0 while `num_requests_waiting` peaked at 7; use `num_requests_waiting` |
| `kv_cache_usage_perc` | Fraction of KV blocks in use (0-1) |
| `prefix_cache_queries_total`, `prefix_cache_hits_total`, `prompt_tokens_cached_total`, `prompt_tokens_by_source_total` | 0 when prefix caching is off |
| `external_prefix_cache_*` | KV connector only |
| `mm_cache_queries_total`, `mm_cache_hits_total` | Multimodal processor cache, in items |
| `num_preemptions_total` | See Preemption below |
| `prompt_tokens_total`, `generation_tokens_total` | Throughput counters |
| `iteration_tokens_total` | Histogram of tokens per engine step |
| `request_prompt_tokens`, `request_generation_tokens`, `request_max_num_generation_tokens`, `request_params_n`, `request_params_max_tokens` | Per-request histograms |
| `time_to_first_token_seconds`, `inter_token_latency_seconds`, `request_time_per_output_token_seconds`, `e2e_request_latency_seconds` | Latency histograms |
| `request_queue_time_seconds`, `request_inference_time_seconds`, `request_prefill_time_seconds`, `request_decode_time_seconds` | Per-phase histograms |
| `request_prefill_kv_computed_tokens` | Prefill tokens computed (cached tokens excluded) |
| `request_success_total{finished_reason}` | `stop`, `length`, `abort`, `error`, `repetition`; requests rejected with HTTP 400 are not counted |
| `estimated_flops_per_gpu_total`, `estimated_read_bytes_per_gpu_total`, `estimated_write_bytes_per_gpu_total` | 0 unless `--enable-mfu-metrics` |
| `spec_decode_*` | Present only with `--speculative-config` |
| `engine_sleep_state{sleep_state}` | `awake` 1 unless sleep mode is used |
| `cache_config_info` | Config as labels, value 1 |

## Reading Them

**Gauges against a 30 s scrape.** On GPU an engine step takes milliseconds,
so the gauges are current: a burst of 4 requests showed all 4 running at the
first 0.5 s read of `/metrics`. Prometheus, however, keeps one sample per
30 s, so bursts shorter than that can fall between samples. Alert and graph
on `max_over_time(...[5m])` rather than the instant value.

**Waiting with free KV cache.** In a 16-client soak (below)
`num_requests_waiting` peaked at 7 and `request_queue_time_seconds` p95 was
0.29 s while `kv_cache_usage_perc` stayed under 9% and `max-num-seqs` (32)
was above the client count. The remaining limit is the per-step token budget
(`--max-num-batched-tokens=1280`): new prompts wait until a step has room for
their prefill. A non-zero waiting count does not by itself mean the KV cache
is full; check `kv_cache_usage_perc` before adding memory.

**`kv_cache_usage_perc` on a hybrid model.** vLLM puts attention KV and Mamba
state in pages of the same size. For Qwen3.5-0.8B it raised the block to 544
tokens (`block_size="544"`, Mamba page padded 2.64%), giving 1224 blocks and
`kv_cache_size_tokens="527737"`. A 2192-token request held 8 blocks: 5 for
attention KV and 3 more, which match one Mamba-state block for each of the
model's three groups of linear-attention layers (24 layers: 6 full, 18
linear). Short requests therefore use more of the pool than their token count
suggests, and `kv_cache_size_tokens` overstates the token capacity.

**Preemption.** A request needs at most `ceil(8192 / 544) + 3 = 19` blocks
here, so 32 sequences need at most 608 of the 1224 blocks: with these
arguments the KV cache cannot run out, and `num_preemptions_total` stays 0.
It becomes a useful alert after raising `--max-num-seqs` or
`--max-model-len`. Recompute this bound for the target model from
`cache_config_info`.

**GPU memory does not follow load.** vLLM reserves
`--gpu-memory-utilization` of the GPU at startup. `DCGM_FI_DEV_FB_USED` read
10,096-10,108 MiB (85% of the card) over 30 minutes covering idle, the burst
and the soak. Use `kv_cache_usage_perc` for how full the cache is, and FB
used/free only to catch other processes on the GPU or a restart that failed
to reserve memory.

**GPU utilization is not saturation.** `DCGM_FI_DEV_GPU_UTIL` reached 100
during the soak with the KV cache under 9%. It reports the share of time a
kernel was running, not how much work the GPU could still take; capacity
shows in throughput flattening while latency rises.

**Prefix caching.** vLLM 0.24 left prefix caching off for this hybrid model
(`enable_prefix_caching="False"`, `mamba_cache_mode="none"`), so the prefix
counters record neither queries nor hits. A repeated 1892-token prompt was
prefilled again (TTFT 0.13 s on the idle server, against 0.46 s inside the
4-request burst). Models with prefix caching on report hit rate as
`prefix_cache_hits_total / prefix_cache_queries_total`.

**Multimodal cache.** Sending the same image twice raised
`mm_cache_queries_total` by 1 each time and `mm_cache_hits_total` by 1 on the
second request, whose TTFT fell from 0.127 s to 0.029 s. The counters are in
items (images), not tokens.

**Counters in PromQL.** `increase()` extrapolates to the window edges, so it
returns fractional counts (411.5 and 2.1 in the table below); round it for
display.

## PromQL

Run against the reference Prometheus during the soak; `S` stands for
`namespace="<ns>",model_name="<isvc>"`. Values are from that run.

| Question | Query | Soak value |
|---|---|---|
| Output tokens/s | `sum(rate(vllm:generation_tokens_total{S}[2m]))` | 859 |
| Prompt tokens/s | `sum(rate(vllm:prompt_tokens_total{S}[2m]))` | 3808 |
| TTFT p95 | `histogram_quantile(0.95, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{S}[2m])))` | 0.24 s |
| Inter-token latency p95 | `histogram_quantile(0.95, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket{S}[2m])))` | 0.048 s |
| End-to-end p95 | `histogram_quantile(0.95, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{S}[2m])))` | 9.1 s |
| Queue time p95 | `histogram_quantile(0.95, sum by (le) (rate(vllm:request_queue_time_seconds_bucket{S}[5m])))` | 0.29 s |
| Peak running / waiting | `max_over_time(vllm:num_requests_running{S}[3m])`, same for `num_requests_waiting` | 16 / 7 |
| Peak KV usage | `max_over_time(vllm:kv_cache_usage_perc{S}[3m])` | 0.087 |
| Results by reason | `sum by (finished_reason) (increase(vllm:request_success_total{S}[3m]))` | `length` 411.5, `stop` 2.1 |
| GPU utilization | `max_over_time(DCGM_FI_DEV_GPU_UTIL{exported_namespace="<ns>",exported_pod=~"<isvc>-predictor-.*"}[3m])` | 100 |
| GPU memory fraction | `DCGM_FI_DEV_FB_USED{exported_pod=~"<isvc>-predictor-.*"} / (DCGM_FI_DEV_FB_USED{...} + DCGM_FI_DEV_FB_FREE{...})` | 0.85 |

## Reference Measurements

Burst: 4 concurrent 1892-token prompts with 300 output tokens each
(`ignore_eos`, greedy), then one of them again with 8 output tokens.

| | Value |
|---|---|
| Wall time | 2.6 s |
| TTFT mean, burst / repeated prompt | 0.46 s / 0.13 s |
| `inter_token_latency_seconds` mean | 7.2 ms |
| `request_queue_time_seconds` mean | 0.067 s |
| Peak `kv_cache_usage_perc` | 0.026 (32 blocks) |
| `iteration_tokens_total` count | 304 steps |

Soak: 16 clients for 90 s, prompts of 50-600 words, 64-512 output tokens
(`ignore_eos`); 387 requests completed. Results are in the PromQL table.
