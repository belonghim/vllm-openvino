# OpenVINO CPU plugin patch: per-expert GEMM for MoE prefill

The image replaces the pip wheel's `openvino/libs/libopenvino_intel_cpu_plugin.so`
with `libopenvino_intel_cpu_plugin.so` from this directory (see `Containerfile`).

- Problem: `GatherMatmulDnnlExecutor::execute` uses GEMM for M > 1 only on
  bf16+AMX, so MoE prefill elsewhere runs one GEMV per routed row.
  Upstream: openvinotoolkit/openvino#38517; PR openvinotoolkit/openvino#38822 (open).
- Patch: `openvino-gathermatmul-grouped-gemm.patch` (2 files, +67/-10). Without
  AMX each expert with at least 4 routed rows runs one GEMM over those rows; experts
  with fewer stay on GEMV (2-token forwards were ~10% slower otherwise). The AMX
  branch is untouched.
- Binary: CPU plugin only, built from tag `2026.4.1` (e213a147) with the patch
  applied. A stock build with the same procedure is not byte-identical to the wheel's
  plugin, but gives identical NLL on real text.
  The `.so` is gitignored, so copy it here before `podman build`.

| File | sha256 |
|---|---|
| patched (`libopenvino_intel_cpu_plugin.so`) | `bd4b1036c173a900944b5f6e0dd99b656c194a49f6dbdada8f2b381bc6361c38` |
| stock 2026.4.1 wheel | `6599e3dd300d047b5ed72f8e48101c5f312c25321f35f57f803641ff8372900b` |

Both binaries, the OpenVINO source tree, and the build scripts are kept in `~/build`
(`cpu_plugin_final.so`, `cpu_plugin_stock_rebuilt.so`, `README.md`, `scripts/`).

Measured on Zen 4, 8 CPUs, stock -> patched:
- LFM2.5-8B-A1B-int4: prefill 106 -> 425 tok/s at 1024 tokens; through vLLM,
  TTFT for a 4000-token prompt 36.7 s -> 7.9 s.
- Qwen3.6-35B-A3B-int4: prefill 74 -> 204 tok/s at 1024 tokens.
- Decode unchanged. NLL on 1024 tokens of real text: wheel 4.2144, patched 4.2192.

## Upstream status

- Reviewer (maxnick) suggested relaxing the AMX-only condition to AVX512_BF16
  instead of changing the GEMV path. Measured on master 6a8184f (Zen 4, 8 CPUs),
  prefill at 1024 tokens, stock / relaxed condition only / relaxed + #38830 / this patch:
  - LFM2.5-8B-A1B: 108 / 114 / 442 / 423 tok/s
  - Qwen3.6-35B-A3B: 67 / 35 / 188 / 188 tok/s
- So the relaxed condition alone does not help, because the AMX branch runs every expert
  over all tokens. With #38830 it matches this patch, but without a GEMV fallback for
  experts with fewer than 4 rows, chunks of 2-8 tokens get 20-40% slower.
- Serving needs that fallback. vLLM flattens all scheduled tokens, so a decode step with
  N concurrent requests is M = N. At N <= 16, nearly every active expert gets fewer than
  4 rows; at N <= 64 for Qwen3.6, whose 256 experts spread the rows thinner.
- openvinotoolkit/openvino#38830 sizes the GEMM per expert in the AMX bf16 branch only.
  It is open, with maxnick's change request awaiting re-review.
- If #38830 merges alone, non-AMX CPUs (all AMD, Intel client, pre-Sapphire Rapids
  Xeon) and f32 sources stay on per-row GEMV, so this patch is still needed there.
- Plan for #38822: after #38830 merges, a follow-up that drops the AMX-only
  condition and keeps the GEMV fallback for few-row experts.

The binary is tied to OpenVINO 2026.4.1. When upgrading OpenVINO:
- If the release takes the per-expert GEMM on non-AMX CPUs (#38822 or the follow-up
  above), or #37444 replaces it, delete the `COPY` line from `Containerfile` and this
  directory. #38830 alone is not enough.
- Otherwise rebuild the plugin from the new tag with the patch, replace the `.so`
  and the sha256 above, and re-measure. If #38830 is in the tag, the patch conflicts
  in `execute()`: drop the AMX-only condition instead and add the 4-row GEMV fallback
  (branch `n-all830thr` in `~/build/ov-master`).
