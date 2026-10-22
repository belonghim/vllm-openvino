# OpenVINO CPU plugin patch: per-expert GEMM for MoE prefill

The image replaces the pip wheel's `openvino/libs/libopenvino_intel_cpu_plugin.so`
with `libopenvino_intel_cpu_plugin.so` from this directory (see `Containerfile`).

- Problem: `GatherMatmulDnnlExecutor::execute` uses GEMM for M > 1 only on
  bf16+AMX, so MoE prefill elsewhere runs one GEMV per routed row.
  Upstream: openvinotoolkit/openvino#38517; PR openvinotoolkit/openvino#39086 (open).
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
- openvinotoolkit/openvino#38830 (GEMM sized per expert, AMX bf16 branch only) merged
  into master on 2026-10-21 (1ac0ccb7, milestone 2026.5). It is not in 2026.4.1.
- Non-AMX CPUs (all AMD, Intel client, pre-Sapphire Rapids Xeon) and f32 sources still
  run per-row GEMV on master, so this patch is still needed there.
- openvinotoolkit/openvino#39086 (open, replaces #38822, now closed) takes the
  per-expert GEMM branch for every f32 and bf16 source and keeps GEMV for experts with
  fewer than 4 routed rows (+28/-11, same two files plus a test comment). On master
  31489b9c: `ov_cpu_func_tests` MoE/GroupedMatMul filter 329 passed, 240 with
  `OV_CPU_MAX_ISA=AVX2`. Zen 4, 8 CPUs, LFM2.5-8B-A1B: prefill at 1024 tokens 421 tok/s
  (bf16), 165 tok/s (AVX2 f32), 286 tok/s (AVX-512 f32, stock 110); 2-token chunk 54 ms
  (stock 49). The 4-row threshold is untested on AMX; the #38830 author was asked to try
  it on Xeon 6972P.
- maxnick is running performance validation on more hardware. On AMX the PR replaces
  #38830's 16-row GEMM with GEMV for experts with fewer than 4 rows, which may be slower
  there, so I offered to keep the fallback off on bf16 AMX. That variant is ready
  (`~/build/gathermatmul-followup-amx.patch`, branch `follow-up-gemm-all-amx`, same
  test results); push it only if maxnick agrees.

The binary is tied to OpenVINO 2026.4.1. When upgrading OpenVINO:
- If the release contains #39086, or #37444 replaces the path, delete the `COPY` line
  from `Containerfile` and this directory. #38830 alone is not enough.
- Otherwise rebuild the plugin from the new tag with the patch, replace the `.so`
  and the sha256 above, and re-measure. 2026.5 contains #38830, so the patch conflicts
  in `execute()`: apply #39086 instead (`~/build/gathermatmul-followup.patch`, branch
  `follow-up-gemm-all` in `~/build/ov-master`).
