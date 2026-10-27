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
  The `.so` is gitignored. `build-cpu-plugin.sh` clones the tag, applies the patch,
  builds the plugin in a UBI 10 container, and copies it here only if the sha256
  matches (a fresh clone reproduces it byte for byte).

| File | sha256 |
|---|---|
| patched (`libopenvino_intel_cpu_plugin.so`) | `bd4b1036c173a900944b5f6e0dd99b656c194a49f6dbdada8f2b381bc6361c38` |
| stock 2026.4.1 wheel | `6599e3dd300d047b5ed72f8e48101c5f312c25321f35f57f803641ff8372900b` |

The benchmark scripts and the upstream follow-up branches are kept in `~/build`.

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
- openvinotoolkit/openvino#39086 (open, replaces #38822, now closed): for M > 1 every
  source the executor accepts (f32, bf16) takes the per-expert GEMM branch. Without bf16
  AMX, experts with fewer than 4 routed rows stay on GEMV; on bf16 AMX every expert runs
  a GEMM, as after #38830 (commits c068d3a121, 6dd69ac13f, 09c802d526, 3026141a51;
  +84/-91, mostly re-indent; same two files plus a test comment).
  On master 31489b9c: `ov_cpu_func_tests` MoE/GroupedMatMul filter 329 passed, 240 with
  `OV_CPU_MAX_ISA=AVX2`. Zen 4, 8 CPUs, LFM2.5-8B-A1B: prefill at 1024 tokens 421 tok/s
  (bf16), 165 tok/s (AVX2 f32), 286 tok/s (AVX-512 f32, stock 110); 2-token chunk 54 ms
  (stock 49). The Jenkins functional tests on Sapphire Rapids (AMX) pass.
- maxnick requested changes (drop the flag, define the threshold where it is used, comment
  the GEMV branch) and a commit email linked to GitHub; all addressed on 2026-10-26.
  rkazants asked whether it fixes #38517. On the issue's reproducer (64 experts, top-8,
  f32, 8 threads), the fused GatherMatmul path went from 165.5 to 46.4 ms at 1024 tokens
  and is faster than the issue's plain-op `grouped` lowering from 64 tokens up. The
  `torch.cond` weight duplication in that issue is out of scope.
- maxnick then asked (2026-10-26) for an ISA-dependent threshold (all of M > 1 on GEMM
  for AMX) and single-thread gather/scatter below the L2 size; both pushed as 6dd69ac13f.
  Tests 329/329 and 240/240 (AVX2 cap). Zen 4, previous revision -> 6dd69ac13f, within
  run-to-run spread: LFM2.5-8B-A1B prefill 1024 tokens ~500 -> ~494 tok/s, L=2/4/8 chunks
  unchanged; Qwen3.6-35B-A3B 218 -> 210 tok/s, chunks 1-3% faster. maxnick's
  A clang-tidy fix and moving the row-count assert before the GEMV path followed as
  09c802d526.
- maxnick's Xeon benchmarks (2026-10-27) showed 2x slower prefill at 1K+ tokens, attributed
  to the single-thread gather/scatter (no difference on Zen 4). Rolled back in 3026141a51:
  gather/scatter use `parallel_for` again; the threshold change stays.
  `~/build/gathermatmul-followup.patch` holds all four commits (`git am`).
- Thread scaling, Zen 4 (12C/24T, host shared with VMs), LFM2.5-8B-A1B, stock -> #39086,
  prefill at 1024 tokens: bf16 107 -> 431 / 123 -> 465 / 123 -> 456 tok/s and f32
  109 -> 272 / 111 -> 308 / 127 -> 308 tok/s at 8 / 12 / 24 threads. Chunks of 2-4 tokens
  stay within run-to-run spread at every thread count (f32 L=4 about 6% slower at 8
  threads in three runs); L >= 8 is faster. No sign that more threads move the threshold.

The binary is tied to OpenVINO 2026.4.1. When upgrading OpenVINO:
- If the release contains #39086, or #37444 replaces the path, delete the `COPY` line
  from `Containerfile` and this directory. #38830 alone is not enough.
- Otherwise update `OV_TAG` and the patch in `build-cpu-plugin.sh`, rebuild, re-measure,
  and update the sha256 there, above, and in `Containerfile`. 2026.5 contains #38830, so the patch conflicts
  in `execute()`: apply #39086 instead (`~/build/gathermatmul-followup.patch`, branch
  `follow-up-gemm-all` in `~/build/ov-master`).
