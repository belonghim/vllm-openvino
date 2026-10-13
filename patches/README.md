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

The binary is tied to OpenVINO 2026.4.1. When upgrading OpenVINO:
- If the release contains the fix (or #37444 replaces it), delete the `COPY`
  line from `Containerfile` and this directory.
- Otherwise rebuild the plugin from the new tag with the patch, replace the `.so`
  and the sha256 above, and re-measure.
