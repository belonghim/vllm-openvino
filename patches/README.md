# OpenVINO CPU plugin patch: per-expert GEMM for MoE prefill

The image replaces the pip wheel's `openvino/libs/libopenvino_intel_cpu_plugin.so`
with `libopenvino_intel_cpu_plugin.so` from this directory (see `Containerfile`).

- Problem: `GatherMatmulDnnlExecutor::execute` uses GEMM for M > 1 only on
  bf16+AMX, so MoE prefill elsewhere runs one GEMV per routed row.
  Upstream: openvinotoolkit/openvino#38517; PR openvinotoolkit/openvino#38822 (open). The PR's final commit also keeps GEMV for experts with fewer than 4 routed rows (2-token forwards were ~10% slower otherwise); the patch below predates that.
- Patch: `openvino-gathermatmul-grouped-gemm.patch` (2 files, +64/-10). Without
  AMX each expert runs one GEMM over its own routed rows. The AMX branch is untouched.
- Binary: CPU plugin only, built from tag `2026.4.1` (e213a147) with the patch
  applied. A stock build with the same procedure is bit-identical to the wheel's plugin.
  The `.so` is gitignored, so copy it here before `podman build`.

| File | sha256 |
|---|---|
| patched (`libopenvino_intel_cpu_plugin.so`) | `297ab8e0a8f8ca98a2b0ad9b8c75d97f20ce6fb0d358d7853955e7c75890d9ea` |
| stock 2026.4.1 (= wheel) | `0f839e2f8f4813f41d5190dec781a4e5a5fabad8449c2a3a82af40369c8899cd` |

Both binaries, the OpenVINO source tree, and the build scripts are kept in `~/build`
(`cpu_plugin_patched.so`, `cpu_plugin_stock.so`, `README.md`, `scripts/`).

Measured on Zen 4, 8 CPUs, stock -> patched:
- LFM2.5-8B-A1B-int4: prefill 106 -> 425 tok/s at 1024 tokens; through vLLM,
  TTFT for a 4000-token prompt 36.7 s -> 7.9 s.
- Qwen3.6-35B-A3B-int4: prefill 74 -> 204 tok/s at 1024 tokens.
- Decode and NLL unchanged.

The binary is tied to OpenVINO 2026.4.1. When upgrading OpenVINO:
- If the release contains the fix (or #37444 replaces it), delete the `COPY`
  line from `Containerfile` and this directory.
- Otherwise rebuild the plugin from the new tag with the patch, replace the `.so`
  and the sha256 above, and re-measure.
