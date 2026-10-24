#!/bin/bash
# Rebuild patches/libopenvino_intel_cpu_plugin.so from the OpenVINO tag with the patch applied.
# Usage: patches/build-cpu-plugin.sh   (env: WORK, JOBS)
set -euo pipefail

OV_TAG=2026.4.1
# The build number is embedded in the binary; a shallow clone would change it.
CI_BUILD_NUMBER=2026.4.1-22982-e213a147257
EXPECTED_SHA256=bd4b1036c173a900944b5f6e0dd99b656c194a49f6dbdada8f2b381bc6361c38
WORK=${WORK:-$HOME/.cache/vllm-openvino/cpu-plugin}
JOBS=${JOBS:-8}
here=$(cd "$(dirname "$0")" && pwd)
patch_file="$here/openvino-gathermatmul-grouped-gemm.patch"
out="$here/libopenvino_intel_cpu_plugin.so"

mkdir -p "$WORK/ctx" "$WORK/build"
if [ ! -d "$WORK/src" ]; then
    git clone --depth 1 --branch "$OV_TAG" --recurse-submodules --shallow-submodules \
        https://github.com/openvinotoolkit/openvino.git "$WORK/src"
fi
git -C "$WORK/src" checkout -- .
git -C "$WORK/src" apply "$patch_file"
# The reference build had no readable oneDNN git metadata, so its binary lacks that commit hash.
rm -f "$WORK/src/src/plugins/intel_cpu/thirdparty/onednn/.git"

podman build -q -t vllm-openvino-ov-builder -f - "$WORK/ctx" <<'EOF'
FROM registry.access.redhat.com/ubi10/ubi
RUN dnf install -y gcc-c++ cmake make git python3 patch which findutils && dnf clean all
EOF

# Same paths and options as the build that produced EXPECTED_SHA256; both are embedded in the binary.
podman run --rm --cpus="$JOBS" -e GIT_CEILING_DIRECTORIES=/src/src -v "$WORK/src:/src:Z" -v "$WORK/build:/build:Z" \
    vllm-openvino-ov-builder bash -c "
set -e
CI_BUILD_NUMBER=$CI_BUILD_NUMBER cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_INTEL_GPU=OFF -DENABLE_INTEL_NPU=OFF -DENABLE_PYTHON=OFF -DENABLE_TESTS=OFF -DENABLE_FUNCTIONAL_TESTS=OFF \
  -DENABLE_SAMPLES=OFF -DENABLE_OV_ONNX_FRONTEND=OFF -DENABLE_OV_TF_FRONTEND=OFF -DENABLE_OV_TF_LITE_FRONTEND=OFF \
  -DENABLE_OV_PADDLE_FRONTEND=OFF -DENABLE_OV_PYTORCH_FRONTEND=OFF -DENABLE_OV_JAX_FRONTEND=OFF -DENABLE_OV_GGUF_FRONTEND=OFF \
  -DENABLE_AUTO=OFF -DENABLE_HETERO=OFF -DENABLE_AUTO_BATCH=OFF -DENABLE_TEMPLATE=ON -DENABLE_PROXY=OFF \
  -DENABLE_CPPLINT=OFF -DENABLE_CLANG_FORMAT=OFF -DENABLE_NCC_STYLE=OFF -DTHREADING=TBB -DENABLE_SYSTEM_TBB=OFF \
  > /build/configure.log
cmake --build /build --target openvino_intel_cpu_plugin -j $JOBS"

built="$WORK/src/bin/intel64/Release/libopenvino_intel_cpu_plugin.so"
actual=$(sha256sum "$built" | cut -d' ' -f1)
if [ "$actual" != "$EXPECTED_SHA256" ]; then
    echo "$built: sha256 $actual differs from $EXPECTED_SHA256 (toolchain or source changed)." >&2
    echo "Re-measure as in patches/README.md before copying it here and updating the hash there and in Containerfile." >&2
    exit 1
fi
cp "$built" "$out"
echo "Built $out ($actual)"
