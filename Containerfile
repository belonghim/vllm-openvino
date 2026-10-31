FROM registry.access.redhat.com/ubi10/ubi:latest AS builder
RUN dnf install -y python3 && \
    dnf clean all
ENV VIRTUAL_ENV=/opt/vllm-env
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip setuptools wheel && \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly" \
    pip install --no-cache-dir "torch==2.13.0+cpu" "torchvision==0.28.0+cpu" "openvino==2026.5.0.dev20261030" "transformers==5.18.0" "vllm @ https://wheels.vllm.ai/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl" && \
    pip uninstall -y \
        triton \
        flashinfer-cubin flashinfer-python \
        cuda-bindings cuda-core cuda-pathfinder cuda-python cuda-tile cuda-toolkit \
        nvidia-cuda-cccl nvidia-cuda-crt nvidia-cuda-nvcc nvidia-cuda-nvrtc nvidia-cuda-runtime \
        nvidia-cuda-tileiras nvidia-cuda-nvdisasm nvidia-cudnn-frontend nvidia-cutlass-dsl \
        nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 \
        nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 \
        nvidia-ml-py nvidia-nvjitlink nvidia-nvvm \
        humming-kernels pynvvideocodec quack-kernels \
        tokenspeed-mla tokenspeed-triton tilelang && \
    pip cache purge && \
    SP=/opt/vllm-env/lib/python3.12/site-packages && \
    rm -rf $SP/vllm/vllm_flash_attn $SP/vllm/third_party/deep_gemm $SP/vllm/_flashkda_C.abi3.so $SP/nccl && \
    find /opt/vllm-env -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
WORKDIR /opt/vllm
COPY pyproject.toml ./
RUN pip install --no-deps --no-cache-dir .
FROM registry.access.redhat.com/ubi10/ubi-minimal:latest
RUN microdnf install -y python3 && microdnf clean all
COPY --from=builder /opt/vllm-env /opt/vllm-env
ENV VIRTUAL_ENV=/opt/vllm-env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /opt/app-root
COPY vllm_openvino ./vllm_openvino
RUN mkdir /tmp/hf_home && chgrp -R 0 . && chmod -R g+rwX .
ENV PYTHONPATH=/opt/app-root VLLM_CACHE_ROOT=/tmp/vllm HOME=/tmp HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_COMPILE_DISABLE=1
ENTRYPOINT ["vllm", "serve"]
