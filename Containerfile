FROM registry.access.redhat.com/ubi10/ubi:latest AS builder
RUN dnf install -y python3 && \
    dnf clean all
ENV VIRTUAL_ENV=/opt/vllm-env
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip setuptools wheel && \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" \
    pip install --no-cache-dir "torch==2.11.0+cpu" "torchvision==0.26.0+cpu" "openvino==2026.2.1" "transformers==5.5.3" "vllm==0.24.0" && \
    pip uninstall -y \
        triton \
        flashinfer-cubin flashinfer-python \
        cuda-bindings cuda-core cuda-pathfinder cuda-python cuda-tile cuda-toolkit \
        nvidia-cuda-cccl nvidia-cuda-crt nvidia-cuda-nvcc nvidia-cuda-nvrtc nvidia-cuda-runtime \
        nvidia-cuda-tileiras nvidia-cudnn-frontend nvidia-cutlass-dsl \
        nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu13 \
        nvidia-ml-py nvidia-nvjitlink nvidia-nvvm \
        tokenspeed-mla tokenspeed-triton tilelang && \
    pip cache purge && \
    find /opt/vllm-env -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
WORKDIR /opt/vllm
COPY pyproject.toml ./
RUN pip install --no-deps --no-cache-dir .
FROM registry.access.redhat.com/ubi10/ubi-minimal:latest
RUN microdnf install -y python3 shadow-utils && microdnf clean all
COPY --from=builder /opt/vllm-env /opt/vllm-env
ENV VIRTUAL_ENV=/opt/vllm-env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /opt/app-root
COPY vllm_openvino ./vllm_openvino
RUN mkdir /tmp/hf_home && chgrp -R 0 . && chmod -R g+rwX .
ENV PYTHONPATH=/opt/app-root VLLM_CACHE_ROOT=/tmp/vllm HOME=/tmp HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_COMPILE_DISABLE=1 VLLM_OPENVINO_DEVICE=CPU
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
