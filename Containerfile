FROM registry.access.redhat.com/ubi10/ubi:latest AS builder
RUN dnf install -y python3 && \
    dnf clean all
ENV VIRTUAL_ENV=/opt/vllm-env
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip setuptools wheel && \
    PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" \
    pip install --no-cache-dir "torch==2.10.0+cpu" "torchvision==0.25.0+cpu" "openvino==2026.1.0" && \
    pip install --no-cache-dir "vllm==0.18.1" && \
    pip uninstall -y triton && \
    pip cache purge
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
ENV VLLM_CACHE_ROOT=/tmp/vllm HOME=/tmp HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_COMPILE_DISABLE=1 VLLM_OPENVINO_DEVICE=CPU VLLM_OPENVINO_KVCACHE_SPACE=8
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
