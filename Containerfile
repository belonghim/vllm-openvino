FROM registry.access.redhat.com/ubi10/ubi:latest AS builder
RUN dnf install -y git python3 python3-devel gcc gcc-c++ make cmake && \
    dnf clean all
ENV VIRTUAL_ENV=/opt/vllm-env
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install -U pip && pip install -U "transformers<4.58" setuptools wheel packaging
WORKDIR /opt/app-root
COPY pyproject.toml .
RUN VLLM_TARGET_DEVICE="openvino" PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" pip install --no-build-isolation --ignore-installed . && \
    pip uninstall -y triton && \
    pip cache purge
FROM registry.access.redhat.com/ubi10/ubi-minimal:latest
RUN microdnf install -y python3 shadow-utils && microdnf clean all
COPY --from=builder /opt/vllm-env /opt/vllm-env
ENV VIRTUAL_ENV=/opt/vllm-env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /opt/app-root
COPY vllm_openvino ./vllm_openvino
RUN chgrp -R 0 . && chmod -R g+rwX .
ENV VLLM_CACHE_ROOT=/tmp/vllm HOME=/tmp
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
