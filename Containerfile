FROM registry.access.redhat.com/ubi10/ubi:latest
RUN dnf install -y git python3 python3-devel gcc gcc-c++ make cmake && \
    dnf clean all
RUN pip install -U pip && pip install -U "transformers<4.58" setuptools wheel packaging
WORKDIR /opt/app-root
COPY pyproject.toml .
RUN VLLM_TARGET_DEVICE="openvino" PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu" pip install --no-build-isolation --ignore-installed . && \
    pip uninstall -y triton && \
    pip cache purge
COPY vllm_openvino ./vllm_openvino
RUN mkdir ./src && chgrp -R 0 . && chmod -R g+rwX .
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
