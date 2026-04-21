# Build from source takes 30+ min. Use pre-built quay.io image as base for faster builds.
# When vLLM wheels become available for your Python version, switch back to build-from-source approach.
FROM quay.io/joopark/vllm-openvino:0.14.1 AS base
WORKDIR /opt/app-root
COPY vllm_openvino ./vllm_openvino
RUN mkdir -p /tmp/hf_home && chgrp -R 0 . && chmod -R g+rwX .
ENV VLLM_CACHE_ROOT=/tmp/vllm HOME=/tmp HF_HOME=/tmp/hf_home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCH_COMPILE_DISABLE=1 VLLM_OPENVINO_DEVICE=CPU VLLM_OPENVINO_KVCACHE_SPACE=8
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
