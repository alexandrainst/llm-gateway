ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

ARG NCCL_VERSION=""
ENV PIP_NO_CACHE_DIR=1

RUN set -eux; \
    python3 -m pip install --upgrade pip; \
    if [ -n "$NCCL_VERSION" ]; then \
      python3 -m pip install -U "nvidia-nccl-cu12==${NCCL_VERSION}"; \
    else \
      python3 -m pip install -U nvidia-nccl-cu12; \
    fi

