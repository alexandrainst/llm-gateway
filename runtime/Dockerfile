ARG BASE_IMAGE=vllm/vllm-openai:v0.10.1.1
FROM ${BASE_IMAGE}

ARG NCCL_VERSION=""
ENV PIP_NO_CACHE_DIR=1

RUN set -eux; \
    python3 -m pip install --upgrade pip; \
    if [ -n "$NCCL_VERSION" ]; then \
      python3 -m pip install -U "nvidia-nccl-cu12==${NCCL_VERSION}" "transformers>=4.56"; \
    else \
      python3 -m pip install -U nvidia-nccl-cu12 "transformers>=4.56"; \
    fi

