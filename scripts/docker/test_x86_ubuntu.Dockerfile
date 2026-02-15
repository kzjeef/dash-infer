# DashInfer x86 CPU-only test environment
# Ubuntu 24.04 + Python 3.10
#
# Build:
#   docker build -f scripts/docker/test_x86_ubuntu.Dockerfile -t dashinfer/test-x86-ubuntu:latest .
# Run:
#   docker run -it dashinfer/test-x86-ubuntu:latest

FROM ubuntu:24.04

ARG PY_VER=3.10

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System packages + Python 3.10 from deadsnakes PPA
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update -y && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    python${PY_VER} \
    python${PY_VER}-dev \
    python${PY_VER}-venv \
    numactl \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Python virtual environment with Python 3.10
RUN python${PY_VER} -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Minimal test dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        pytest \
        "transformers>=4.40,<5" \
        tokenizers \
        accelerate \
        "protobuf>=3.18,<4"

WORKDIR /root/
