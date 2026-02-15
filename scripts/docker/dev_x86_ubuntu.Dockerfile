# DashInfer x86 CPU-only development environment
# Ubuntu 24.04 + Python 3.10 + Conan 2.x
#
# Build:
#   docker build -f scripts/docker/dev_x86_ubuntu.Dockerfile -t dashinfer/dev-x86-ubuntu:latest .
# Run:
#   docker run -it dashinfer/dev-x86-ubuntu:latest

FROM ubuntu:24.04

ARG PY_VER=3.10

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# System packages + Python 3.10 from deadsnakes PPA
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    git-lfs \
    vim \
    wget \
    curl \
    unzip \
    ca-certificates \
    libssl-dev \
    libcurl4-openssl-dev \
    python${PY_VER} \
    python${PY_VER}-dev \
    python${PY_VER}-venv \
    patchelf \
    numactl \
    rpm \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Python virtual environment with Python 3.10
RUN python${PY_VER} -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Python build & development tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        "conan>=2.0,<3" \
        pybind11-global \
        pytest \
        "protobuf>=3.18,<4" \
        "transformers>=4.40,<5" \
        tokenizers \
        accelerate \
        scons \
        pandas \
        tabulate

# PyTorch (CPU)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Initialize Conan 2.x default profile
RUN conan profile detect --force

WORKDIR /root/
