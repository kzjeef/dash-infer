# DashInfer CUDA development environment
# Ubuntu 24.04 + CUDA 12.6 + Python 3.10 + Conan 2.x
#
# Build:
#   docker build -f scripts/docker/dev_ubuntu_cuda.Dockerfile -t dashinfer/dev-ubuntu-cuda:latest .
# Run:
#   docker run --gpus all -it dashinfer/dev-ubuntu-cuda:latest

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

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
    autoconf \
    automake \
    libtool \
    openssh-client \
    zsh \
    && rm -rf /var/lib/apt/lists/*

# Python virtual environment with Python 3.10
RUN python${PY_VER} -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Python build & development tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        "conan>=2.0,<3" \
        pybind11-global \
        auditwheel==6.1.0 \
        pytest \
        "protobuf>=3.18,<4" \
        "transformers>=4.40,<5" \
        tokenizers \
        accelerate \
        scons \
        pandas \
        tabulate \
        pyopenssl \
        jsonlines \
        GitPython \
        editdistance \
        sacrebleu \
        nltk \
        rouge-score

# PyTorch (CUDA 12.6)
RUN pip install --no-cache-dir torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

# Initialize Conan 2.x default profile
RUN conan profile detect --force

# Timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

WORKDIR /root/
