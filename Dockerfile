FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/workspace/.cache/huggingface/hub \
    TORCH_EXTENSIONS_DIR=/workspace/.cache/torch_extensions \
    PATH=/opt/conda/bin:$PATH

WORKDIR /workspace/DeepSpec

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    vim \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge, which uses conda-forge by default
RUN wget -q \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# Create isolated Python 3.11 environment
RUN conda create -y -n deepspec python=3.11 \
    && conda clean -afy

ENV PATH=/opt/conda/envs/deepspec/bin:/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV=deepspec

RUN git lfs install

COPY requirements.txt .

RUN python --version \
    && python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

RUN mkdir -p \
    /workspace/.cache/huggingface \
    /workspace/.cache/torch_extensions \
    /workspace/data \
    /workspace/checkpoints \
    /workspace/outputs

COPY . .

CMD ["/bin/bash"]