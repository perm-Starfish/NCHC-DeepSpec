FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Miniforge
ENV CONDA_DIR=/opt/conda
ENV PATH=${CONDA_DIR}/bin:${PATH}

WORKDIR /workspace/DeepSpec

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    git-lfs \
    ninja-build \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

ARG LOCAL_UID=1000
ARG LOCAL_GID=1000

RUN groupadd -g ${LOCAL_GID} deepspec \
    && useradd -m -u ${LOCAL_UID} -g ${LOCAL_GID} -s /bin/bash deepspec

# Install Miniforge, which uses conda-forge by default
RUN curl -L \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

# Create isolated Python 3.11 environment
RUN conda create -y -n deepspec python=3.11 \
    && conda clean -afy

ENV PATH=${CONDA_DIR}/envs/deepspec/bin:${CONDA_DIR}/bin:${PATH}
ENV CONDA_DEFAULT_ENV=deepspec

# copy requirements for Docker build cache
COPY requirements.txt /tmp/requirements.txt

# install PyTorch with CUDA 12.6 support
RUN python -m pip install --no-cache-dir \
    torch==2.11.0 \
    torchvision==0.26.0 \
    torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu126

# install other dependencies
RUN python -m pip install --no-cache-dir \
    -r /tmp/requirements.txt

#RUN python -m pip install --upgrade pip setuptools wheel \
    #&& python -m pip install --no-cache-dir -r /tmp/requirements.txt

# copy project files; compose will override with host versions
COPY . /workspace/DeepSpec

RUN git config --system --add safe.directory /workspace/DeepSpec

CMD ["/bin/bash"]