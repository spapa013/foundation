FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG PYTHON_VERSION=3.8
ARG ENV_NAME=foundation
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    unzip \
    rsync \
    openssh-server \
    libhdf5-dev \
    libsm6 \
    libgl1-mesa-glx \
    graphviz \
    fish \
    # FFmpeg libraries for PyAV
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libavfilter-dev \
    libavresample-dev \
    ffmpeg \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip" \
    && unzip /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
      -o /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# Install mamba from conda-forge only, bypassing Anaconda ToS
RUN conda config --add channels conda-forge && \
    conda config --set channel_priority strict && \
    conda install --override-channels -c conda-forge -y -n base mamba && \
    conda clean -ya

# Install scientific Python packages
RUN mamba create -y -n ${ENV_NAME} -c conda-forge \
      python=${PYTHON_VERSION} \
      libstdcxx-ng>=11.2.0 \
      jupyterlab=3.6.3 \
      numpy=1.23.5 \
      pandas=2.0.0 \
      h5py=3.8.0 \
      scipy=1.9.1 \
      scikit-learn=1.2.2 \
      scikit-image=0.20.0 \
      pynndescent=0.5.8 \
      numdifftools=0.9.41 \
      lmfit=1.2.0 \
      statsmodels=0.13.5 \
      networkx=3.1 \
      gitpython=3.1.31 \
      matplotlib=3.7.1 \
      seaborn=0.12.2 \
      plotly=5.14.1 \
      pillow=9.5.0 \
      umap-learn=0.5.3 \
      hdbscan=0.8.29 \
      tqdm=4.65.0 \
      av=10.0.0 \
      datajoint=0.12.9 && \
    conda clean --all --yes && \
    find /opt/conda -name '__pycache__' -type d -exec rm -rf {} + && \
    find /opt/conda -name '*.a' -delete && \
    find /opt/conda -name '*.pyc' -delete && \
    find /opt/conda -name '*.pyo' -delete && \
    strip --strip-unneeded /opt/conda/envs/${ENV_NAME}/lib/*.so || true
    
# Install PyTorch + CUDA
RUN conda run -n ${ENV_NAME} pip install --no-cache-dir \
      torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1 \
      --extra-index-url https://download.pytorch.org/whl/cu118

# Set the environment's Python and tools in PATH
ENV PATH=/opt/conda/envs/${ENV_NAME}/bin:$PATH
