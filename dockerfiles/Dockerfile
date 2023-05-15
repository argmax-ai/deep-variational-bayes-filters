FROM nvidia/cuda@sha256:7adef016a31559f149f0fdc5a102996eddbf69e1532947fb15a5a11413afe871

ENV TZ=Europe/Berlin
ENV DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# install apt packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential=12.8ubuntu* \
        bzip2=1.0.8-2 \
        ccache=3.7.7-1 \
        checkinstall=1.6.2+git20170426.d24a630-2ubuntu* \
        curl=7.68.0-1ubuntu* \
        gfortran=4:9.3.0-1ubuntu* \
        git=1:2.25.1-1ubuntu* \
        htop=2.2.0-2build1 \
        iputils-ping=3:20190709-3 \
        less=551-1ubuntu* \
        libblas-dev=3.9.0-1build1 \
        libbz2-dev=1.0.8-2 \
        libc6-dev=2.31-0ubuntu* \
        libffi-dev=3.3-4 \
        libfreetype6-dev=2.10.1-2ubuntu* \
        libgdbm-dev=1.18.1-5 \
        libhdf5-dev=1.10.4+repack-11ubuntu* \
        liblapack-dev=3.9.0-1build1 \
        libncursesw5-dev=6.2-0ubuntu* \
        libopenblas-dev=0.3.8+ds-1ubuntu* \
        libpng-dev=1.6.37-2 \
        libreadline-gplv2-dev=5.2+dfsg-3build3 \
        libsqlite3-dev=3.31.1-4ubuntu* \
        libssl-dev=1.1.1f-1ubuntu* \
        libzmq3-dev=4.3.2-2ubuntu* \
        make=4.2.1-1.2 \
        nano=4.8-1ubuntu* \
        openssl=1.1.1f-1ubuntu* \
        parallel=20161222-1.1 \
        pkg-config=0.29.1-0ubuntu* \
        rsync=3.1.3-8ubuntu* \
        software-properties-common \
        swig=4.0.1-5build1 \
        telnet=0.17-41.2build1 \
        tmux=3.0a-2ubuntu* \
        tree=1.8.0-1 \
        unzip=6.0-25ubuntu* \
        vim=2:8.1.2269-1ubuntu* \
        wget=1.20.3-1ubuntu* \
        zip=3.0-11build1 \
        zlib1g-dev=1:1.2.11.dfsg-2ubuntu* \
        zsh=5.8-3ubuntu* \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install conda
ENV CONDA_VERSION=py38_4.12.0-Linux-x86_64
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}.sh -o ~/miniconda.sh \
    && \
    /bin/bash ~/miniconda.sh -b -p /opt/miniconda && \
    rm -rf ~/miniconda.sh
ENV PATH=/opt/miniconda/bin:${PATH}

# install pytorch build requirements
RUN conda install -y \
        astunparse==1.6.3 \
        cmake==3.22.1 \
        cffi==1.15.1 \
        future==0.18.2 \
        dataclasses==0.8 \
        hypothesis==6.29.3 \
        mkl==2021.4.0 \
        mkl-include==2022.1.0 \
        ninja==1.10.2 \
        numpy==1.23.3 \
        pillow==9.2.0 \
        pyyaml==6.0 \
        setuptools==65.4.0 \
        six==1.16.0 \
        requests==2.28.1 \
        typing==3.10.0.0 \
        typing_extensions==4.3.0 \
    && \
    conda clean -ya

# install pytorch and related packages
ENV TORCHTEXT_VERSION=0.14.0
ENV WHEEL_URL=https://download.pytorch.org/whl
RUN conda install -y \
    torchvision==0.14.0 \
    torchaudio==0.13.0 \
    pytorch-cuda=11.7 -c pytorch -c nvidia \
    && \
    conda clean -ya
RUN pip install --no-cache-dir \
    ${WHEEL_URL}/torchtext-${TORCHTEXT_VERSION}-cp38-cp38-linux_x86_64.whl

# install general packages
RUN conda install -y \
        cython==0.29.32 \
        h5py==3.7.0 \
        notebook==6.5.4 \
        matplotlib==3.5.2 \
        pandas==1.4.4 \
        pybind11==2.9.2 \
        scikit-learn==1.1.2 \
        scipy==1.9.1 \
        virtualenv==20.16.2 \
    && \
    conda clean -ya

# install pytorch-lightning
RUN pip install --no-cache-dir \
    pytorch-lightning==1.7.7

# Pick up some dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    autoconf \
    autotools-dev \
    doxygen \
    ffmpeg \
    fonts-roboto \
    graphviz \
    libassimp-dev \
    libczmq-dev \
    libegl1 \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    libqt5opengl5-dev \
    libsdl2-dev \
    libspdlog-dev \
    libtool \
    libxcursor-dev \
    libxinerama-dev \
    libxi-dev \
    libxrandr-dev \
    mesa-common-dev \
    mesa-vulkan-drivers \
    openssh-server \
    pigz \
    qtbase5-dev \
    tar \
    xvfb \
    vulkan-tools \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY dockerfiles/robolab/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY dockerfiles/robolab/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Python packages
RUN pip install --no-cache-dir pip==22.3.1 setuptools==65.6.3
COPY dockerfiles/common/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
RUN pip install --no-cache-dir cffi

ENV MUJOCO_GL=egl
ENV PYOPENGL_PLATFORM=egl
ENV SHELL /bin/bash
ENV PIP_ROOT_USER_ACTION=ignore

WORKDIR /code/robolab
