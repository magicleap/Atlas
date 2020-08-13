FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04 

RUN apt-get update && apt-get install -y \
    cmake \
    wget \
    curl \
    git \
    rsync \
    sudo \
    zip \ 
    vim \
    ssh \
    unzip \
    screen \
    openssh-server \
    pkg-config \
    build-essential \
    libboost-all-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    freeglut3-dev \
    qt5-default \
    libxmu-dev \
    libxi-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libcgal-dev \ 
    libeigen3-dev \
  && rm -rf /var/lib/apt/lists/*

# Install Ceres Solver
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
RUN mkdir -p ceres-solver/build
WORKDIR ceres-solver/build
RUN cmake -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF .. && make -j$(nproc) && make install && make clean 
WORKDIR /

# Install Colmap
RUN git clone https://github.com/colmap/colmap
RUN mkdir -p colmap/build
WORKDIR colmap/build
RUN cmake -DCMAKE_BUILD_TYPE=Release -DTESTS_ENABLED=OFF .. && make -j$(nproc) && make install && make clean
WORKDIR /


# Install Miniconda
RUN curl -so /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH


# Create a Python 3.6 environment
RUN conda install -y conda-build \
 && conda create -y --name py36 python=3.6.7 \
 && conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install dependencies
RUN conda install -y pytorch=1.5.0 torchvision=0.6.0 cudatoolkit=10.2 -c pytorch
RUN conda install opencv

RUN pip install \
  open3d>=0.10.0.0 \
  trimesh>=3.7.6 \
  pyquaternion>=0.9.5 \
  pytorch-lightning>=0.8.5 \
  pyrender>=0.1.43 \
  scikit-image
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.5/index.html

# For 16bit mixed precision
RUN git clone https://github.com/NVIDIA/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

# add headless support for pyrender
# https://pyrender.readthedocs.io/en/latest/install/index.html
# install OSMesa
RUN apt update -y \
  && wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb \
  && dpkg -i ./mesa_18.3.3-0.deb || true \
  && apt install -f -y
# install a Compatible Fork of PyOpenGL
RUN git clone https://github.com/mmatl/pyopengl.git \
  && pip install ./pyopengl


RUN echo "export PATH=$CONDA_PREFIX/bin:$PATH" >> /etc/profile
RUN echo "export PYTHONPATH=/miniconda/envs/lib/python3.6/site-packages:$PYTHONPATH" >> /etc/profile
RUN echo "export NCCL_LL_THRESHOLD=0" >> /etc/profile
RUN echo "umask 002" >> /etc/profile

ENV NCCL_LL_THRESHOLD=0

