Bootstrap: docker
From: nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

%help
To start your container simply try
singularity exec THIS_CONTAINER.simg bash

To use GPUs, try
singularity exec --nv THIS_CONTAINER.simg bash

Container based on recipe by drinkingkazu

%labels
Maintainer gnperdue
Version ubuntu16.04-py3-tf110-gpu

#------------
# Global installation
#------------
%environment
    export XDG_RUNTIME_DIR=/tmp/$USER
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}
    export CUDA_DEVICE_ORDER=PCI_BUS_ID

%post
    # add wilson cluster mount points
    mkdir /scratch /data /lfstev

    # apt-get
    apt-get -y update
    apt-get -y install dpkg-dev g++ gcc binutils libqt4-dev python3-dev python3-tk python3-pip openslide-tools git autoconf libtool build-essential 

    # pip
    python3 -m pip install --upgrade setuptools pip
    python3 -m pip install "https://github.com/lakshayg/tensorflow-build/releases/download/tf1.9.0-ubuntu16.04-py27-py35/tensorflow-1.9.0-cp35-cp35m-linux_x86_64.whl"
    python3 -m pip install Keras==2.0.0
    python3 -m pip install numpy
    python3 -m pip install pandas
    python3 -m pip install matplotlib
    python3 -m pip install ipython
    python3 -m pip install scikit-learn
    python3 -m pip install scikit-image
    python3 -m pip install scipy
    python3 -m pip install Pillow
    python3 -m pip install jedi
    python3 -m pip install pathlib
    python3 -m pip install notebook
    python3 -m pip install jupyter
    python3 -m pip install h5py
    python3 -m pip install opencv-python
    python3 -m pip install tqdm
    python3 -m pip install openslide-python
    python3 -m pip install seaborn
