FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND noninteractive
ENV TZ=Asia/Shanghai

##############################################################################
# Change apt source to Ksyun
##############################################################################
RUN sed -i "s#\S\+#http://apt.ksyun.cn/ubuntu/#2" /etc/apt/sources.list && \
    > /etc/apt/apt.conf.d/docker-clean && \
    > /etc/dpkg/dpkg.cfg.d/pkg-config-hook-config

##############################################################################
# Install basic utilities
##############################################################################
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common build-essential autotools-dev \
        nfs-common pdsh \
        curl wget vim tmux less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools sudo \
        tzdata psmisc screen libx11-dev llvm-dev && \
    apt-get clean

##############################################################################
# SSH configuration (not secure, only for development purpose)
##############################################################################
RUN mkdir -p /run/sshd && \
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config

##############################################################################
# Uninstall unnecessary packages and their dependencies
##############################################################################
RUN pip install pip-autoremove && \
    pip-autoremove torch torchvision torchaudio torch-tensorrt \
        pytorch-quantization pytorch-triton \
        flash-attn transformer-engine tensorboard -y

##############################################################################
# Install Miniconda 
##############################################################################
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 || { echo 'Miniconda installation failed' ; exit 1; } && \
    rm -rf ~/miniconda3/miniconda.sh && \
    ~/miniconda3/bin/conda init bash || { echo 'conda init failed' ; exit 1; } && \
    ~/miniconda3/bin/conda config --set auto_activate_base false || { echo 'conda config failed' ; exit 1; }

##############################################################################
# Create Conda Environment and Install PyTorch
##############################################################################
SHELL ["bash", "-c"]
RUN source ~/miniconda3/bin/activate && \
    conda create -y --name flagscale python=3.10.12 && \
    conda activate flagscale && \
    conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia || { echo 'PyTorch installation failed' ; exit 1; }
