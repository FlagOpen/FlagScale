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
        curl wget vim tmux less unzip \
        htop iftop iotop ca-certificates openssh-client openssh-server \
        rsync iputils-ping net-tools \
        tzdata psmisc screen libx11-dev && \
    apt-get clean

##############################################################################
# SSH configuration (not secure, only for development purpose)
##############################################################################
RUN mkdir -p /run/sshd && \
    ssh-keygen -t rsa -N "" -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config

##############################################################################
# Install Miniconda 
##############################################################################
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 || { echo 'Miniconda installation failed' ; exit 1; } && \
    rm -rf ~/miniconda3/miniconda.sh && \
    ~/miniconda3/bin/conda init bash || { echo 'conda init failed' ; exit 1; } && \
    ~/miniconda3/bin/conda config --set auto_activate_base false || { echo 'conda config failed' ; exit 1; }