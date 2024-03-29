# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#swsok, change runtime to devel image to include nvcc and cuda libs.
#FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# metainformation
LABEL org.opencontainers.image.version = "2.0"
LABEL org.opencontainers.image.authors = "contact@eleuther.ai"
LABEL org.opencontainers.image.source = "https://www.github.com/eleutherai/gpt-neox"
LABEL org.opencontainers.image.licenses = " Apache-2.0"
LABEL org.opencontainers.image.base.name="docker.io/nvidia/cuda:11.7.1-runtime-ubuntu20.04"

#### System package (uses default Python 3 version in Ubuntu 20.04)
RUN apt-get update -y && \
    apt-get install -y \
        git python3.9 python3-dev libpython3-dev python3-pip sudo pdsh \
        htop llvm-9-dev tmux zstd software-properties-common build-essential autotools-dev \
        nfs-common pdsh cmake g++ gcc curl wget vim less unzip htop iftop iotop ca-certificates ssh \
        rsync iputils-ping net-tools libcupti-dev libmlx4-1 infiniband-diags ibutils ibverbs-utils \
        rdmacm-utils perftest rdma-core nano && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    pip install --upgrade pip && \
    pip install gpustat

### SSH
RUN mkdir /var/run/sshd && \
    # Prevent user being kicked off after login
    sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd && \
    echo 'AuthorizedKeysFile     .ssh/authorized_keys' >> /etc/ssh/sshd_config && \
    echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config && \
    # FIX SUDO BUG: https://github.com/sudo-project/sudo/issues/42
    echo "Set disable_coredump false" >> /etc/sudo.conf

# Expose SSH port
EXPOSE 22

#### OPENMPI
ENV OPENMPI_BASEVERSION=4.1
ENV OPENMPI_VERSION=${OPENMPI_BASEVERSION}.0
RUN mkdir -p /build && \
    cd /build && \
    wget -q -O - https://download.open-mpi.org/release/open-mpi/v${OPENMPI_BASEVERSION}/openmpi-${OPENMPI_VERSION}.tar.gz | tar xzf - && \
    cd openmpi-${OPENMPI_VERSION} && \
    ./configure --prefix=/usr/local/openmpi-${OPENMPI_VERSION} && \
    make -j"$(nproc)" install && \
    ln -s /usr/local/openmpi-${OPENMPI_VERSION} /usr/local/mpi && \
    # Sanity check:
    test -f /usr/local/mpi/bin/mpic++ && \
    cd ~ && \
    rm -rf /build

# Needs to be in docker PATH if compiling other items & bashrc PATH (later)
ENV PATH=/usr/local/mpi/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:${LD_LIBRARY_PATH}

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/mpi/bin/mpirun /usr/local/mpi/bin/mpirun.real && \
    echo '#!/bin/bash' > /usr/local/mpi/bin/mpirun && \
    echo 'mpirun.real --allow-run-as-root --prefix /usr/local/mpi "$@"' >> /usr/local/mpi/bin/mpirun && \
    chmod a+x /usr/local/mpi/bin/mpirun

#### User account
RUN useradd --create-home --uid 1000 --shell /bin/bash mchorse && \
    usermod -aG sudo mchorse && \
    echo "mchorse ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

## SSH config and bashrc
RUN mkdir -p /home/mchorse/.ssh /job && \
    echo 'Host *' > /home/mchorse/.ssh/config && \
    echo '    StrictHostKeyChecking no' >> /home/mchorse/.ssh/config && \
    echo 'export PDSH_RCMD_TYPE=ssh' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/home/mchorse/.local/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export PATH=/usr/local/mpi/bin:$PATH' >> /home/mchorse/.bashrc && \
    echo 'export LD_LIBRARY_PATH=/usr/local/lib:/usr/local/mpi/lib:/usr/local/mpi/lib64:$LD_LIBRARY_PATH' >> /home/mchorse/.bashrc

#### Python packages
RUN pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117 && pip cache purge
COPY requirements/requirements.txt .
COPY requirements/requirements-wandb.txt .
COPY requirements/requirements-onebitadam.txt .
COPY requirements/requirements-sparseattention.txt .
COPY requirements/requirements-flashattention.txt .
RUN pip install -r requirements.txt && pip install -r requirements-onebitadam.txt && \
    pip install -r requirements-sparseattention.txt && \
    pip install -r requirements-flashattention.txt && \
    pip install -r requirements-wandb.txt && \
    pip install protobuf==3.20.* && \
    pip cache purge

## Install APEX
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597

COPY megatron/ megatron
RUN python megatron/fused_kernels/setup.py install

#swsok, fix compatibility issue
#RUN sed -i "s/from requests.packages.urllib3.util.retry/from urllib3.util.retry/g" /usr/local/lib/python3.8/dist-packages/best_download/__init__.py
RUN pip uninstall urllib3 -y && \
        pip install urllib3==1.26.16

#swsok, fix train_batch miscalculation error.
RUN sed -e "899 i\\        #swsok, fix train_batch calculation error." -i /usr/local/lib/python3.8/dist-packages/deepspeed/runtime/config.py
RUN sed -e "900 i\\        train_batch = None" -i /usr/local/lib/python3.8/dist-packages/deepspeed/runtime/config.py

#swsok, install and setup ssh
RUN sed -i "/#Port/cPort 60022" /etc/ssh/sshd_config
RUN chown -R mchorse:mchorse /home/mchorse/.ssh
RUN echo 'Host *' >> /home/mchorse/.ssh/config && \
    echo 'Port 60022' >> /home/mchorse/.ssh/config
#EXPOSE 22

#swsok, install nsight systems
RUN apt-get install -y --no-install-recommends apt-transport-https ca-certificates gnupg wget && \
    rm -rf /var/lib/apt/lists/*
RUN wget -qO - https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/nvidia.pub | apt-key add -
RUN echo "deb https://developer.download.nvidia.com/devtools/repos/ubuntu2004/amd64/ /" >> /etc/apt/sources.list.d/nsight.list && \
    apt-get update -y && \
    apt-get install -y --no-install-recommends nsight-systems-2023.4.1

# Clear staging
RUN mkdir -p /tmp && chmod 0777 /tmp

#### SWITCH TO mchorse USER
USER mchorse
RUN ssh-keygen -q -t rsa -N '' -f /home/mchorse/.ssh/id_rsa

RUN cat /home/mchorse/.ssh/id_rsa.pub >> /home/mchorse/.ssh/authorized_keys
COPY scripts_swsok/run_sshd.sh /home/mchorse/run_sshd.sh
WORKDIR /gpt-neox

CMD ["nsys", "status", "-e"]
CMD ["/home/mchorse/run_sshd.sh"]
