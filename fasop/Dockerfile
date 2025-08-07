ARG FROM_IMAGE_NAME=pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel
FROM ${FROM_IMAGE_NAME}

# Install dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        net-tools \
        iproute2 \
        inetutils-ping \
        openssh-client \
        vim \
	wget \
	git \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install transformers==4.46.2 datasets

WORKDIR /workspace
RUN wget https://github.com/huggingface/transformers/archive/refs/tags/v4.46.2.tar.gz \
 && tar xvfz v4.46.2.tar.gz \
 && rm v4.46.2.tar.gz

RUN git clone https://github.com/ai-computing/aicomp.git \
 && cp aicomp/compiler_fx/modeling_llama.py  /workspace/transformers-4.46.2/src/transformers/models/llama

#download FASOP from Google drive
#RUN pip3 install gdown \
# && ~/.local/bin/gdown https://drive.google.com/uc?id=1w5urKalJVFZscNfWrq4A3l11dqxrZDgL&usp

#or just copy from local directory if you have it already.	
COPY tdpp.zip /workspace/tdpp.zip
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
 	unzip
RUN unzip tdpp.zip
RUN mkdir -p tdpp/log2/tp_output
RUN rm tdpp.zip
COPY _06_get_layer_time.py /workspace/tdpp/Megatron-LM/_06_get_layer_time.py
COPY profile.py /workspace/tdpp/FASOP/known_cost/profile.py

 
