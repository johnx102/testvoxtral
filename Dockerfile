FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

RUN rm -f /etc/apt/sources.list.d/*.list

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

WORKDIR /

# Update and upgrade the system packages
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends sudo ca-certificates git wget curl bash libgl1 libx11-6 software-properties-common ffmpeg build-essential -y &&\
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN apt-get update -y && \
    apt-get install python3.10 python3.10-dev python3.10-venv python3-pip -y --no-install-recommends && \
    ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm -f /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*


# 2.  cache directories
RUN mkdir -p /cache/models /root/.cache/torch

# 3.  clone whisperx *before* pip needs it
#RUN git clone --depth 1 https://github.com/m-bain/whisperx.git /tmp/whisperx && \
#    cd /tmp/whisperx && \
#    git reset --hard 58f00339af7dcc9705ef40d97a1f40764b7cf555

# 4.  requirements file (local copy that uses the clone)
COPY builder/requirements.txt /builder/requirements.txt

# 5.  python dependencies
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install hf_transfer \ 
 && python3 -m pip install --no-cache-dir -r /builder/requirements.txt

# 6.  local VAD model
COPY models/whisperx-vad-segmentation.bin /root/.cache/torch/whisperx-vad-segmentation.bin

# 7.  builder scripts + model downloader
COPY builder /builder
RUN chmod +x /builder/download_models.sh
RUN --mount=type=secret,id=hf_token /builder/download_models.sh
# 8.  application code
COPY src .

CMD ["python3", "-u", "/rp_handler.py"]
