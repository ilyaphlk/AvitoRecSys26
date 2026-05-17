FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip python3.11-dev \
    git curl vim unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf awscliv2.zip aws/

RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /project
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir --timeout=300 -r requirements.txt
RUN python3.11 -m pip install --force-reinstall cffi
RUN python3.11 -m pip install --upgrade s3fs