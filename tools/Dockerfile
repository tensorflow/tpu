# Docker image for running TPU tensorflow examples.
FROM ubuntu:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python \
    python-pip \
    python-setuptools

RUN pip install tf-nightly

# Checkout tensorflow/tpu at HEAD
RUN git clone https://github.com/tensorflow/tpu.git /tensorflow_models