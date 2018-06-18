# Docker image for running TPU tensorflow examples.
FROM ubuntu:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python \
    python-pip \
    python-setuptools

RUN pip install tensorflow==1.8

# Checkout tensorflow/models at the appropriate branch
RUN git clone -b r1.8.1 https://github.com/tensorflow/models.git /tensorflow_models

# Checkout tensorflow/tpu at the appropriate branch
RUN git clone -b r1.8 https://github.com/tensorflow/tpu.git /tensorflow_tpu_models

COPY mnist_tpu.py /tensorflow_models/official/mnist/mnist_tpu.py
