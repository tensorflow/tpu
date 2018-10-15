# Docker image of TensorBoard and TPU Profiler.
FROM ubuntu:bionic
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        build-essential \
        git \
        python \
        python-pip \
        python-setuptools && \
    pip install tensorflow==1.11 && \
    pip install google-cloud-storage && \
    pip install google-api-python-client && \
    pip install oauth2client && \
    pip install cloud-tpu-profiler==1.11