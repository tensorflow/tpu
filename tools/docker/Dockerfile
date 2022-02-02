# Docker image for running TPU tensorflow examples.
FROM ubuntu:bionic

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        wget \
        sudo \
        gnupg \
        lsb-release \
        ca-certificates \
        build-essential \
        git \
        python3 \
        python3-pip \
        python-setuptools && \
    export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb https://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" > /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3 /tmp/get-pip.py

RUN pip3 install pyyaml && \
    pip3 install wheel && \
    pip3 install tensorflow==2.8.0 && \
    pip3 install google-cloud-storage && \
    pip3 install google-api-python-client && \
    pip3 install oauth2client

# Checkout tensorflow/models at the appropriate branch
RUN git clone -b r2.8.0 --depth 1 https://github.com/tensorflow/models.git /tensorflow_models

# Checkout tensorflow/tpu at the appropriate branch
RUN git clone -b r2.8 --depth 1 https://github.com/tensorflow/tpu.git /tensorflow_tpu_models
