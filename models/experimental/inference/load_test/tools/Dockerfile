FROM tensorflow/tensorflow:nightly

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        pybind11-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install absl-py numpy pillow mock tensorflow-serving-api --no-deps
RUN git clone --recurse-submodules -b r1.0 https://github.com/mlcommons/inference.git
RUN cd inference/loadgen && \
    CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel && \
    pip install --force-reinstall dist/mlperf_loadgen-0.5a0-cp36-cp36m-linux_x86_64.whl
RUN rm -rf inference

WORKDIR /
RUN git clone https://github.com/tensorflow/tpu.git

ENV PYTHONPATH "${PYTHONPATH}:/tpu/models/experimental/inference"

CMD ["/bin/bash"]
