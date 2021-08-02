# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
FROM google/cloud-sdk:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        pybind11-dev \
        python3-setuptools \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install absl-py numpy pillow mock tensorflow-serving-api
RUN git clone --recurse-submodules -b r1.0 https://github.com/mlcommons/inference.git
RUN cd inference/loadgen && \
    CFLAGS="-std=c++14 -O3" python3 setup.py bdist_wheel && \
    pip3 install --force-reinstall dist/mlperf_loadgen-*
RUN rm -rf inference

WORKDIR /
RUN git clone https://github.com/tensorflow/tpu.git

ENV PYTHONPATH "${PYTHONPATH}:/tpu/models/experimental/inference"

CMD ["/bin/bash"]
