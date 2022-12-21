# MLPerf inference benchmark for Vertex Prediction

This folder containers a tool for benchmarking models deployed on
Vertex Prediction or running locally on TensorFlow Model Server using
[MLPerf inference loadgen](https://github.com/mlcommons/inference/tree/master/loadgen).

## Example Usage

The following instructions demonstrate how to run MLPerf load test against a Vertex AI Endpoint.

### Deploy model to Vertex AI

In order to deploy the NLP model used in the [MLPerf NLP benchmark](https://github.com/mlcommons/inference/tree/master/language/bert#readme)
please follow the official [documentation](https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api).

### Run the benchmark from Docker container

Running the benchmark involves building a docker image and executing the load test from it.

```bash
IMAGE_NAME=load-test-image

# Build the load test image
docker build -t $IMAGE_NAME -f tools/Dockerfile .

# Start the container in interactive mode
docker run -it $IMAGE_NAME bash
```

After docker container is built, you can run benchmark from docker container.

### Run the benchmark locally

Alternatively you can manually setup your environment and run benchmark from
the environment you are using. This is more convenient if you want to run
benchmark from Colab or Jupyter Notebook.

```bash
# Install dependencies.
pip3 install --user absl-py numpy pillow mock tensorflow-serving-api \
  transformers google-cloud-aiplatform tf-models-official

# Download and build MLPerf loadgen.
# See https://github.com/mlcommons/inference/tree/master/loadgen/demos/lon for details.
git clone --recurse-submodules -b r1.0 https://github.com/mlcommons/inference.git
pushd inference/loadgen
CFLAGS="-std=c++14 -O3" python3 setup.py bdist_wheel
pip3 install --force-reinstall dist/mlperf_loadgen-*
popd

# Download benchmark tool
git clone https://github.com/tensorflow/tpu.git
cd tpu/models/experimental/inference
```

#### Run benchmark

The commands to run benchmarks from docker container or from local environment
are same.

```bash
# Obtain GCP user credentials. Follow the instructions on the screen.
# You might not need to run this if you are running from Colab or Jupyter Notebook
# that is already configured to use your project.
gcloud auth application-default login --no-browser

# Set parameters.
PROJECT_ID=your-gcp-project-id
ENDPOINT_ID=123456789123
REGION=us-central1
DURATION=10000 # In milliseconds
API_TYPE=rest # rest | grpc | gapic
QPS=10 # QPS to send requests at, you can specify multiple values.
DATASET=generic_jsonl # criteo | sentiment_bert | squad_bert | generic_jsonl
DATA_FILE=gs://path/to/requests.jsonl # A jsonl file with requests is required for generic_jsonl, criteo and sentiment_bert datasets. Either a path to a GCS location or a local path.
CSV_REPORT_FILENAME="local file path"  # Optional file name to dump benchmark results to.

# Run the benchmark against Vertex AI Endpoint.
cd tpu/models/experimental/inference/load_test/examples
python3 -m loadgen_vertex_main  \
  --project_id=${PROJECT_ID}    \
  --endpoint_id=${ENDPOINT_ID}  \
  --region=${REGION}            \
  --min_duration_ms=${DURATION} \
  --api_type=${API_TYPE}        \
  --qps=${QPS}                  \
  --dataset=${DATASET}          \
  --data_file=${DATA_FILE}      \
  --csv_report_filename=${CSV_REPORT_FILENAME}
```

The gRPC protocol will only work with private endpoints. Please follow `Setup private endpoint for online prediction` section from [Vertex AI Samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/vertex_endpoints/optimized_tensorflow_runtime/tabular_optimized_online_prediction.ipynb) to set up a private endpoint.

See `examples/loadgen_vertex_main.py` for all available flags.

<sub>Readme author: cezarym@</sub>
