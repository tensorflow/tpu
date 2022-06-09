# Example Usage

The following instructions demonstrate how to run MLPerf load test against a Vertex AI Endpoint.

## Deploy model to Vertex AI

In order to deploy the NLP model used in the [MLPerf NLP benchmark](https://github.com/mlcommons/inference/tree/master/language/bert#readme)
please follow the official [documentation](https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api).

## Run the benchmark

Running the benchmark involves building a docker image and executing the load test from it.

```bash
IMAGE_NAME=load-test-image

# Build the load test image
docker build -t $IMAGE_NAME -f tools/Dockerfile .

# Start the container in interactive mode
docker run -it $IMAGE_NAME bash
```

Then from within the docker container run:

```bash
# Obtain GCP user credentials. Follow the instructions on the screen.
gcloud auth application-default login --no-browser

# Set parameters.
PROJECT_ID=your-gcp-project-id
ENDPOINT_ID=123456789123
REGION=us-central1
DURATION=10000 # In milliseconds
API_TYPE=rest # rest | grpc | gapic
QPS=10
DATASET=criteo # criteo | sentiment_bert | squad_bert
DATA_FILE=gs://path/to/requests.jsonl # A jsonl file with requests is required for criteo and sentiment_bert datasets. Either a path to a GCS location or a local path.

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
  --data_file=${DATA_FILE}
```

The gRPC protocol will only work with private endpoints. Please follow `Setup private endpoint for online prediction` section from [Vertex AI Samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/community/vertex_endpoints/optimized_tensorflow_runtime/tabular_optimized_online_prediction.ipynb) to set up a private endpoint.

See `examples/loadgen_vertex_main.py` for all available flags.

<sub>Readme author: cezarym@</sub>
