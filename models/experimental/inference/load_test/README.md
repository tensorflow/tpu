# Example Usage

The following instructions demonstrate how to run MLPerf load test against a Vertex AI Endpoint.

## Deploy model to Vertex AI

In order to deploy the NLP model used in the [MLPerf NLP benchmark](https://github.com/mlcommons/inference/tree/master/language/bert#readme)
please follow the official [documentation](https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api).

## Run the benchmark

Running the benchmark invovles building a docker image and executing the load test from it.

```bash
# Set environment variables
export IMAGE_NAME=load-test-image
export PROJECT_ID=your-gcp-project-id
export ENDPOINT_ID=your-vertex-ai-endpoint-id
export QPS=10

# Build the load test image
docker build -t $IMAGE_NAME -f tools/Dockerfile .

# Start the container in interactive mode
docker run -it \
  -e PROJECT_ID=${PROJECT_ID} \
  -e ENDPOINT_ID=${ENDPOINT_ID} \
  -e QPS=${QPS} \
  $IMAGE_NAME \
  bash
```

Then from within the docker container run:

```bash
# Obtain GCP user credentials
gcloud auth application-default login --no-browser

# Run the load test against Vertex AI Endpoint
cd tpu/models/experimental/inference/load_test/examples
python3 -m loadgen_vertex_gapic_main \
  --project_id=${PROJECT_ID}   \
  --endpoint_id=${ENDPOINT_ID} \
  --qps=${QPS}  # 10 queries per second
```

Other common flags include:

* `duration_ms` - minimum number of miliseconds the benchmark should run (defaults to 1 minute).
* `query_count` - minimum number of queries the benchmark should send (defaults to 1024).
* `total_sample_count` - total number of different samples available to the benchmark (defaults to all the samples available in the dataset file).
* `region` - GCP region in which the Vertex AI endpoint is deployed (defaults to *us-central-1*).
* `features_cache` - a path to the cache file containing the pre-processed features for the SQuAD dataset. Improves the dataset loading time from O(minutes) to O(seconds). If file does not exist, the cache will be created at the specified location.

See `examples/loadgen_vertex_gapic_main.py` for all available flags.


