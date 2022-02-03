# TensorFlow Serving with TPU VM Example

This contains an *experimental* fork of TensorFlow Serving `Dockerfile`s specifically for usage with [TPU VM](https://cloud.google.com/tpu/docs/users-guide-tpu-vm).

You can use TensorFlow Serving with TPU VMs in the same way as you can use TensorFlow serving with CPU/GPU VMs.

This document assumes all commands are being run on a TPU VM, e.g. created with:

```
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
  --zone=${ZONE} \
  --accelerator-type=${ACCELERATOR_TYPE} \
  --version=${VERSION}
```

For more information about using TensorFlow Serving with Docker, please refer to [TensorFlow Serving with Docker](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md).

For more information about TPU VMs and Cloud TPUs, please refer to the [official Cloud TPU Documentation](https://cloud.google.com/tpu).


# Example Usage

The following instructions demonstrate how you can use the provided Dockerfiles to
create your own model server running on TPU VMs.

## Set sample environment variables
```
export IMAGE_NAME=tf-serve-tpu
export CONTAINER_NAME=$USER-$IMAGE_NAME
```

## Build TensorFlow Serving for TPU VM

Start by building the base Docker image for TF serving.

```
docker build --pull -t ${IMAGE_NAME}-dev \
  -f Dockerfile.devel-tpu .
```

Next, build the model server container.

```
docker build -t ${IMAGE_NAME} \
  --build-arg=TF_SERVING_BUILD_IMAGE=${IMAGE_NAME}-dev \
  -f Dockerfile.tpu .
```

* Note: this uses a version of TensorFlow that is fixed at a known stable commit.

## Start the model server
Make sure you set `MODEL_NAME`.

```
docker run -d -p 8500:8500 --name ${CONTAINER_NAME} \
             --privileged \
             -v "/lib/libtpu.so:/lib/libtpu.so" \
             -v "/home/$USER/models:/models" \
             -e MODEL_NAME=${MODEL_NAME} \
             ${IMAGE_NAME}
```
