# TPU Embedding example model

## Prerequisites

### Setup a Google Cloud project

Follow the instructions at the [Quickstart Guide](https://cloud.google.com/tpu/docs/quickstart)
to get a GCE VM with access to Cloud TPU.

To run this model, you will need:

* A GCE VM instance with an associated Cloud TPU resource. It might be helpful if the VM has a large number of CPUs and large memory as it is used for generating training and evaluation data.
* A GCS bucket to store data.

## Setup Model
Clone the `tpu` respository and move to the example directory:

```shell
git clone https://github.com/tensorflow/tpu
cd tpu/models/experimental/embedding
```

Setup a Google Cloud Bucket for your training data and model storage:

```shell
BUCKET_NAME=your_bucket_name
```

Create a new `embedding` subdirectory in your bucket.

## Run the training data generator

```shell
python3 models/experimental/embedding/create_data.py \
--train_dataset_path gs://${BUCKET_NAME}/embedding/train.tfrecord \
--eval_dataset_path gs://${BUCKET_NAME}/embedding/eval.tfrecord
```

## Train and Eval

```shell
python3 models/experimental/embedding/model.py \
--train_dataset_path="gs://${BUCKET_NAME}/embedding/train.tfrecord*" \
--eval_dataset_path="gs://${BUCKET_NAME}/embedding/eval.tfrecord*" \
--model_dir="gs://${BUCKET_NAME}/embedding/model_dir"
```
