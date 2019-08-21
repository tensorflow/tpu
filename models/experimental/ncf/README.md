# Neural Collaborative Filtering (NCF) on TPU

## Prerequisites

### Setup a Google Cloud project

Follow the instructions at the [Quickstart Guide](https://cloud.google.com/tpu/docs/quickstart)
to get a GCE VM with access to Cloud TPU.

To run this model, you will need:

* A GCE VM instance with an associated Cloud TPU resource. It might be helpful if the VM has a large number of CPUs and large memory as it is used for generating training and evaluation data. TF nightly is required.
* A GCS bucket to store data. To avoid downloading MovieLens dataset, you can copy it from `gs://ncf/data_dir`.

### Setup NCF from tensorflow/models

Neural collaborative filtering on Cloud TPU depends on [the same model under tensorflow/models](https://github.com/tensorflow/models/tree/master/official/recommendation). In your working directory, run `git clone https://github.com/tensorflow/models.git`, and add `models/` to your python path by running `export PYTHONPATH=$PYTHONPATH:/your/working/directory/models/`.

## Setup NCF
Copy `./ncf_main.py` to your working directory.

```
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/experimental/ncf/ncf_main.py
```

Setup a Google Cloud Bucket for your training data and model storage:

```shell
BUCKET_NAME=your_bucket_name
```

## Run the training data generator

From the `models/` directory run the command:

```shell
python official/recommendation/create_ncf_data.py \
--data_dir gs://${BUCKET_NAME}/ncf_data \
--meta_data_file_path gs://${BUCKET_NAME}/ncf_data/metadata.json \
--train_prebatch_size 98304 \
--eval_prebatch_size 160000
```

This will download an preprocess your data and take several minutes to process
the data.

NOTE The prebatch sizes and number of epochs passed as arguments to
`create_ncf_data.py` must match what you pass to ncf_main.py. In this
case they match the defaults values for `--batch_size` and `--eval_batch_size`
and the number of epochs for both scripts defaults to 14.

## Train and Eval

```shell
EXPERIMENT_NAME=your_experiment_name
python ncf_main.py \
--train_dataset_path="gs://${BUCKET_NAME}/ncf_data/training_cycle_{}/*" \
--eval_dataset_path="gs://${BUCKET_NAME}/ncf_data/eval_data/*" \
--input_meta_data_path=gs://${BUCKET_NAME}/ncf_data/metadata.json \
--model_dir gs://${BUCKET_NAME}/model_dirs/${EXPERIMENT_NAME} |& tee ${EXPERIMENT_NAME}.log
```

Most of the time, the hit rate metric (HR) reaches 0.635 in around 10 epochs.

