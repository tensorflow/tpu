# Neural Collaborative Filtering (NCF) on TPU

## Prerequisites

### Setup a Google Cloud project

Follow the instructions at the [Quickstart Guide](https://cloud.google.com/tpu/docs/quickstart)
to get a GCE VM with access to Cloud TPU.

To run this model, you will need:

* A GCE VM instance with an associated Cloud TPU resource. It might be helpful if the VM has a large number of CPUs and large memory as it is used for generating training and evaluation data. TF1.2 is required.
* A GCS bucket to store data. To avoid downloading MovieLens dataset, you can copy it from `gs://ncf/data_dir`.

### Setup NCF from tensorflow/models

Neural collaborative filtering on Cloud TPU depends on [the same model under tensorflow/models](https://github.com/tensorflow/models/tree/master/official/recommendation). In your working directory, run `git clone https://github.com/tensorflow/models.git`, and add `models/` to your python path by running `export PYTHONPATH=$PYTHONPATH:/your/working/directory/models/`.

### Setup TPU embedding library

In your working directory, run the following to get the TPU embedding library.

```shell
wget https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/contrib/tpu/python/tpu/tpu_embedding.py
```

This will be updated once the TPU embedding library is included in future TF releases.

## Setup NCF
Copy `./ncf_main.py` to your working directory.

```
wget https://raw.githubusercontent.com/tensorflow/tpu/master/models/experimental/ncf/ncf_main.py
```

## Train and Eval

```shell
BUCKET_NAME=your_bucket_name
EXPERIMENT_NAME=your_experiment_name
python ncf_main.py --data_dir gs://${BUCKET_NAME}/data_dir --learning_rate 0.00136794 --beta1 0.781076 --beta2 0.977589 --epsilon 7.36321e-8  --model_dir gs://${BUCKET_NAME}/model_dirs/${EXPERIMENT_NAME} |& tee ${EXPERIMENT_NAME}.log
```
Most of the time, the hit rate metric (HR) reaches 0.635 in around 10 epochs.

