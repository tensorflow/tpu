# Deeplab on TPU

## Prerequisites

### Setup a Google Cloud project

Follow the instructions at the [Quickstart Guide](https://cloud.google.com/tpu/docs/quickstart)
to get a GCE VM with access to Cloud TPU.

To run this model, you will need:

* A GCE VM instance with an associated Cloud TPU resource
* A GCS bucket to store your training checkpoints
* A GCS bucket to store your training and evaluation data.

### Setup Deeplab under tensorflow/models

Deeplab on Cloud TPU depends on [Deeplab under tensorflow/models](https://github.com/tensorflow/models/tree/master/research/deeplab). Please follow the [instructions](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md) to add the library to `PYTHONPATH` and test the installation.

You can use their [script](https://github.com/tensorflow/models/blob/master/research/deeplab/datasets/download_and_convert_voc2012.sh) to download PASCAL VOC 2012 semantic segmentation dataset and convert it to TFRecord.

You can download their [pretrained checkpoints](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md). In particular, we use a [modified resnet 101 pretrained on ImageNet](http://download.tensorflow.org/models/resnet_v1_101_2018_05_04.tar.gz) below.

## Train and Eval

```shell
python main.py \
--mode='train' \
--num_shards=8 \
--alsologtostderr=true \
--model_dir=${MODEL_DIR} \
--dataset_dir=${DATASET_DIR} \
--init_checkpoint=${INIT_CHECKPOINT} \
--model_variant=resnet_v1_101_beta \
--image_pyramid=1. \
--aspp_with_separable_conv=false \
--multi_grid=1 \
--multi_grid=2 \
--multi_grid=4 \
--decoder_use_separable_conv=false
```
You can use `mode=eval` for evaluation after training is completed. The model should train to close to 0.77 MIOU in around 9 hours.

