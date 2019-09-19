# UNet 3D Model Codebase on TPU

This folder contains an implementation of the [3D UNet](https://arxiv.org/abs/1606.06650) model.

## Prerequsites

In Google Cloud console, please run the following command to create both cloud VM and TPU VM.

```shell
ctpu up -name=[tpu_name]  -tf-version=nightly -tpu-size=v3-8  -zone=us-central1-b
```

## Setup

Before running any binary, please install necessary packages on cloud VM.

```shell
pip install -r requirements.tx
```

## Data Preparation

This software uses TFRecords as input. We provide example scripts to convert
Numpy (.npy) files or NIfTI-1 (.nii) files to TFRecords, using the Liver Tumor
Segmentation (LiTS) dataset
(Christ et al. https://competitions.codalab.org/competitions/17094).
You can download the dataset by registering on the competition website.

**Example**:

```shell
cd data_preprocess

# Change input_path and output_path in convert_lits_nii_to_npy.py
# Then run the script to convert nii to npy.
python convert_lits_nii_to_npy.py

# Convert npy files to TFRecords.
python convert_lits.py \
  --image_file_pattern=Downloads/.../volume-{}.npy \
  --label_file_pattern=Downloads/.../segmentation-{}.npy \
  --output_path=Downloads/...
```

## Training

Working configs on TPU V3-8:

+ TF 1.13, train_batch_size=32, use_batch_norm=false, use_bfloat16=true
+ TF 1.13, train_batch_size=32, use_batch_norm=true, use_bfloat16=false
+ TF 1.13, train_batch_size=16, use_batch_norm=true, use_bfloat16=true
+ tf-nightly, train_batch_size=32, use_batch_norm=true, use_bfloat16=true

The following example shows how to train volumic UNet on TPU v3-8.
The loss is *adaptive_dice32*. The training batch size is 32. For detail config, refer to `unet_config.py` and `v3-8_128x128x128_ce.yaml`.

**Example**:

```shell
DATA_BUCKET=<GS bucket for data>
TRAIN_FILES="${DATA_BUCKET}/tfrecords/trainbox*.tfrecord"
VAL_FILES="${DATA_BUCKET}/tfrecords/validationbox*.tfrecord"
MODEL_BUCKET=<GS bucket for model checkpoints>
EXP_NAME=unet_20190610_dice_t1

python unet_main.py \
--use_tpu \
--tpu=<TPU name> \
--model_dir="gs://${MODEL_BUCKET}/models/${EXP_NAME}" \
--training_file_pattern="${TRAIN_FILES}" \
--eval_file_pattern="${VAL_FILES}" \
--iterations_per_loop=10 \
--mode=train \
--num_cores=8 \
--config_file="./configs/cloud/v3-8_128x128x128_ce.yaml" \
--params_override="{\"optimizer\":\"momentum\",\"train_steps\":100}"
```

The following script example is for running evaluation on TPU v3-8. It is only
one line change from previous script: changes the mode to "eval". Also, modify
the "eval_steps" in the yaml file or the "--params_override" to adjust
evaluation duration.

### Train with Spatial Partition

The following example specify spatial partition with the "--input_partition_dims" flag.

**Example: Train with 8-way spatial partition**:

```shell
DATA_BUCKET=<GS bucket for data>
TRAIN_FILES="${DATA_BUCKET}/tfrecords/trainbox*.tfrecord"
VAL_FILES="${DATA_BUCKET}/tfrecords/validationbox*.tfrecord"
MODEL_BUCKET=<GS bucket for model checkpoints>
EXP_NAME=unet_20190610_dice_t1

python unet_main.py \
--use_tpu \
--tpu=<TPU name> \
--model_dir="gs://${MODEL_BUCKET}/models/${EXP_NAME}" \
--training_file_pattern="${TRAIN_FILES}" \
--eval_file_pattern="${VAL_FILES}" \
--iterations_per_loop=10 \
--mode=train \
--num_cores=8 \
--input_partition_dims=[1,8,1,1,1] \
--config_file="./configs/cloud/v3-8_128x128x128_ce.yaml" \
--params_override="{\"optimizer\":\"momentum\",\"train_steps\":100}"
```

## Evaluation

```shell
DATA_BUCKET=<GS bucket for data>
TRAIN_FILES="${DATA_BUCKET}/tfrecords/trainbox*.tfrecord"
VAL_FILES="${DATA_BUCKET}/tfrecords/validationbox*.tfrecord"
MODEL_BUCKET=<GS bucket for model checkpoints>
EXP_NAME=unet_20190610_dice_t1

python unet_main.py \
--use_tpu \
--tpu=<TPU name> \
--model_dir="gs://${MODEL_BUCKET}/models/${EXP_NAME}" \
--training_file_pattern="${TRAIN_FILES}" \
--eval_file_pattern="${VAL_FILES}" \
--iterations_per_loop=10 \
--mode="eval" \
--num_cores=8 \
--config_file="./configs/cloud/v3-8_128x128x128_ce.yaml" \
--params_override="{\"optimizer\":\"momentum\",\"eval_steps\":10}"
```


## Export Saved Model

Exports model that takes serialized tensorflow.Example as input.

```shell
CHECKPOINT_DIR="<checkpoint folder>"
EXPORT_DIR="<output folder>"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/model.ckpt-4200"
CONFIG="${CHECKPOINT_DIR}/params.yaml"
USE_TPU=false
BATCH_SIZE=1
INPUT_TYPE="tf_example"
INPUT_NAME="serialized_example"

python export_saved_model.py \
  --export_dir="${EXPORT_DIR?}" \
  --checkpoint_path="${CHECKPOINT_PATH?}" \
  --config_file="${CONFIG}" \
  --use_tpu=${USE_TPU?} \
  --input_type="${INPUT_TYPE?}" \
  --input_name="${INPUT_NAME?}" \
  --batch_size=${BATCH_SIZE?}
```


Exports model that takes serialized numpy array as input.

```shell
CHECKPOINT_DIR="<checkpoint folder>"
EXPORT_DIR="<output folder>"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/model.ckpt-4200"
CONFIG="${CHECKPOINT_DIR}/params.yaml"
USE_TPU=false
BATCH_SIZE=1
INPUT_TYPE="image_tensor"
INPUT_NAME="input"

python export_saved_model.py \
  --export_dir="${EXPORT_DIR?}" \
  --checkpoint_path="${CHECKPOINT_PATH?}" \
  --config_file="${CONFIG}" \
  --use_tpu=${USE_TPU?} \
  --input_type="${INPUT_TYPE?}" \
  --input_name="${INPUT_NAME?}" \
  --batch_size=${BATCH_SIZE?}
```

### Run Inference with the exported model

Inference with tfrecord file.

```shell
IMAGE_FILE_PATTERN="<path to .tfrecord.gz file>"
SAVED_MODEL_DIR="<saved model folder>"
TAG_SET="serve"
INPUT_TYPE="tf_example"
INPUT_NODE="Placeholder:0"
CLASSES_NODE="unet/Classes:0"
SCORES_NODE="unet/Scores:0"
OUTPUT_DIR="${SAVED_MODEL_DIR?}/output"
python saved_model_inference.py \
  --image_file_pattern="${IMAGE_FILE_PATTERN?}" \
  --saved_model_dir="${SAVED_MODEL_DIR?}" \
  --tag_set="${TAG_SET?}" \
  --input_type="${INPUT_TYPE?}" \
  --input_node="${INPUT_NODE?}" \
  --output_classes_node="${CLASSES_NODE?}" \
  --output_scores_node="${SCORES_NODE?}" \
  --output_dir="${OUTPUT_DIR?}"
```

To Visualize

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

file_path = '<output npz file>'
with tf.gfile.Open(file_path, 'r') as f:
  npzfile = np.load(f, allow_pickle=False)

print(npzfile.files)
scores = npzfile['scores']
classes = npzfile['classes']

plt.imshow(classes[..., 64])
```
