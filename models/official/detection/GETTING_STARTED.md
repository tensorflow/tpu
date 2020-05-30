# Getting started

## Installation

To get started, make sure you install Tensorflow 1.15+.

* For GPU training, make sure it has the GPU support. See the [guideline](https://www.tensorflow.org/install/gpu) by Tensorflow.

```bash
pip install tensorflow-gpu==1.15  # GPU
```

* For Cloud TPU / TPU Pods training, make sure Tensorflow 1.15+ is pre-installed in your Google Cloud VM.


Also, there are a few packages that you need to install.

```bash
sudo apt-get install -y python-tk && \
pip install --user Cython matplotlib opencv-python-headless pyyaml Pillow && \
pip install --user 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
```

## Dataset download and conversion

Next, download the latest code from [tpu github](https://github.com/tensorflow/tpu) repository.

```bash
git clone https://github.com/tensorflow/tpu/
```

The training expects the data in TFExample format stored in TFRecord.
Tools and scripts are provided to download and convert datasets.

|  Dataset  |      Tool     |
|:---------:|:-------------:|
| ImageNet  | [instructions](https://cloud.google.com/tpu/docs/classification-data-conversion) |
| COCO      | [instructions](https://cloud.google.com/tpu/docs/tutorials/retinanet#prepare_the_coco_dataset) |


## Model Training

We support both GPU training on a single machine, and Cloud TPU / TPU Pods training.
Below we provide sample commands to launch RetinaNet training on different platforms.

### GPU training on a single machine

```bash
MODEL_DIR="<path to the directory to store model files>"
TRAIN_FILE_PATTERN="<path to the TFRecord training data>"
EVAL_FILE_PATTERN="<path to the TFRecord validation data>"
VAL_JSON_FILE="<path to the validation annotation JSON file>"
RESNET_CHECKPOINT="gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602"
python ~/tpu/models/official/detection/main.py \
  --model="retinanet" \
  --model_dir="${MODEL_DIR?}" \
  --mode=train \
  --eval_after_training=True \
  --use_tpu=False \
  --params_override="{ train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} } }"
```

### Training on Cloud TPU

To train this model on Cloud TPU, you will need:

* A GCE VM instance with an associated Cloud TPU resource.
* A GCS bucket to store your training checkpoints (the `--model_dir` flag).
* Install TensorFlow 1.15+ for both GCE VM and Cloud TPU instances.

See the RetinaNet [tutorial](https://cloud.google.com/tpu/docs/tutorials/retinanet)
for more instructuions about TPU training.

```bash
TPU_NAME="<your GCP TPU name>"
MODEL_DIR="<path to the directory to store model files>"
TRAIN_FILE_PATTERN="<path to the TFRecord training data>"
EVAL_FILE_PATTERN="<path to the TFRecord validation data>"
VAL_JSON_FILE="<path to the validation annotation JSON file>"
RESNET_CHECKPOINT="gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602"
python ~/tpu/models/official/detection/main.py \
  --model="retinanet" \
  --model_dir="${MODEL_DIR?}" \
  --use_tpu=True \
  --tpu="${TPU_NAME?}" \
  --num_cores=8 \
  --mode=train \
  --eval_after_training=True \
  --params_override="{ train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} } }"
```

### Training on Cloud TPU Pods

You can leverage large [Cloud TPU Pods](https://cloud.google.com/blog/products/ai-machine-learning/googles-scalable-supercomputers-for-machine-learning-cloud-tpu-pods-are-now-publicly-available-in-beta)
in Google Cloud to significantly improve the training performance.

```bash
TPU_POD_NAME="<your GCP TPU name>"
NUM_CORES=<num cores in TPU pod>  # e.g. v3-32 offers 32 cores.
MODEL_DIR="<path to the directory to store model files>"
TRAIN_FILE_PATTERN="<path to the TFRecord training data>"
EVAL_FILE_PATTERN="<path to the TFRecord validation data>"
VAL_JSON_FILE="<path to the validation annotation JSON file>"
RESNET_CHECKPOINT="gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602"
CONFIG=""
python ~/tpu/models/official/detection/main.py \
  --model="retinanet" \
  --model_dir="${MODEL_DIR?}" \
  --use_tpu=True \
  --tpu="${TPU_POD_NAME?}" \
  --num_cores=${NUM_CORES} \
  --mode=train \
  --config_file="" \
  --params_override="{ train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} } }"
```

### Customize configurations

The framework supports three levels of parameter overrides to accommodate different use cases.

1. `<xxx>_config.py` under [`./configs`](https://github.com/tensorflow/tpu/tree/master/models/official/detection/configs) directory.

  This defines and sets the default values of all the parameters required by the particular model.

2. `<xxx>.yaml` and override through the `--config_file` flag.

  This provides the first level override on top of the default defined by `<xxx>_config.py`.
  One can use it to define a controlled experiment by
  first defining a `.yaml` file as the template and passing to the `--config_file` flag
  and then changing only one or two parameters using the `--params_override` flag.

3. parameters in JSON string and override through the `--params_override` flag.

  This provides the final override on top of 1 and 2.

#### Example: Train RetinaNet using customized configurations.

First, create a YAML config file, e.g. *my_retinanet.yaml*,
to define training / evaluation dataset.

```YAML
# my_retinanet.yaml
type: 'retinanet'
train:
  train_file_pattern: <path to the TFRecord training data>
eval:
  eval_file_pattern: <path to the TFRecord validation data>
  val_json_file: <path to the validation annotation JSON file>
```

Override learning rate hyper-parameter via `--params_override` in the launch command.

```bash
python ~/tpu/models/official/detection/main.py \
  ... \
  --config_file="my_retinanet.yaml" \
  --params_override="{ train: { learnin_rate: { init_learning_rate: 0.2 } } }"
```

## Model Export

### Export to SavedModel

Given the checkpoint, one can easily export the [SavedModel](https://www.tensorflow.org/guide/saved_model) for serving using the following command.

```bash
EXPORT_DIR="<path to the directory to store the exported model>"
CHECKPOINT_PATH="<path to the checkpoint>"
PARAMS_OVERRIDE=""  # if any.
BATCH_SIZE=1
INPUT_TYPE="image_bytes"
INPUT_NAME="input"
INPUT_IMAGE_SIZE="640,640"
python ~/tpu/models/official/detection/export_saved_model.py \
  --export_dir="${EXPORT_DIR?}" \
  --checkpoint_path="${CHECKPOINT_PATH?}" \
  --params_override="${PARAMS_OVERRIDE?}" \
  --batch_size=${BATCH_SIZE?} \
  --input_type="${INPUT_TYPE?}" \
  --input_name="${INPUT_NAME?}" \
  --input_image_size="${INPUT_IMAGE_SIZE?}" \
```

### Export to TF-lite

Given the exported SavedModel, one can further convert it to the [TF-lite](https://www.tensorflow.org/lite) format that can be deployed on mobile platform.

```bash
SAVED_MODEL_DIR="<path to the SavedModel directory>"
OUTPUT_DIR="<path to the directory to store the tflite model>"
python ~/tpu/models/official/detection/export_tflite_model.py \
  --saved_model_dir="${SAVED_MODEL_DIR?}" \
  --output_dir="${OUTPUT_DIR?}" \
```

## Model Inference

Given the checkpoint, one can easily run the model inference using the following command.

```bash
MODEL="retinanet"
IMAGE_SIZE=640
CHECKPOINT_PATH="<path to the checkpoint>"
PARAMS_OVERRIDE=""  # if any.
LABEL_MAP_FILE="~/tpu/models/official/detection/datasets/coco_label_map.csv"
IMAGE_FILE_PATTERN="<path to the JPEG image that you want to run inference on>"
OUTPUT_HTML="./test.html"
python ~/tpu/models/official/detection/inference.py \
  --model="${MODEL?}" \
  --image_size=${IMAGE_SIZE?} \
  --checkpoint_path="${CHECKPOINT_PATH?}" \
  --label_map_file="${LABEL_MAP_FILE?}" \
  --image_file_pattern="${IMAGE_FILE_PATTERN?}" \
  --output_html="${OUTPUT_HTML?}" \
```
