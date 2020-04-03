# Object Detection Models on TPU

## Prerequsite
To get started, make sure to use Tensorflow 1.13+ on Google Cloud. Also here are a few package you need to install to get started:

```bash
sudo apt-get install -y python-tk && \
pip install --user Cython matplotlib opencv-python-headless pyyaml Pillow && \
pip install --user 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
```

Next, download the code from tpu github repository or use the pre-installed Google Cloud VM.

```bash
git clone https://github.com/tensorflow/tpu/
```

## Train RetinaNet on TPU
### Train a vanilla ResNet-50 based RetinaNet.

```bash
TPU_NAME="<your GCP TPU name>"
MODEL_DIR="<path to the directory to store model files>"
RESNET_CHECKPOINT="<path to the pre-trained Resnet-50 checkpoint>"
TRAIN_FILE_PATTERN="<path to the TFRecord training data>"
EVAL_FILE_PATTERN="<path to the TFRecord validation data>"
VAL_JSON_FILE="<path to the validation annotation JSON file>"
python ~/tpu/models/official/detection/main.py \
  --use_tpu=True \
  --tpu="${TPU_NAME?}" \
  --num_cores=8 \
  --model_dir="${MODEL_DIR?}" \
  --mode=train \
  --eval_after_training=True \
  --params_override="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} } }"
```

### Train a custom RetinaNet using the config file.

First, create a YAML config file, e.g. *my_retinanet.yaml*. This file specifies the parameters to be overridden, which should at least include the following fields.

```YAML
# my_retinanet.yaml
type: 'retinanet'
train:
  train_file_pattern: <path to the TFRecord training data>
eval:
  eval_file_pattern: <path to the TFRecord validation data>
  val_json_file: <path to the validation annotation JSON file>
```

Once the YAML config file is created, you can launch the training using the following command.

```bash
TPU_NAME="<your GCP TPU name>"
MODEL_DIR="<path to the directory to store model files>"
python ~/tpu/models/official/detection/main.py \
  --use_tpu=True \
  --tpu="${TPU_NAME?}" \
  --num_cores=8 \
  --model_dir="${MODEL_DIR?}" \
  --mode=train \
  --eval_after_training=True \
  --config_file="my_retinanet.yaml"
```

### Available RetinaNet templates.

* NAS-FPN: [arXiv](https://arxiv.org/abs/1904.07392), [yaml](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/yaml/retinanet_nasfpn.yaml)
* Auto-augument: [arXiv](https://arxiv.org/abs/1805.09501), [yaml](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/yaml/retinanet_autoaugment.yaml)
* SpineNet: [arXiv](https://arxiv.org/abs/1912.05027), [yaml](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet49_retinanet.yaml)

## Export to SavedModel for serving
Once the training is finished, you can export the model in the SavedModel format for serving using the following command.

```bash
EXPORT_DIR="<path to the directory to store the exported model>"
CHECKPOINT_PATH="<path to the pre-trained checkpoint>"
USE_TPU=true
PARAMS_OVERRIDE=""  # if any.
BATCH_SIZE=1
INPUT_TYPE="image_bytes"
INPUT_NAME="input"
INPUT_IMAGE_SIZE="640,640"
OUTPUT_IMAGE_INFO=true
OUTPUT_NORMALIZED_COORDINATES=false
CAST_NUM_DETECTIONS_TO_FLOAT=true
python ~/tpu/models/official/detection/export_saved_model.py \
  --export_dir="${EXPORT_DIR?}" \
  --checkpoint_path="${CHECKPOINT_PATH?}" \
  --use_tpu=${USE_TPU?} \
  --params_override="${PARAMS_OVERRIDE?}" \
  --batch_size=${BATCH_SIZE?} \
  --input_type="${INPUT_TYPE?}" \
  --input_name="${INPUT_NAME?}" \
  --input_image_size="${INPUT_IMAGE_SIZE?}" \
  --output_image_info=${OUTPUT_IMAGE_INFO?} \
  --output_normalized_coordinates=${OUTPUT_NORMALIZED_COORDINATES?} \
  --cast_num_detections_to_float=${CAST_NUM_DETECTIONS_TO_FLOAT?}
```
