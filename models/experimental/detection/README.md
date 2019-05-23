# Object Detection Models on TPU

## Prerequsite
To get started, make sure to use Tensorflow nightly on Google Cloud. Also here are a few package you need to install to get started:
```
sudo apt-get install -y python-tk && \
pip install Cython matplotlib opencv-python-headless pyyaml Pillow && \
pip install 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
```

## Train RetinaNet on TPU
```
TPU_NAME=
MODEL_DIR=
RESNET_CHECKPOINT=
TRAIN_FILE_PATTERN=
EVAL_FILE_PATTERN=
VAL_JSON_FILE=
python ~/tpu/models/experimental/detection/main.py \
  --use_tpu=True \
  --tpu="${TPU_NAME?}" \
  --num_cores=8 \
  --model_dir="${MODEL_DIR?}" \
  --mode=train \
  --eval_after_training=True \
  --params_overrides="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?}, eval_samples: 5000 } }"
```

## Export to SavedModel for serving
```
EXPORT_DIR=
CHECKPOINT_PATH=
USE_TPU=true
PARAMS_OVERRIDES=""
BATCH_SIZE=1
INPUT_TYPE="image_bytes"
INPUT_NAME="input"
INPUT_IMAGE_SIZE="640,640"
OUTPUT_IMAGE_INFO=true
OUTPUT_NORMALIZED_COORDINATES=false
CAST_NUM_DETECTIONS_TO_FLOAT=true
python ~/tpu/models/experimental/detection/export_saved_model.py \
  --export_dir="${EXPORT_DIR?}" \
  --checkpoint_path="${CHECKPOINT_PATH?}" \
  --use_tpu=${USE_TPU?} \
  --params_overrides="${PARAMS_OVERRIDES?}" \
  --batch_size=${BATCH_SIZE?} \
  --input_type="${INPUT_TYPE?}" \
  --input_name="${INPUT_NAME?}" \
  --input_image_size="${INPUT_IMAGE_SIZE?}" \
  --output_image_info=${OUTPUT_IMAGE_INFO?} \
  --output_normalized_coordinates=${OUTPUT_NORMALIZED_COORDINATES?} \
  --cast_num_detections_to_float=${CAST_NUM_DETECTIONS_TO_FLOAT?}
```
