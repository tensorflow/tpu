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
