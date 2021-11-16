# Open-Vocabulary Detection via Vision and Language Knowledge Distillation
• [Paper](https://arxiv.org/abs/2104.13921) • [Colab](https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/models/official/detection/projects/vild/ViLD_demo.ipynb)

<p style="text-align:center;"><img src="https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/assets/new_teaser.png" alt="teaser" width="500"/></p>

Xiuye Gu, Tsung-Yi Lin, Weicheng Kuo, Yin Cui,
[Open-Vocabulary Detection via Vision and Language Knowledge Distillation](https://arxiv.org/abs/2104.13921).

This repo contains the colab demo, code, and pretrained checkpoints for our open-vocabulary detection method, ViLD (**Vi**sion and **L**anguage **D**istillation).

Open-vocabulary object detection detects objects described by arbitrary text inputs. The fundamental challenge is the availability of training data. Existing object detection datasets only contain hundreds of categories, and it is costly to scale further. To overcome this challenge, we propose ViLD. Our method distills the knowledge from a pretrained open-vocabulary image classification model (teacher) into a two-stage detector (student). Specifically, we use the teacher model to encode category texts and image regions of object proposals. Then we train a student detector, whose region embeddings of detected boxes are aligned with the text and image embeddings inferred by the teacher. 

We benchmark on LVIS by holding out all rare categories as novel categories not seen during training. ViLD obtains 16.1 mask APr, even outperforming the supervised counterpart by 3.8 with a ResNet-50 backbone. The model can directly transfer to other datasets without finetuning, achieving 72.2 AP50, 36.6 AP and 11.8 AP on PASCAL VOC, COCO and Objects365, respectively. On COCO, ViLD outperforms previous SOTA by 4.8 on novel AP and 11.4 on overall AP.

The figure below shows an overview of ViLD's architecture.
![architecture overview](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/assets/new_overview_new_font.png)


# Colab Demo
In this [colab](https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/models/official/detection/projects/vild/ViLD_demo.ipynb) or this [jupyter notebook](./ViLD_demo.ipynb), we created a demo with two examples. You can also try your own images and specify the categories you want to detect. 


# Getting Started
## Prerequisite
* Install [TensorFlow](https://www.tensorflow.org/install).
* Install the packages in [`requirements.txt`](./requirements.txt).


## Data preprocessing
1. Download and unzip the [LVIS v1.0](https://www.lvisdataset.org/dataset) validation sets to `DATA_DIR`.

The `DATA_DIR` should be organized as below:

```
DATA_DIR
+-- lvis_v1_val.json
+-- val2017
|   +-- ***.jpg
|   +-- ...
```

2. Create tfrecords for the validation set (adjust `max_num_processes` if needed; specify `DEST_DIR` to the tfrecords output directory):

```shell
DATA_DIR=[DATA_DIR]
DEST_DIR=[DEST_DIR]
VAL_JSON="${DATA_DIR}/lvis_v1_val.json"
python3 preprocessing/create_lvis_tf_record.py \
  --image_dir="${DATA_DIR}" \
  --json_path="${VAL_JSON}" \
  --dest_dir="${DEST_DIR}" \
  --include_mask=True \
  --split='val' \
  --num_parts=100 \
  --max_num_processes=100
```

## Trained checkpoints
| Method        | Backbone     | Distillation weight | APr   |  APc |  APf | AP   | config | ckpt |
|:------------- |:-------------| -------------------:| -----:|-----:|-----:|-----:|--------|------|
| ViLD          | ResNet-50    | 0.5                 | 16.6  | 19.8 | 28.2 | 22.5 | [vild_resnet.yaml](./configs/vild_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet50_vild.tar.gz)|
| ViLD-ensemble | ResNet-50    | 0.5                 |  18	 | 24.7	| 30.6 | 25.9 | [vild_resnet.yaml](./configs/vild_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet50_vild_ensemble.tar.gz)|
| ViLD          | ResNet-152   | 1.0                 | 19.6	 | 21.6	| 28.5 | 24.0 | [vild_ensemble_resnet.yaml](./configs/vild_ensemble_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet152_vild.tar.gz)|
| ViLD-ensemble | ResNet-152   | 2.0                 | 19.2	 | 24.8	| 30.8 | 26.2 | [vild_ensemble_resnet.yaml](./configs/vild_ensemble_resnet.yaml) |[ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/pretrained_ckpts/resnet152_vild_ensemble.tar.gz)|

## Inference
1. Download the [classification weights](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/weights/clip_synonym_prompt.npy) (CLIP text embeddings) and the [binary masks](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/vild/weights/lvis_rare_masks.npy) for rare categories. And put them in `[WEIGHTS_DIR]`.
2. Download and unzip the trained model you want to run inference in `[MODEL_DIR]`.
3. Replace `[RESNET_DEPTH], [MODEL_DIR], [DATA_DIR], [DEST_DIR], [WEIGHTS_DIR], [CONFIG_FILE]` with your values in the script below and run it.

Please refer [getting_started.md](https://github.com/tensorflow/tpu/blob/master/models/official/detection/GETTING_STARTED.md) for more information.

```shell
BATCH_SIZE=1
RESNET_DEPTH=[RESNET_DEPTH]
MODEL_DIR=[MODEL_DIR]
EVAL_FILE_PATTERN="[DEST_DIR]/val*"
VAL_JSON_FILE="[DATA_DIR]/lvis_v1_val.json"
RARE_MASK_PATH="[WEIGHTS_DIR]/lvis_rare_masks.npy"
CLASSIFIER_WEIGHT_PATH="[WEIGHTS_DIR]/clip_synonym_prompt.npy"
CONFIG_FILE="tpu/models/official/detection/projects/vild/configs/[CONFIG_FILE]"
python3 tpu/models/official/detection/main.py \
  --model="vild" \
  --model_dir="${MODEL_DIR?}" \
  --mode=eval \
  --use_tpu=False \
  --config_file="${CONFIG_FILE?}" \
  --params_override="{ resnet: {resnet_depth: ${RESNET_DEPTH?}}, predict: {predict_batch_size: ${BATCH_SIZE?}}, eval: {eval_batch_size: ${BATCH_SIZE?}, val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} }, frcnn_head: {classifier_weight_path: ${CLASSIFIER_WEIGHT_PATH?}}, postprocess: {rare_mask_path: ${RARE_MASK_PATH?}}}"
```


# License
This repo is under the same license as  [tensorflow/tpu](https://github.com/tensorflow/tpu), see
[license](https://github.com/tensorflow/tpu/blob/master/LICENSE).

# Citation
If you find this repo to be useful to your research, please cite our paper:

```
@article{gu2021open,
  title={Open-Vocabulary Detection via Vision and Language Knowledge Distillation},
  author={Gu, Xiuye and Lin, Tsung-Yi and Kuo, Weicheng and Cui, Yin},
  journal={arXiv preprint arXiv:2104.13921},
  year={2021}
}
```

# Acknowledgements
In this repo, we use [OpenAI's CLIP model](https://github.com/openai/CLIP) as the open-vocabulary image classification model, i.e., the teacher model.

The code is built upon [Cloud TPU detection](https://github.com/tensorflow/tpu/tree/master/models/official/detection).