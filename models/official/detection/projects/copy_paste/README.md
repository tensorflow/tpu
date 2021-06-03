# Simple Copy-Paste Augmentation

Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D. Cubuk, Quoc V. Le, Barret Zoph
[Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/abs/2012.07177)

<img src="https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/copy_paste.jpg" height="320" />

## Training models
To train a mask-rcnn model with Copy-Paste augmentation follow the instruction
[here](https://github.com/tensorflow/tpu/blob/master/models/official/detection/GETTING_STARTED.md) and update the following attributes in the config:

```YAML
# Attributes to update in the config to use Copy-Paste augmentation.
type: 'mask_rcnn' # or 'cascade_mask_rcnn'
train:
  pre_parser_dataset:
    file_pattern: <path to the TFRecord training data>
architecture:
  pre_parser: 'extract_objects_parser'
maskrcnn_parser:
  copy_paste: True
```

The [extract_objects_parser](https://github.com/tensorflow/tpu/blob/master/models/official/detection/dataloader/extract_objects_parser.py) gets an input dataset and parses the objects which will
be pasted in copy-paste augmentation. The path of this dataset can be set via
train.pre_parser_dataset.file_pattern (this path may be set same as the main
training dataset path: train.train_file_pattern).
[maskrcnn_parser_with_copy_paste](https://github.com/tensorflow/tpu/blob/master/models/official/detection/dataloader/maskrcnn_parser_with_copy_paste.py)
gets input dataset and also output of
extract_objects_parser and pastes objects on the images to create new images
with Copy-Paste augmentation. Also, it updates the ground-truth data accordingly.






## Checkpoints

Checkpoints of object detection and instance segmentation models trained on COCO:

| model                           | #FLOPs    | #Params  | Box AP (val)   | Mask AP (val)     |   download             |
| --------------------------------|:---------:| --------:|---------------:|------------------:|-----------------------:|
| Res50-FPN (1024) w/ Copy-Paste  | 431 B     | 48 M     |      48.3      | 42.4              | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/maskrcnn_resnet50_1024.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/copy_paste/configs/maskrcnn_resnet50_1024.yaml) |
| Res101-FPN (1024) w/ Copy-Paste | 509 B     |  67 M    |      49.8      | 43.5              | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/maskrcnn_resnet101_1024.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/copy_paste/configs/maskrcnn_resnet101_1024.yaml) |
| Res101-FPN (1280) w/ Copy-Paste | 693 B     | 67 M     |      50.3      | 44.1              | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/maskrcnn_resnet101_1280.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/copy_paste/configs/maskrcnn_resnet101_1280.yaml) |
| Eff-B7 FPN (640) w/ Copy-Paste  | 286 B     |  86 M    |       50.0     |  43.7             | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/maskrcnn_effb7_640.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/copy_paste/configs/maskrcnn_effb7_640.yaml) |
| Eff-B7 FPN (1024) w/ Copy-Paste | 447 B     |  86 M    |       51.9     | 45.1              | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/maskrcnn_effb7_1024.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/copy_paste/configs/maskrcnn_effb7_1024.yaml) |
| Eff-B7 FPN (1280) w/ Copy-Paste | 595 B     |  86 M    |       52.5     | 45.8              | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/maskrcnn_effb7_1280.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/copy_paste/configs/maskrcnn_effb7_1280.yaml) |
| Cascade Eff-B7 FPN (1280) w/ Copy-Paste | 854 B | 118 M |     54.0      |  46.3             | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/cascade_maskrcnn_effb7_1280.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/copy_paste/configs/cascade_maskrcnn_effb7_1280.yaml) |
| Cascade Eff-B7 NAS-FPN (1280)                             |  1440 B | 185 M   |     54.4        |      46.6         | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/cascade_maskrcnn_effb7_nasfpn_1280.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/copy_paste/configs/cascade_maskrcnn_effb7_nasfpn_vanilla_1280.yaml) |
| Cascade Eff-B7 NAS-FPN (1280) w/ Copy-Paste               | 1440 B |  185 M    |    55.8        |      47.1         | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/cascade_maskrcnn_effb7_nasfpn_1280_copypaste.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/copy_paste/configs/cascade_maskrcnn_effb7_nasfpn_1280.yaml) |
| Cascade Eff-B7 NAS-FPN (1280) w/ self-training Copy-Paste | 1440 B |  185 M    |    57.0   |  48.8    | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/cascade_maskrcnn_effb7_nasfpn_1280_selftraining_copypaste.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/copy_paste/configs/cascade_maskrcnn_effb7_nasfpn_1280.yaml) |



## Prepare Data

The training expects the data in TFExample format stored in TFRecord.
Tools and scripts are provided to download and convert datasets.

|  Dataset  |      Tool     |
|:---------:|:-------------:|
| COCO      | [instructions](https://cloud.google.com/tpu/docs/tutorials/retinanet#prepare_the_coco_dataset) |

## Citation

```make
@article{ghiasi2020simple,
  title={Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation},
  author={Ghiasi, Golnaz and Cui, Yin and Srinivas, Aravind and Qian, Rui and Lin, Tsung-Yi and Cubuk, Ekin D and Le, Quoc V and Zoph, Barret},
  journal={arXiv preprint arXiv:2012.07177},
  year={2020}
}
```
