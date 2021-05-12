# Simple Copy-Paste Augmentation

Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin, Ekin D. Cubuk, Quoc V. Le, Barret Zoph
[Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation](https://arxiv.org/abs/2012.07177)

<img src="https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/copy-paste/copy_paste.jpg" height="320" />

## Training models
To train a mask-rcnn model with Copy-Paste augmentation follow the instruction
[here](https://github.com/tensorflow/tpu/blob/master/models/official/detection/GETTING_STARTED.md) and update the following attributes in the config:

```YAML
# Attributes to update in the config to use Copy-Paste augmentation.
type: 'mask_rcnn'
train:
  pre_parser_dataset:
    file_pattern: <path to the TFRecord training data>
architecture:
  pre_parser: 'extract_objects_parser'
maskrcnn_parser:
  copy_paste: True
```

The 'extract_objects_parser' gets input dataset and parse the objects which will
be pasted in copy-paste augmentation. The path of this dataset can be set via  train.pre_parser_dataset.file_pattern (this path may be set same as the main
training dataset path: train.train_file_pattern).
maskrcnn_parser_with_copy_paste gets input dataset and also output of extract_objects_parser and paste objects on the images to create new images.
It also updates the ground-truth data accordingly.






## Checkpoints

Object detection and instance segmentation on COCO (models trained with Copy-Paste):

| model            | #FLOPs    | #Params  | Box AP (val)   | Mask AP (val)     |   download             |
| -----------------|:---------:| --------:|---------------:|------------------:|-----------------------:|
|                  |           |          |                |                   | [ckpt]() \| [config]() |
|                  |           |          |                |                   | [ckpt]() \| [config]() |
|                  |           |          |                |                   | [ckpt]() \| [config]() |

## Prepare Data

The training expects the data in TFExample format stored in TFRecord.
Tools and scripts are provided to download and convert datasets.

|  Dataset  |      Tool     |
|:---------:|:-------------:|
| COCO      | [instructions](https://cloud.google.com/tpu/docs/tutorials/retinanet#prepare_the_coco_dataset) |
| LVIS      |               |

## Citation

```make
@article{ghiasi2020simple,
  title={Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation},
  author={Ghiasi, Golnaz and Cui, Yin and Srinivas, Aravind and Qian, Rui and Lin, Tsung-Yi and Cubuk, Ekin D and Le, Quoc V and Zoph, Barret},
  journal={arXiv preprint arXiv:2012.07177},
  year={2020}
}
```
