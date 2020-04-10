# TPU Object Detection and Segmentation Framework

TPU Object Detection and Segmentation Framework provides implementations of
common image classification, object detection and instance segmentation models in Tensorflow.
Our models produce the competitive results,
can be trained on multiple platforms including GPU and [TPUs](https://cloud.google.com/tpu),
and have been highly optimized for TPU performance.
It also features latest research including
[Auto-Augument](https://arxiv.org/abs/1805.09501),
[NAS-FPN](https://arxiv.org/abs/1904.07392),
[ShapeMask](https://arxiv.org/abs/1904.03239), and
[SpineNet](https://arxiv.org/abs/1912.05027).

![alt text](https://storage.googleapis.com/gweb-cloudblog-publish/images/Mask_R-CNN_instance_segmentation_results..max-2000x2000.png)
 ** Instance segmentation results of our Mask R-CNN model.


## Updates

* April 10, 2020: Launch the new
[README.md](https://github.com/tensorflow/tpu/blob/master/models/official/detection/README.md),
[GETTING_STARTED.md](https://github.com/tensorflow/tpu/blob/master/models/official/detection/GETTING_STARTED.md), and
[MODEL_ZOO.md](https://github.com/tensorflow/tpu/blob/master/models/official/detection/MODEL_ZOO.md).
Release initial models.

## Major Features

* Tasks:
  - Image classification
  - Object detection
  - Instance segmentation
* Meta-architectures:
  - RetinaNet
  - Faster / Mask R-CNN
  - **[ShapeMask](https://arxiv.org/abs/1904.03239)**
* Backbones:
  - ResNet
  - **[SpineNet](https://arxiv.org/abs/1912.05027)**
* Feature pyramids:
  - FPN
  - **[NAS-FPN](https://arxiv.org/abs/1904.07392)**
* Other model features:
  - **[Auto-Augument](https://arxiv.org/abs/1805.09501)**
* Training platforms:
  - Single machine GPUs
  - [Cloud TPU](https://cloud.google.com/tpu)
  - [Cloud TPU Pods](https://cloud.google.com/blog/products/ai-machine-learning/googles-scalable-supercomputers-for-machine-learning-cloud-tpu-pods-are-now-publicly-available-in-beta)


## Model Zoo

[MODEL_ZOO.md](https://github.com/tensorflow/tpu/blob/master/models/official/detection/MODEL_ZOO.md)
provides a large collection of baselines and checkpoints for object detection, instance segmentation, and image classification.


## Get started

Please follow the instructions in [GETTING_STARTED.md](https://github.com/tensorflow/tpu/blob/master/models/official/detection/GETTING_STARTED.md).
