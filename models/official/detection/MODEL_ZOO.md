# TPU Object Detection and Segmentation Model Zoo

## Introduction
Model zoo provides a large collection of baselines and checkpoints for object detection, instance segmentation, and image classification.

## Object Detection and Instance Segmentation
### Common Settings and Notes
* We provide models based on two detection frameworks, [RetinaNet](https://arxiv.org/abs/1708.02002) or [Mask R-CNN](https://arxiv.org/abs/1703.06870), and three backbones, [ResNet-FPN](https://arxiv.org/abs/1612.03144), [ResNet-NAS-FPN](https://arxiv.org/abs/1904.07392), or [SpineNet](https://arxiv.org/abs/1912.05027).
* Models are all trained on COCO train2017 and evaluated on COCO val2017.
* Training details:
  * Models finetuned from ImageNet pretrained checkpoints adopt the 36 epochs (~3x) schedule, where 1x is around 12 COCO epochs.
  * Most models trained from scratch adopt the 72 or 350 epochs schedule.
  * The default training data augmentation implements horizontal flipping and scale jittering with a random scale between [0.5, 2.0].
  * Unless noted, all models are trained with l2 weight regularization and ReLU activation.
  * We use batch size 256 and stepwise learning rate that decays at the last 30 and 10 epoch.
  * We use square image as input by resizing the long side of an image to the target size then padding the short side with zeros.
* [Inference latency](https://github.com/tensorflow/tpu/blob/master/models/official/detection/utils/saved_model_benchmark.py):
  * Latency is measured on a V100/P100 GPU from inputs to raw outputs (without image pre-processing or post-processing, e.g. NMS).
  * TensorRT optimization is not implemented in all tests.

### COCO Object Detection Baselines
#### RetinaNet (ImageNet pretrained)
Coming soon.

#### RetinaNet (Trained from scratch)
| model        | resolution    | epochs  | FLOPs (B)     | params (M) | V100 / P100 <br /> lat (ms/im)  |  box AP |   download |
| ------------ |:-------------:| ---------:|-----------:|--------:|------:|---------:|-----------:|
| R50-FPN      | 640x640       |    350    | 97.0 | 34.0 | 23 / 37 |40.4 |[ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/r50-fpn.tar.gz?organizationId=433637338589) \| config|
| R101-FPN  | 1024x1024       |    350     | 326.3 | 53.1 | 55 / 95 | 43.9 | ckpt \| config |
| R152-FPN  | 1280x1280       |    350     | 630.5 | 68.7 | 100 / 167 |45.2 | ckpt \| config |
| R50-NAS-FPN  | 640x640       |    72     | 140.6 | 60.3 | 29 / 48 |37.3 | N/A |
| R50-NAS-FPN  | 640x640       |    350    | 140.6 | 60.3 | 29 / 48 |42.4 |[ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/r50-nasfpn.tar.gz?organizationId=433637338589) \| config|
| SpineNet-49  | 640x640       |    72     | 85.4| 28.5 | 24 / 38 |37.7| N/A |
| SpineNet-49  | 640x640       |    350    | 85.4| 28.5 | 24 /38 |42.8|[ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenet-49.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet49_retinanet.yaml) |
| SpineNet-49S | 640x640     |    350    | 33.8 | 11.9 | 19 / 26|39.5 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenet-49S.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet49S_retinanet.yaml) |
| SpineNet-96  | 1024x1024     |    350    | 265.4 | 43.0 | 53 / 87 |46.7 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenet-96.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet96_retinanet.yaml) |
| SpineNet-143 | 1280x1280     |    350    | 524.0 | 67.0 |97 / 159 |48.0 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenet-143.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet143_retinanet.yaml) |


SpineNet models trained with stochastic depth and swish activation for a longer shedule:
| model        | resolution    | epochs  | FLOPs (B)    | params (M) |  box AP |   download |
| ------------ |:-------------:| ---------:|-----------:|--------:|--------:|-----------:|
| SpineNet-49S | 640x640       |    500    | 33.8  | 11.9 |41.5  | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenet-49S-best.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet49S_retinanet.yaml) |
| SpineNet-49  | 640x640       |    500    | 85.4  | 28.5 |44.3  | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenet-49-best.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet49_retinanet.yaml) |
| SpineNet-96  | 1024x1024     |    500    | 265.4 | 43.0 | 48.5 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenet-96-best.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet96_retinanet.yaml) |
| SpineNet-143 | 1280x1280     |    500    | 524.0 | 67.0 | 50.6 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenet-143-best.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet143_retinanet.yaml) |
| SpineNet-190 | 1280x1280     |    400    | 1885.0 | 163.6 | 52.0 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenet-190-best.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet190_retinanet.yaml) |

#### Mobile RetinaNet (Trained from scratch)
| model           | resolution    | epochs   | FLOPs (B)    | params (M) |  box AP |   download |
| --------------- |:-------------:| ----------:|-----------:|--------:|--------:|-----------:|
| SpineNetMB-49   | 384x384       |    600     | 1.0 | 2.34 | 28.6 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenetmbconv-49-best.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet-mbconv49_retinanet.yaml) |

### Instance Segmentation Baselines
#### Mask R-CNN (ImageNet pretrained)
Coming soon.

#### Mask R-CNN (Trained from scratch)
| model        | resolution    | epochs  | FLOPs (B)  | params (M)  |  box AP |  mask AP  |   download |
| ------------ |:-------------:| ---------:|-----------:|--------:|--------:|-----------:|-----------:|
| SpineNet-49  | 640x640       |    350    | 215.7 | 40.8 | 42.8 | 37.8 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/maskrcnn/spinenet-49.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet49_mrcnn.yaml) |
| SpineNet-96  | 1024x1024     |    350    | 314.6 | 55.2 | 46.8 | 41.2 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/maskrcnn/spinenet-96.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet96_mrcnn.yaml) |
| SpineNet-143 | 1280x1280     |    350    | 498.4 | 79.2 | 48.7 | 42.6 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/maskrcnn/spinenet-143.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet143_mrcnn.yaml) |

SpineNet-190 trained with stochastic depth and swish activation for a longer shedule:
| model        | resolution    | epochs  | FLOPs (B)  | params (M)  |  box AP |  mask AP  |   download |
| ------------ |:-------------:| ---------:|-----------:|--------:|--------:|-----------:|-----------:|
| SpineNet-190  | 1536x1536      |    400    | 1685.7 | 168.2 | 52.0 | 45.9 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/maskrcnn/spinenet-190.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet190_mrcnn.yaml) |

## Image Classification
### Common Settings and Notes
* We provide ImageNet and [iNaturalist-2017](https://arxiv.org/abs/1707.06642) pretrained checkpoints for [ResNet](https://arxiv.org/abs/1512.03385) and [SpineNet](https://arxiv.org/abs/1912.05027) models at various scales.
* Training details:
  * All models are trained from scratch for 200 epochs with cosine learning rate decay and batch size 4096.
  * Unless noted, all models are trained with l2 weight regularization and ReLU activation.

### ImageNet Baselines
| model        | resolution    | epochs  | FLOPs (B)    | params (M)  |  Top-1  |  Top-5   |   download |
| ------------ |:-------------:| ---------:|-----------:|--------:|--------:|---------:|-----------:|
| ResNet-34    | 224x224       |    200    | 3.7 | 21.8 | 74.4 | 92.0 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/classification/resnet-34-imagenet.tar.gz?organizationId=433637338589) \| config|
| ResNet-50    | 224x224       |    200    | 4.1 | 25.6 | 77.1 | 93.6 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/classification/resnet-34-imagenet.tar.gz?organizationId=433637338589) \| config|
| ResNet-101   | 224x224       |    200    | 7.8 | 44.6 | 78.2 | 94.2 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/classification/resnet-101-imagenet.tar.gz?organizationId=433637338589) \| config |
| ResNet-152   | 224x224       |    200    | 11.5 | 60.2 | 78.4 | 94.2 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/classification/resnet-152-imagenet.tar.gz?organizationId=433637338589) \| config |
| SpineNet-49  | 224x224       |    200    | 3.5 | 22.1 | 77.0 | 93.3 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/classification/spinenet-49-imagenet.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet49_classification.yaml)|
| SpineNet-96  | 224x224       |    200    | 5.7 | 36.5 | 78.2 | 94.0 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/classification/spinenet-96-imagenet.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet96_classification.yaml)|
| SpineNet-143 | 224x224       |    200    | 9.1 | 60.5 | 79.0 | 94.4 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/classification/spinenet-143-imagenet.tar.gz?organizationId=433637338589)\| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet143_classification.yaml)|

SpineNet models trained with stochastic depth, swish activation, and label smoothing:
| model        | resolution    | epochs  | FLOPs (B)  | params (M)  |  Top-1  |  Top-5   |   download |
| ------------ |:-------------:| ---------:|-----------:|--------:|--------:|---------:|-----------:|
| SpineNet-49  | 224x224       |    200    | 3.5 | 22.1 | 78.1 | 94.0 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/classification/spinenet-49-best-imagenet.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet49_classification.yaml) |
| SpineNet-96  | 224x224       |    200    | 5.7 | 36.5 | 79.4 | 94.6 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/classification/spinenet-96-best-imagenet.tar.gz?organizationId=433637338589)\| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet96_classification.yaml)|
| SpineNet-143 | 224x224       |    200    | 9.1 | 60.5 | 80.1 | 95.0 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/classification/spinenet-143-best-imagenet.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet143_classification.yaml) |
| SpineNet-190 | 224x224       |    200    | 19.1 | 127.1 | 80.8 | 95.3 | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/classification/spinenet-190-best-imagenet.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet190_classification.yaml) |

### iNaturalist-2017 Baselines
| model        | resolution    | epochs  | FLOPs (B)   | params (M)  |  Top-1  |  Top-5   |
| ------------ |:-------------:| ---------:|-----------:|--------:|--------:|---------:|
| ResNet-34    | 224x224       |    200    | 3.7 | 23.9 | 54.1 | 76.7 |
| ResNet-50    | 224x224       |    200    | 4.1 | 33.9 | 54.6 | 77.2 |
| ResNet-101   | 224x224       |    200    | 7.8 | 52.9 | 57.0 | 79.3 |
| ResNet-152   | 224x224       |    200    | 11.5 | 68.6 | 58.4 | 80.2 |
| SpineNet-49  | 224x224       |    200    | 3.5 | 23.1 | 59.3 | 81.9 |
| SpineNet-96  | 224x224       |    200    | 5.7 | 37.6 | 61.7 | 83.4 |
| SpineNet-143 | 224x224       |    200    | 9.1 | 61.6 | 63.6 | 84.8 |

SpineNet models trained with stochastic depth, swish activation, and label smoothing:
| model        | resolution    | epochs  | FLOPs (B)   | params (M)  |  Top-1  |  Top-5   |
| ------------ |:-------------:| ---------:|-----------:|--------:|--------:|---------:|
| SpineNet-49  | 224x224       |    200    | 3.5 | 23.1 | 63.3 | 85.1 |
| SpineNet-96  | 224x224       |    200    | 5.7 | 37.6 | 64.7 | 85.9 |
| SpineNet-143 | 224x224       |    200    | 9.1 | 61.6 | 66.7 | 87.1 |
| SpineNet-190 | 224x224       |    200    | 19.1 | 129.2 | 67.6 | 87.4 |
