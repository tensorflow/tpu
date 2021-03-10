# Revisiting ResNets: Improved Training Methodologies and Scaling Rules
<img src="https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/promo_fig.png" height="450" />

We release configs and checkpoints of the ResNet-RS model family trained on ImageNet in Tensorflow 1.

## ImageNet Checkpoint

| Model        | Input Size    | V100 Lat (s) | TPU Lat (ms)    |   Top-1 Accuracy  |   Download |
| ------------ |:-------------:| -----------:|--------:|-----------:|-----------:|
| ResNet-RS-50 | 160x160  | 0.31 | 70  | 78.8 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs50_i160.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-50-i160.tar.gz) |
| ResNet-RS-101 | 160x160 | 0.48 | 120 | 80.3 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs101_i160.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-101-i160.tar.gz) |
| ResNet-RS-101 | 192x192 | 0.70 | 170 | 81.2 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs101_i192.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-101-i192.tar.gz) |
| ResNet-RS-152 | 192x192 | 0.99 | 240 | 82.0 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs152_i192.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-152-i192.tar.gz) |
| ResNet-RS-152 | 224x224 | 1.48 | 320 | 82.2 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs152_i224.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-152-i224.tar.gz) |
| ResNet-RS-152 | 256x256 | 1.76 | 410 | 83.0 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs152_i256.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-152-i256.tar.gz) |
| ResNet-RS-200 | 256x256 | 2.86 | 570 | 83.4 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs200_i256.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-200-i256.tar.gz) |
| ResNet-RS-270 | 256x256 | 3.76 | 780 | 83.8 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs270_i256.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-270-i256.tar.gz) |
| ResNet-RS-350 | 256x256 | 4.72 | 1100| 84.0 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs350_i256.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-350-i256.tar.gz) |
| ResNet-RS-350 | 320x320 | 8.48 | 1630| 84.2 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs350_i320.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-350-i320.tar.gz) |
| ResNet-RS-420 | 320x320 | 10.16 | 2090| 84.4 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs420_i320.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-420-i320.tar.gz) |

Benchmarking details:

*  Latencies on Tesla V100 GPUs are measured withfull precision (float32).
*  Latencies on TPUv3 are measured using bfloat16 precision.
*  All latencies are measured with an initialtraining batch size of 128 images, which is divided by 2 until it fits onto the accelerator.

Code and checkpoints are avaliable in Tensorflow 2 at the official Tensorflow [Model Garden](https://github.com/tensorflow/models/tree/master/official/vision/beta).

## Citation

```make
```
