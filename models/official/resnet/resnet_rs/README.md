# Revisiting ResNets: Improved Training Methodologies and Scaling Rules
<img src="https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/promo_fig.png" height="450" />

We release configs and checkpoints of the ResNet-RS model family trained on ImageNet in Tensorflow 1.

## ImageNet Checkpoint

| Model        | Input Size    | Latency (ms)    |   Top-1   |   download |
| ------------ |:-------------:| -----------:|--------:|-----------:|
| ResNet-RS-50 | 160x160  | 74.0  | 78.8 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs50_i160.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-50-i160.tar.gz) |
| ResNet-RS-101 | 160x160 | 119.1 | 80.3 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs101_i160.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-101-i160.tar.gz) |
| ResNet-RS-101 | 192x192 | 167.9 | 81.2 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs101_i192.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-101-i192.tar.gz) |
| ResNet-RS-152 | 192x192 | 239.8 | 82.0 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs152_i192.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-152-i192.tar.gz) |
| ResNet-RS-152 | 224x224 | 318.0 | 82.2 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs152_i224.yaml) \| [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/resnet-rs/resnet-rs-152-i224.tar.gz) |
| ResNet-RS-152 | 256x256 | 409.6 | 83.0 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs152_i256.yaml) \| ckpt |
| ResNet-RS-200 | 256x256 | 568.9 | 83.4 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs200_i256.yaml) \| ckpt |
| ResNet-RS-270 | 256x256 | 781.7 | 83.8 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs270_i256.yaml) \| ckpt |
| ResNet-RS-350 | 256x256 | 1101.1| 84.0 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs350_i256.yaml) \| ckpt |
| ResNet-RS-350 | 320x320 | 1625.4| 84.2 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs350_i320.yaml) \| ckpt |
| ResNet-RS-420 | 320x320 | 2301.1| 84.4 | [config](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs/configs/resnetrs420_i320.yaml) \| ckpt |

* Latency numbers are the training time measured on TPUv3 devices.

Code and checkpoints are avaliable in Tensorflow 2 at the official Tensorflow [Model Garden](https://github.com/tensorflow/models/tree/master/official/vision/beta).

## Citation

```make
```
