# Rethinking Pre-Training and Self-Training

Barret Zoph, Golnaz Ghiasi, Tsung-Yi Lin, Yin Cui, Hanxiao Liu, Ekin D. Cubuk, Quoc V. Le
[[arXiv](https://arxiv.org/abs/2006.06882)]

<img src="https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/rethinking-pre-training-and-self-training/logo.png" height="180" />

We release the checkpoints of teacher model and student model in rethinking
pre-training and self-training.

## Checkpoint

Object detection on COCO (results with SoftNMS):

| model        | #FLOPs    | #Params  | AP (val)   | AP (test_dev) |   download |
| -------------|:---------:| --------:|-----------:|--------------:|-----------:|
| SpineNet-143 | 524B      |    67M   | 50.9       | 51.0          | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenet-143-best.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet143_retinanet.yaml) |
| SpineNet-143 w/self-training | 524B      |    67M   | 52.6       | 52.8          | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/rethinking-pre-training-and-self-training/spinenet-143-ssl.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/self_training/configs/coco_spinenet143_retinanet.yaml) |
| SpineNet-190 | 1885B | 164M | 52.6       | 52.8          | [ckpt](https://storage.cloud.google.com/cloud-tpu-checkpoints/detection/retinanet/spinenet-190-best.tar.gz?organizationId=433637338589) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/configs/spinenet/spinenet190_retinanet.yaml) |
| SpineNet-190 w/self-training | 1885B      |    164M   | 54.2       | 54.3          | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/rethinking-pre-training-and-self-training/spinenet-190-ssl.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/self_training/configs/coco_spinenet190_retinanet.yaml) |




Semantic segmentation on PASCAL VOC 2012:

| model                  | #FLOPs    | #Params  | mIOU (val)   | mIOU (test) |   download |
| -----------------------|:---------:| --------:|-----------:|--------------:|-----------:|
| EfficientNet-B7-NASFPN | 60B       |    71M   | 85.2       | -             | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/rethinking-pre-training-and-self-training/efficientnet-b7-nasfpn-teacher.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/self_training/configs/pascal_seg_efficientnet-b7-nasfpn.yaml) |
| EfficientNet-B7-NASFPN w/ self-training | 60B       |    71M   | 86.7       | -             | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/rethinking-pre-training-and-self-training/efficientnet-b7-nasfpn-ssl.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/self_training/configs/pascal_seg_efficientnet-b7-nasfpn.yaml) |
| EfficientNet-L2-NASFPN |     229B   | 485M       | 88.7       |   -   | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/rethinking-pre-training-and-self-training/efficientnet-l2-nasfpn-teacher.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/self_training/configs/pascal_seg_efficientnet-l2-nasfpn.yaml) |
| EfficientNet-L2-NASFPN w/ self-training |     229B   | 485M       | 90.0       | 90.5             | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/rethinking-pre-training-and-self-training/efficientnet-l2-nasfpn-ssl.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/self_training/configs/pascal_seg_efficientnet-l2-nasfpn.yaml) |

## Prepare Data

The training expects the data in TFExample format stored in TFRecord.
Tools and scripts are provided to download and convert datasets.

|  Dataset  |      Tool     |
|:---------:|:-------------:|
| ImageNet  | [instructions](https://cloud.google.com/tpu/docs/classification-data-conversion) |
| COCO      | [instructions](https://cloud.google.com/tpu/docs/tutorials/retinanet#prepare_the_coco_dataset) |
| PASCAL    | [instructions](https://github.com/tensorflow/models/blob/31b0e5184a8b86063760ef5b8ea19ed6cb0e5d9e/research/deeplab/g3doc/pascal.md)

## Citation

```make
@article{zoph20selftraining,
  title={Rethinking pre-training and self-training},
  author={Barret Zoph and Golnaz Ghiasi and Tsung-Yi Lin and Yin Cui and Hanxiao Liu and Ekin D. Cubuk and Quoc V. Le},
  journal={CoRR},
  volume={abs/2006.06882},
  year={2020}
}
```
