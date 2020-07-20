# Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset

Menglin Jia*, Mengyun Shi*, Mikhail Sirotenko*, Yin Cui*, Claire Cardie, Bharath Hariharan, Hartwig Adam, Serge Belongie (*equal contribution)
[[dataset](https://fashionpedia.github.io/home/index.html)] [[arXiv](https://arxiv.org/abs/2004.12276)]

<img src="https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/model.png" height="240"/>

We release the checkpoints of Attribute-Mask R-CNN model with ResNet-FPN and [SpineNet](https://arxiv.org/abs/1912.05027) backbone.

Other code including data conversion, model training and inference will be released soon.

## Checkpoint

Object detection and instance segmentation on Fashionpedia:

| backbone       | input <br/> size | lr <br/> sched | FLOPs  | Params | box AP <br/> IoU / IoU+F1 | mask AP <br/> IoU / IoU+F1 | download |
| ---------------|:----------:|:--------------:|:------:|:------:|:----:|:----:|:---------:|
| ResNet-50 FPN  | 1024 | 1x             | 296.7B | 46.4M | 38.7 / 26.6 | 34.3 / 25.5 | N/A |
| ResNet-50 FPN  | 1024 | 2x             | 296.7B | 46.4M | 41.6 / 29.3 | 38.1 / 28.5 | N/A |
| ResNet-50 FPN  | 1024 | 3x             | 296.7B | 46.4M | 43.4 / 30.7 | 39.2 / 29.5 | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-r50-fpn.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/fashionpedia/configs/yaml/r50fpn_amrcnn.yaml) |
| ResNet-50 FPN  | 1024 | 6x             | 296.7B | 46.4M | 42.9 / 31.2 | 38.9 / 30.2 | N/A |
| ResNet-101 FPN | 1024 | 1x             | 374.3B | 65.4M | 41.0 / 28.6 | 36.7 / 27.6 | N/A |
| ResNet-101 FPN | 1024 | 2x             | 374.3B | 65.4M | 43.5 / 31.0 | 39.2 / 29.8 | N/A |
| ResNet-101 FPN | 1024 | 3x             | 374.3B | 65.4M | 44.9 / 32.8 | 40.7 / 31.4 | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-r101-fpn.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/fashionpedia/configs/yaml/r101fpn_amrcnn.yaml) |
| ResNet-101 FPN | 1024 | 6x             | 374.3B | 65.4M | 44.3 / 32.9 | 39.7 / 31.3 | N/A |
| SpineNet-49    | 1024 | 6x             | 267.2B | 40.8M | 43.7 / 32.4 | 39.6 / 31.4 | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-spinenet-49.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/fashionpedia/configs/yaml/spinenet49_amrcnn.yaml) |
| SpineNet-96    | 1024 | 6x             | 314.0B | 55.2M | 46.4 / 34.0 | 41.2 / 31.8 | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-spinenet-96.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/fashionpedia/configs/yaml/spinenet96_amrcnn.yaml) |
| SpineNet-143   | 1280 | 6x             | 498.0B | 79.2M | 48.7 / 35.7 | 43.1 / 33.3 | [ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/fashionpedia/fashionpedia-spinenet-143.tar.gz) \| [config](https://github.com/tensorflow/tpu/blob/master/models/official/detection/projects/fashionpedia/configs/yaml/spinenet143_amrcnn.yaml) |

For calculating AP (IoU without attribute prediction or IoU + F1 with attribute prediction), please refer to the [[Fahionpedia API](https://github.com/KMnP/fashionpedia-api)].

## Citation

```make
@inproceedings{jia2020fashionpedia,
  title={Fashionpedia: Ontology, Segmentation, and an Attribute Localization Dataset},
  author={Jia, Menglin and Shi, Mengyun and Sirotenko, Mikhail and Cui, Yin and Cardie, Claire and Hariharan, Bharath and Adam, Hartwig and Belongie, Serge},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
