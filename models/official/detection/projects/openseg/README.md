# OpenSeg: Scaling Open-Vocabulary Image Segmentation with Image-Level Labels


Golnaz Ghiasi, Xiuye Gu, Yin Cui, Tsung-Yi Lin
[[arXiv]](https://arxiv.org/abs/2112.12143)
[[demo]](https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/models/official/detection/projects/openseg/OpenSeg_demo.ipynb)
[[poster]](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/OpenSeg-ECCV22-poster.pdf)

OpenSeg can organize pixels into meaningful regions indicated by texts.
In contrast to segmentation models trained with close-vocabulary
categories, OpenSeg can handle arbitrary text queries.

<p style="text-align:center;">
<img src="https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/teaser.jpg" height="120" />
</p>


The figure below shows an overview of OpenSeg architecture.

<p style="text-align:center;">
<img src="https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/overview.jpg" height="400" />
</p>

## Colab Demo
Please try out our colab demo:

[colab](https://colab.sandbox.google.com/github/tensorflow/tpu/blob/master/models/official/detection/projects/openseg/OpenSeg_demo.ipynb)

[jupyter notebook](./OpenSeg_demo.ipynb)

The image tower of the OpenSeg model used in this colab has a backbone of Efficientnet-b7, initialized with the [noisy student checkpoint](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet#2-using-pretrained-efficientnet-checkpoints). The text tower is the frozen text tower of [CLIP ViT-L/14@336px](https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/clip.py#L39).
The model is trained on COCO class-agnostic masks, COCO captions, and localized narrative caption data.


## Class names w/wo ensembling and prompt engineering

We provide class names for ADE20k, COCO Panoptic, PASCAL Context and PASCAL VOC datasets used in OpenSeg. The details are described in Appendix I "Ensembling and prompt engineering" in our paper.

[ade20k_150](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/ade20k_150.txt)

[ade20k_150_with_prompt_eng](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/ade20k_150_with_prompt_eng.txt)

[ade20k_847](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/ade20k_847.txt)

[ade20k_847_with_prompt_eng](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/ade20k_847_with_prompt_eng.txt)

[coco_panoptic](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/coco_panoptic.txt)

[coco_panoptic_with_prompt_eng](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/coco_panoptic_with_prompt_eng.txt)

[pascal_context_459](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/pascal_context_459.txt)

[pascal_context_459_with_prompt_eng](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/pascal_context_459_with_prompt_eng.txt)

[pascal_context_59](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/pascal_context_59.txt)

[pascal_context_59_with_prompt_eng](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/pascal_context_59_with_prompt_eng.txt)

[pascal_voc](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/pascal_voc.txt)

[pascal_voc_with_prompt_eng](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/pascal_voc_with_prompt_eng.txt)

## Citation

```make
@inproceedings{ghiasi2021open,
  title={Scaling Open-Vocabulary Image Segmentation with Image-Level Labels},
  author={Ghiasi, Golnaz and Gu, Xiuye and Cui, Yin and Lin, Tsung-Yi},
  booktitle={ECCV},
  year={2022}
}
```
