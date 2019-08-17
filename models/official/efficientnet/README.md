# EfficientNets

[1] Mingxing Tan and Quoc V. Le.  EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML 2019.
   Arxiv link: https://arxiv.org/abs/1905.11946.

Updates

  - **[Aug 2019] Released EfficientNet-EdgeTPU models: [README](edgetpu/README.md) and [blog post](https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html).**

  - [Jul 2019] Released new EfficientNet checkpoints trained with AutoAugment (better accuracy than paper).

  - [May 2019] Released EfficientNets code and weights: [blog post](https://ai.googleblog.com/2019/05/efficientnet-improving-accuracy-and.html)

## 1. About EfficientNet Models

EfficientNets are a family of image classification models, which achieve state-of-the-art accuracy, yet being an order-of-magnitude smaller and faster than previous models.

We develop EfficientNets based on AutoML and Compound Scaling. In particular, we first use [AutoML MNAS Mobile framework](https://ai.googleblog.com/2018/08/mnasnet-towards-automating-design-of.html) to develop a mobile-size baseline network, named as EfficientNet-B0; Then, we use the compound scaling method to scale up this baseline to obtain EfficientNet-B1 to B7.

<table border="0">
<tr>
    <td>
    <img src="./g3doc/params.png" width="100%" />
    </td>
    <td>
    <img src="./g3doc/flops.png", width="90%" />
    </td>
</tr>
</table>

EfficientNets achieve state-of-the-art accuracy on ImageNet with an order of magnitude better efficiency:


* In high-accuracy regime, our EfficientNet-B7 achieves state-of-the-art 84.4% top-1 / 97.1% top-5 accuracy on ImageNet with 66M parameters and 37B FLOPS, being 8.4x smaller and 6.1x faster on CPU inference than previous best [Gpipe](https://arxiv.org/abs/1811.06965).

* In middle-accuracy regime, our EfficientNet-B1 is 7.6x smaller and 5.7x faster on CPU inference than [ResNet-152](https://arxiv.org/abs/1512.03385), with similar ImageNet accuracy.

* Compared with the widely used [ResNet-50](https://arxiv.org/abs/1512.03385), our EfficientNet-B4 improves the top-1 accuracy from 76.3% of ResNet-50 to 82.6% (+6.3%), under similar FLOPS constraint.

## 2. Using Pretrained EfficientNet Checkpoints

We have provided a list of EfficientNet checkpoints for EfficientNet checkpoints: both w/ and w/o AutoAugment.

  * With standard ResNet preprocessing, we achieved similar accuracy as the original ICML paper.
  * WIth [AutoAugment](https://arxiv.org/abs/1805.09501) preprocessing, we achieved higher accuracy than the original ICML paper.

|               |   B0    |  B1   |  B2    |  B3   |  B4   |  B5    | B6 | B7 |
|----------     |--------  | ------| ------|------ |------ |------ | --- | --- |
| Standard preprocessing |  76.8% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b0.tar.gz))   | 78.8% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b1.tar.gz))  | 79.8% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b2.tar.gz)) | 81.0% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b3.tar.gz)) | 82.6% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b4.tar.gz)) | 83.2% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/efficientnet-b5.tar.gz)) | | |
| AutoAugment |  77.3% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b0.tar.gz))   | 79.2% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b1.tar.gz))  | 80.3% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b2.tar.gz)) | 81.7% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b3.tar.gz)) | 83.0% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b4.tar.gz)) | 83.7% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b5.tar.gz)) |  84.2% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b6.tar.gz)) | 84.5% ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-b7.tar.gz))  |

<!--
| Acc. from paper        |  76.3%   | 78.8% | 79.8% | 81.1% | 82.6% | 83.3% |
-->

<sup>**To train EfficientNets with AutoAugment ([code](https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py)), simply add option "--autoaugment_name=v0". If you use these checkpoints, you can cite this [paper](https://arxiv.org/abs/1805.09501).</sup>

A quick way to use these checkpoints is to run:

    $ export MODEL=efficientnet-b0
    $ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckpts/${MODEL}.tar.gz
    $ tar xf ${MODEL}.tar.gz
    $ wget https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG -O panda.jpg
    $ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt
    $ python eval_ckpt_main.py --model_name=$MODEL --ckpt_dir=$MODEL --example_img=panda.jpg --labels_map_file=labels_map.txt

Please refer to the following colab for more instructions on how to obtain and use those checkpoints.

  * [`eval_ckpt_example.ipynb`](eval_ckpt_example.ipynb): A colab example to load
 EfficientNet pretrained checkpoints files and use the restored model to classify images.


## 3. Using EfficientNet as Feature Extractor

```
    import efficientnet_builder
    features, endpoints = efficientnet_builder.build_model_base(images, 'efficientnet-b0')
```

  * Use `features` for classification finetuning.
  * Use `endpoints['reduction_i']` for detection/segmentation, as the last intermediate feature with reduction level `i`. For example, if input image has resolution 224x224, then:
    * `endpoints['reduction_1']` has resolution 112x112
    * `endpoints['reduction_2']` has resolution 56x56
    * `endpoints['reduction_3']` has resolution 28x28
    * `endpoints['reduction_4']` has resolution 14x14
    * `endpoints['reduction_5']` has resolution 7x7

## 4. Training EfficientNets on TPUs.


To train this model on Cloud TPU, you will need:

   * A GCE VM instance with an associated Cloud TPU resource
   * A GCS bucket to store your training checkpoints (the "model directory")
   * Install TensorFlow version >= 1.13 for both GCE VM and Cloud.

Then train the model:

    $ export PYTHONPATH="$PYTHONPATH:/path/to/models"
    $ python main.py --tpu=TPU_NAME --data_dir=DATA_DIR --model_dir=MODEL_DIR

    # TPU_NAME is the name of the TPU node, the same name that appears when you run gcloud compute tpus list, or ctpu ls.
    # MODEL_DIR is a GCS location (a URL starting with gs:// where both the GCE VM and the associated Cloud TPU have write access
    # DATA_DIR is a GCS location to which both the GCE VM and associated Cloud TPU have read access.


For more instructions, please refer to our tutorial: https://cloud.google.com/tpu/docs/tutorials/efficientnet
