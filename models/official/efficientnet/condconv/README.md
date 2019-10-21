# EfficientNet-CondConv

[1] Brandon Yang, Gabriel Bender, Quoc V. Le, Jiquan Ngiam. CondConv: Conditionally Parameterized Convolutions for Efficient Inference. NeurIPS 2019. Arxiv Link: https://arxiv.org/abs/1904.04971.

## 1. About CondConv

Conditionally parameterized convolutions (CondConv) are a new building block for convolutional neural networks to increase capacity while maintaining efficient inference. In a traditional convolutional layer, each example is processed with the same kernel. In a CondConv layer, each example is processed with a specialized, example-dependent kernel. As an intuitive motivating example, on the ImageNet classification dataset, we might want to classify dogs and cats with different convolutional kernels.

<table border="0" width="50%">
<tr>
    <td>
    <img src="../g3doc/condconv-layer.png"/>
    </td>
</tr>
</table>

A CondConv layer consists of n experts, each of which are the same size as the convolutional kernel of the original convolutional layer. For each example, the example-dependent convolutional kernel is computed as the weighted sum of experts using an example-dependent routing function. Increasing the number of experts enables us to increase the capacity of a network, while maintaining efficient inference.

Replacing convolutional layers with CondConv layers improves the accuracy versus inference cost trade-off on a wide range of models: MobileNetV1, MobileNetV2, ResNets, and EfficientNets. We measure inference cost in multiply-adds (MADDs). When applied to EfficientNets, we obtain EfficientNet-CondConv models. Our EfficientNet-CondConv-B0 model with 8 experts achieves state-of-the-art accuracy versus inference cost performance.

In this directory, we open-source the code to reproduce the EfficientNet-CondConv results in our paper and enable easy experimentation with EfficientNet-CondConv models. Additionally, we open-source the CondConv2d and DepthwiseCondConv2D Keras layers for easy application in new model architectures.

## 2. Using pretrained EfficientNet-CondConv checkpoints

We have provided pre-trained checkpoints for several EfficientNet-CondConv models.

|                                | CondConv Experts | Params | MADDs | Accuracy |
|--------------------------------|------------------|--------|-------|----------|
| EfficientNet-B0                | -                | 5.3M   | 391M  | 77.3     |
| EfficientNet-CondConv-B0 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/condconv/efficientnet-condconv-b0-4e.tar.gz))| 4                | 13.3M  | 402M  | 77.8     |
| EfficientNet-CondConv-B0 ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/condconv/efficientnet-condconv-b0-8e.tar.gz))| 8                | 24.0M  | 413M  | 78.3     |

|                                       | CondConv Experts | Params | MADDs | Accuracy |
|---------------------------------------|------------------|--------|-------|----------|
| EfficientNet-B1                       | -                | 7.8M   | 700M  | 79.2     |
| EfficientNet-CondConv-B0-Depth ([ckpt](https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/condconv/efficientnet-condconv-b0-8e-depth.tar.gz)) | 8                | 39.7M  | 614M  | 79.5     |

A quick way to use these checkpoints is to run:

```shell
$ export MODEL=efficientnet-condconv-b0-8e
$ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/${MODEL}.tar.gz
$ tar zxf ${MODEL}.tar.gz
$ wget https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG -O panda.jpg
$ wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/eval_data/labels_map.txt
$ python eval_ckpt_main.py --model_name=$MODEL --ckpt_dir=$MODEL --example_img=panda.jpg --labels_map_file=labels_map.txt
```

Please refer to the following colab for more instructions on how to obtain and use those checkpoints.

  * [`eval_ckpt_example.ipynb`](eval_ckpt_example.ipynb): A colab example to load
 EfficientNet pretrained checkpoints files and use the restored model to classify images.

## 3. Training EfficientNet-CondConv models on Cloud TPUs
Please refer to our tutorial: https://cloud.google.com/tpu/docs/tutorials/efficientnet.

