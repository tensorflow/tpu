# Cloud TPU Port of DenseNet

This folder contains an implementation of the [DenseNet](https://arxiv.org/pdf/1608.06993.pdf)
image classification model.

## Running the model on ImageNet

The process for running on ImageNet is similar, just specify the directory
containing your converted tfrecord files:

```
python densenet_imagenet.py\
  --alsologtostderr\
  --num_shards=8\
  --batch_size=1024\
  --master=grpc://$TPU_WORKER:8470\
  --use_tpu=1\
  --model_dir=gs://my-cloud-bucket/models/densenet-imagenet/0\
  --data_dir=gs://my-cloud-bucket/data/imagenet
