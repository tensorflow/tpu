# Cloud TPU Port of the Resnet50 model
This is a straightforward port of the Resnet-50 model. The code was based on the original version from the tensorflow/models repository.

The only adjustments have been to add the required code to enable using the TPUEstimator interface, along with the data processing pipeline for ImageNet.

### Running the model
Assuming you have a version of ImageNet converted to the tfrecord format located at gs://my-cloud-bucket/data/imagenet/, you can run this model with the following command:

python resnet_main.py\
  --master=$TPU_WORKER \
  --data_dir=gs://my-cloud-bucket/data/imagenet \
  --model_dir=gs://my-cloud-bucket/models/resnet/v0 \
  --train_batch_size=1024 \
  --eval_batch_size=128 \

You can create the ImageNet dataset in the correct format using this [script](https://github.com/tensorflow/tpu-demos/blob/master/cloud_tpu/datasets/imagenet_to_gcs.py).

### Running the model with Fake Data
If you do not have ImageNet dataset prepared, you can use a randomly generated fake dataset to test the model. It is located at `gs://cloud-tpu-test-datasets/fake_imagenet`. You can pass this path as your `data_dir` to the `resnet_main.py`.
