# Cloud TPU reference model for DCGAN



## Running the model

Assuming you have a version of CIFAR or MNIST converted to the tfrecord format located
at `gs://my-cloud-bucket/data/`, you can run these models with the 
following commands:

For CIFAR
```
python dcgan_main.py\ 
  --dataset=cifar\
  --cifar_train_data_file=gs://my-cloud-bucket/data/cifar/train.tfrecord\
  --cifar_test_data_file=gs://my-cloud-bucket/data/cifar/test.tfrecords\
  --model_dir=gs://my-cloud-bucket/dcgan/cifar\

```

For MNIST
```
python dcgan_main.py\ 
  --dataset=mnist\
  --mnist_train_data_file=gs://my-cloud-bucket/data/mnist/train.tfrecord\
  --mnist_test_data_file=gs://my-cloud-bucket/data/mnist/test.tfrecords\
  --model_dir=gs://my-cloud-bucket/dcgan/mnist\
```


## Getting the data

In case you don't yet have the data available in a bucket, follow the intructions
below to download and convert CIFAR or MNIST datasets into tfrecord format.

First, follow the instructions [here](https://cloud.google.com/storage/docs/creating-buckets) to create GCS buckets.

For CIFAR

This [script](https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py)
both downloads and converts CIFAR data into tfrecord format.


For MNIST

This [script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py)
converts MNIST data into *tfrecord* format.