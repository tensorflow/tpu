# `ctpu` quickstart #

## Introduction ##

This Google Cloud Shell tutorial walks through how to use the open source
[`ctpu`](https://github.com/tensorflow/tpu/tree/master/tools/ctpu) tool. We
will:

1. Download the latest `ctpu` release.
1. Confirm the configuration of `ctpu` through a few basic commands.
1. Launch a Cloud TPU "flock" (a GCE VM and Cloud TPU pair).
1. Create a [GCS](https://cloud.google.com/storage/) bucket for our training
   data.
1. Download the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database)
   and prepare it for use with a Cloud TPU.
1. Train a simple convolutional neural network on the MNIST dataset to recognize
   handwritten digits.
1. Begin training a modern convolutional neural network ([ResNet-50](https://github.com/tensorflow/tpu/tree/master/models/official/resnet))
   on a simulated dataset.
1. View performance and other metrics using [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard).
1. Clean everything up!

## Download `ctpu` ##

`ctpu` is available from <https://dl.google.com/cloud_tpu/ctpu/latest>. Execute
the following commands to get your copy:

```bash
wget https://dl.google.com/cloud_tpu/ctpu/latest/linux/ctpu && chmod a+x ctpu
```

After these commands complete, you should be able to execute the following command successfully:

```bash
./ctpu version
```

You can see all available subcommands by running:

```bash
./ctpu
```

## `ctpu` Configuration ##

`ctpu` is integrated with the Google Cloud Shell environment and should
automatically determine your username, and your project. You can view this
configuration information by executing:

```bash
./ctpu config
```

If you would like to override the inferred configuration for one command
invocation, you may do so on the command line:

```bash
./ctpu --name ilovetpus --project thisprojectdoesntexist config
```

Before we go any further, let's verify that we will be able to launch a Cloud
TPU in the next step. Execute:

```bash
./ctpu quota
```

and navigate to the resulting URL. You should ensure that you can allocate at
least one Cloud TPU (measured in increments of 8 cores).

If you do not have available quota, please request quota at <https://goo.gl/TODO>.

## Launch your Cloud TPU Flock ##

Launch your Cloud TPU flock by executing:

```bash
./ctpu up
```

This subcommand may take a few minutes to run. On your behalf, `ctpu` will:

1. Enable the GCE service (if required).
1. Enable the TPU service (if required).
1. Create a GCE VM with the latest stable TensorFlow version pre-installed.
1. Create a Cloud TPU with the corresponding version of TensorFlow.
1. Ensure your Cloud TPU has access to resource it needs from your project.
1. Perform a number of other checks.
1. Log you in to your new GCE VM.

While the `./ctpu up` command is running, let's prepare oen additional resource: GCS.

Navigate to <https://console.cloud.google.com/storage/browser> and create a new bucket. Pick a unique name,
select the *Regional* default storage class, and select `us-central1` as the region location.

Be sure to remember the name, as we'll need it in the next steps! Substitute the bucket name whenever you see `$GCS_BUCKET_NAME`.

After you have created your GCS bucket and the `./ctpu up` command has finished executing, click "Continue" to train your first model on a Cloud TPU!

## Recognizing handwritten digits using a Cloud TPU ##

You should now be logged into your GCE VM. Verify TensorFlow is installed by executing:

```bash
python -c "import tensorflow; print(tensorflow.__version__)"
```

You should see a version number printed (e.g. `1.6.0`).

> TODO(saeta): Insert the `convert_to_records.py` preprocessing instructions.

Upload the preprocessed records to your GCS bucket (be sure to substitute in the name of the bucket you created in the last step!):

```bash
gsutil cp -r mnist_data gs://$GCS_BUCKET_NAME/mnist
```

Now that you have your data prepared, you're ready to train. Execute:

```bash
python /usr/share/tpu/models/official/mnist/mnist_tpu.py \
  --train_file=gs://$GCS_BUCKET_NAME/mnist/data/train.tfrecords \
  --model_dir=gs://$GCS_BUCKET_NAME/mnist/model \
  --tpu_name=$USER
```

### What's happening? ###

This [python script](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_tpu.py) creates a
[`TPUEstimator`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/tpu/TPUEstimator) and then invokes `estimator.train(...)`.

`TPUEstimator` connects to the Cloud TPU, initializes the device, and begins training the model on the TFRecords stored in GCS.


Congratulations! You have now successfully trained a model on a Cloud TPU. Next, let's run a bigger model and try out TensorBoard.

## ResNet-50 on a Cloud TPU ##

[ResNet-50](https://github.com/tensorflow/tpu/tree/master/models/official/resnet) (published in [Dec 2015](https://arxiv.org/abs/1512.03385))
is a popular image classification model, and is one of the
[officially supported models](https://github.com/tensorflow/tpu/tree/master/models/official) on Cloud TPUs.

> Note: we will train on a fake dataset composed of random tensors available at `gs://cloud-tpu-test-datasets/fake_imagenet`. If you would like to train on
the true ImageNet data, follow the [instructions to download and preprocess the ImageNet data](https://cloud.google.com/tpu/docs/tutorials/resnet#download_and_convert_the_imagenet_data),
and be sure to substitute in the bucket where you've uploaded the preprocessed files instead of `gs://cloud-tpu-test-datasets/fake_imagenet` in the commands below.

### Start TensorBoard ###

Before training the model (which will take a while), first start TensorBoard in the background so you can visualize your training program's progress.

```bash
tensorboard -logdir gs://$GCS_BUCKET_NAME/resnet &
```

`ctpu` automatically set up special port forwarding for the Cloud Shell environment to make TensorBoard available.
All you need to do is click on `walkthrough spotlight-pointer devshell-web-preview-button "Web Preview"`, and select port `8080`.

### Start Training ###

The [ResNet](https://github.com/tensorflow/tpu/tree/master/models/official/resnet) model is pre-loaded on your GCE VM. To start training ResNet-50, execute:

```bash
python /usr/share/tpu/models/official/resnet/resnet_main.py \
  --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet \
  --model_dir=gs://$GCS_BUCKET_NAME/resnet \
  --tpu_name=$USER
```

`resnet_main.py` alternates between training on the labeled data for a few epochs, and performing a full pass over the evaluation dataset. While the loss and accuracy won't improve when
training on the fake dataset, ResNet-50 on the ImageNet dataset should achieve > 76% top-1 accuracy on the validation dataset in 90 epochs.

Be sure to flip back to TensorBoard to watch metrics about your training run.

## Clean up ##

To clean up, open a new Cloud shell window, and execute `./ctpu delete`. This will terminate and delete your GCE VM and your Cloud TPU. Then, go to GCS and delete your bucket.

## Congratulations ##

`walkthrough conclusion-trophy`

You've successfully training a modern image classification model using a Cloud TPU.

To learn more, head over to the [Cloud TPU docs](https://cloud.google.com/tpu/docs/how-to). Check out the [Cloud TPU Tools](https://cloud.google.com/tpu/docs/cloud-tpu-tools) to visualize and debug performance, or check to see if your model is TPU-compatible.

