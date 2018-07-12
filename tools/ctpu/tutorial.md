# ctpu quickstart #

## Introduction ##

This Google Cloud Shell tutorial walks through how to use the open source
[`ctpu`](https://github.com/tensorflow/tpu/tree/master/tools/ctpu) tool to train
an image classification model on a Cloud TPU. In this tutorial, you will:

1. Confirm the configuration of `ctpu` through a few basic commands.
1. Launch a Cloud TPU "flock" (a Compute Engine VM and Cloud TPU pair).
1. Create a [Cloud Storage](https://cloud.google.com/storage/) bucket for your
   training data.
1. Download the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database)
   and prepare it for use with a Cloud TPU.
1. Train a simple convolutional neural network on the MNIST dataset to recognize
   handwritten digits.
1. Begin training a modern convolutional neural network ([ResNet-50](https://github.com/tensorflow/tpu/tree/master/models/official/resnet))
   on a simulated dataset.
1. View performance and other metrics using [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard).
1. Clean everything up!

Before you get started, be sure you have created a GCP Project with
[billing enabled](https://cloud.google.com/billing/docs/how-to/modify-project#enable_billing_for_a_project).
When you have the [project ID](https://support.google.com/cloud/answer/6158840)
in hand (the "short name" found on the cloud console's main landing page), click
"Continue" to get started!

## Setup ##

`ctpu` is pre-installed on your Google Cloud Shell. If you have previously
downloaded `ctpu`, please delete it (`rm -f ~/ctpu`) to ensure you're using the
most up-to-date version.

### Configure Cloud Shell ###

`ctpu` is integrated with the Google Cloud Shell environment and should
automatically determine your username.

When you [launch Cloud Shell within the context of a project](https://cloud.google.com/shell/docs/starting-cloud-shell)
(e.g. from the Cloud console dashboard), `ctpu` automatically determines the
project.

However, Cloud Shell tutorials are not created within the context of a project.
Therefore, for this tutorial you need to set the project environment variable
with your [GCP Project ID](https://support.google.com/cloud/answer/6158840).

```bash
export DEVSHELL_PROJECT_ID=<fill_in_your_project_id>
```

You can view the configuration inferred by `ctpu` by executing:

```bash
ctpu print-config
```

### Test your installation ###

You can see all available subcommands by running:

```bash
ctpu
```

You should see a list of commands and a brief description of each one.

Click "Continue" to launch your resources.

## Create resources ##

It's now time to create your GCP resources.

### Launch your flock ###

Launch your Cloud TPU flock by executing:

```bash
ctpu up
```

This subcommand may take a few minutes to run. On your behalf, `ctpu` will:

1. Enable the Compute Engine and Cloud TPU service (if necessary).
1. Create a Compute Engine VM with the latest stable TensorFlow version
   pre-installed.
1. Create a Cloud TPU with the corresponding version of TensorFlow.
1. Ensure your Cloud TPU has access to resources it needs from your project.
1. Perform a number of other checks.
1. Log you in to your new Compute Engine VM.

> Note: the first time you run `ctpu up` on a project, it takes longer than
> normal, including ssh key propagation and API turn-up. Later invocations
> should be much faster.

### Create your Cloud Storage Bucket ###

While the `ctpu up` command is running, prepare one additional resource:
[Cloud Storage](https://cloud.google.com/storage/).

Navigate to <https://console.cloud.google.com/storage/browser> and create a new
bucket. Pick a unique name, select the *Regional* default storage class, and
select `us-central1` as the region location.

Be sure to remember the name, as you'll need it in the next steps!

### Verify your Compute Engine resources ###

When the `ctpu up` command has finished executing, you should now be logged into
your Compute Engine VM. (Your shell prompt should change from `username@project`
to `username@username`.) Verify TensorFlow is installed by executing:

```bash
python -c "import tensorflow; print(tensorflow.__version__)"
```

You should see a version number printed (e.g. `1.8.0`).

### Set environment variables ###

To make it easier to run subsequent commands, set an environment variable with
the name of the Cloud Storage bucket you just created.

```bash
export GCS_BUCKET_NAME=<fill_me_in>
```

After you have configured your Cloud Storage bucket, click "Continue" to train
your first model on a Cloud TPU.

## Recognizing handwritten digits using a Cloud TPU ##

### Prepare the data ###

Run the following [script](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/how_tos/reading_data/convert_to_records.py)
to download and preprocess the
[images](http://yann.lecun.com/exdb/mnist/index.html):

```bash
python /usr/share/tensorflow/tensorflow/examples/how_tos/reading_data/convert_to_records.py --directory=./data && \
  gunzip ./data/*.gz
```

Upload the preprocessed records to your Cloud Storage bucket (the environment
variable you set in the last step will be automatically substituted so you can
copy-paste the following commands unmodified):

```bash
gsutil cp -r ./data gs://$GCS_BUCKET_NAME/mnist/data
```

### Train your model ###

Now that you have your data prepared, you're ready to train. Execute:

```bash
python /usr/share/models/official/mnist/mnist_tpu.py \
  --data_dir=gs://$GCS_BUCKET_NAME/mnist/data/ \
  --model_dir=gs://$GCS_BUCKET_NAME/mnist/model \
  --tpu=$TPU_NAME
```

### What's happening? ###

This [Python script](https://github.com/tensorflow/models/blob/master/official/mnist/mnist_tpu.py)
creates a
[`TPUEstimator`](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/tpu/TPUEstimator)
and then invokes `estimator.train(...)`.

`TPUEstimator` connects to the Cloud TPU, initializes the device, and begins
training the model on the TFRecords stored in Cloud Storage.

Congratulations! You have now successfully trained a model on a Cloud TPU. Next,
you can run a bigger model and try out TensorBoard.

## ResNet-50 on a Cloud TPU ##

[ResNet-50](https://github.com/tensorflow/tpu/tree/master/models/official/resnet)
(published in [Dec 2015](https://arxiv.org/abs/1512.03385)) is a popular image
classification model, and is one of the
[officially supported models](https://github.com/tensorflow/tpu/tree/master/models/official)
on Cloud TPUs.

> Note: This tutorial shows you how to train on a fake dataset composed of
> random tensors available at
> `gs://cloud-tpu-test-datasets/fake_imagenet`. If you would like to train on
> the true ImageNet data, follow the [instructions to download and preprocess
> the ImageNet data](https://cloud.google.com/tpu/docs/tutorials/resnet#full-imagenet),
> and be sure to substitute in the bucket where you've uploaded the preprocessed
> files instead of `gs://cloud-tpu-test-datasets/fake_imagenet` in the commands
> below.

### Start TensorBoard ###

Before training the model (which takes hours to complete), start
TensorBoard in the background so you can visualize your training program's
progress.

```bash
tensorboard -logdir gs://$GCS_BUCKET_NAME/resnet &
```

`ctpu` automatically set up special port forwarding for the Cloud Shell
environment to make TensorBoard available.

All you need to do is click on the Web Preview button
(<walkthrough-web-preview-icon></walkthrough-web-preview-icon> -
<walkthrough-spotlight-pointer spotlightId="devshell-web-preview-button">
  click me to highlight it
</walkthrough-spotlight-pointer>),
and open port `8080`.

> Note: because you haven't started training yet, tensorboard should be empty.

### Start Training ###

The [ResNet](https://github.com/tensorflow/tpu/tree/master/models/official/resnet)
model is pre-loaded on your Compute Engine VM.

To start training ResNet-50, execute:

```bash
python /usr/share/tpu/models/official/resnet/resnet_main.py \
  --data_dir=gs://cloud-tpu-test-datasets/fake_imagenet \
  --model_dir=gs://$GCS_BUCKET_NAME/resnet \
  --tpu=$TPU_NAME
```

`resnet_main.py` will connect to your Cloud TPU, initialize the device, and
train a ResNet-50 model on the provided data. Checkpoints will be regularly
saved to `gs://$GCS_BUCKET_NAME/resnet`.

While the loss and accuracy won't improve when training on the fake dataset,
ResNet-50 on the ImageNet dataset should achieve > 76% top-1 accuracy on the
validation dataset in 90 epochs.

Be sure to flip back to TensorBoard to watch metrics about your training run.

You can cancel training at any time by hitting `ctrl+c` or deleting your Cloud
TPU and/or Compute Engine VM. Checkpoints are saved in your Cloud Storage
bucket. To resume training from
the latest checkpoint, just re-run the `python` command from above passing in
the same `--model_dir` value.

## Clean up ##

To clean up, stop training and sign out of your Compute Engine VM (use `exit`).
Then, in your Cloud Shell (your prompt should be `user@projectname`) execute
`ctpu delete`. This will delete your Compute Engine VM and your Cloud TPU.
Then, go to Cloud Storage and delete your bucket (if you don't need it any
more).

You can run `ctpu status` to make sure you have no instances allocated, although
note that deletion may take a minute or two.

## Congratulations ##

<walkthrough-conclusion-trophy></walkthrough-conclusion-trophy>

You've successfully started training a modern image classification model using a
Cloud TPU.

To learn more, head over to the [Cloud TPU docs](https://cloud.google.com/tpu/docs/how-to).
Check out the [Cloud TPU Tools](https://cloud.google.com/tpu/docs/cloud-tpu-tools)
to visualize and debug performance, or check to see if your model is
TPU-compatible.

You can refer back to this tutorial to see all the commands by opening it on
[GitHub](https://github.com/tensorflow/tpu/blob/master/tools/ctpu/tutorial.md)
or by executing in your Google Cloud shell:

```bash
teachme ~/tpu/tools/ctpu/tutorial.md
```

All the code used in this tutorial is open source. Check out the
[TPU](https://github.com/tensorflow/tpu),
[TensorFlow](https://github.com/tensorflow/tensorflow), and
[models](https://github.com/tensorflow/models) repositories for
pre-processing scripts, and additional sample models.

Finally, below is a "cheat sheet" for using Cloud TPUs:

 - `ctpu status`: Prints the current status of your Compute Engine VM and Cloud
   TPU.
 - `ctpu up`: Gets everything ready and logs in to your VM. You can run this
   multiple times.
 - `ctpu pause`: Turns off your Compute Engine VM and Cloud TPU. Software
   installed, data saved on your Compute Engine VM disk, and data in Cloud
   Storage will persist.
 - `ctpu delete`: Cleans up all Compute Engine resources (VM & Cloud TPU). Any
   software installation or configuration and all data on your Compute Engine VM
   will be deleted. Data stored in Cloud Storage will persist.
 - `ctpu`: Displays a summary of all available commands.
 - `/usr/share/tpu`, `/usr/share/tensorflow`, `/usr/share/models`: Contain
   copies of scripts and tools for use with Cloud TPUs.
