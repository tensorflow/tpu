# Training RetinaNet on Cloud TPU

This folder contains an implementation of the
[RetinaNet](https://arxiv.org/pdf/1708.02002.pdf) object detection model.

The instructions below assume you are already familiar with running a model on
the TPU.  If you haven't already, please review the [instructions for running
the ResNet model on the Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/resnet).

## Check for the RetinaNet model

If you are running on the prepared TPU image, the RetinaNet model files should
be pre-installed:

```
ls /usr/share/tpu/models/official/retinanet/
```

If they are not available, you can find the latest version on GitHub:

```
git clone https://github.com/tensorflow/tpu/
ls tpu/models/official/retinanet
```

## Before we start

### Setting up our TPU VM

The commands below assume you have started a TPU VM and set its name in an
environment variable:

```
gcloud beta compute tpus list
export TPU_NAME=my-tpu-name

```
See the [quickstart documentation](https://cloud.google.com/tpu/docs/quickstart)
for how to start a TPU VM.

### GCS bucket for model checkpoints and training data

We will also need a bucket to store out data and model files.  We'll specify
that with the `${GCS_BUCKET}` variable.

```
GCS_BUCKET=gs://my-ml-bucket
```

You can create a bucket using the
[web interface](https://cloud.google.com/storage/docs/creating-buckets) or on
the command line with gsutil:

```
gsutil mb ${GCS_BUCKET}
```

## Preparing the COCO dataset

Before we can train, we need to prepare our training data.  The RetinaNet 
model here has been configured to train on the COCO dataset.

The `tpu/tools/datasets/download_and_preprocess_coco.sh` script will convert 
the COCO dataset into a set of TFRecords that our trainer expects.

This requires at least 100GB of disk space for the target directory, and will
take approximately 1 hour to complete.  If you don't have this amount of space
on your VM, you will need to attach a data drive to your VM.  See the
[add persistent disk](https://cloud.google.com/compute/docs/disks/add-persistent-disk)
instructions for details on how to do this.

Once you have a data directory available, you can run the preprocessing script:

```
cd tpu/tools/datasets
bash download_and_preprocess_coco.sh ./data/dir/coco
```

This will install the required libraries and then run the preprocessing script.
It outputs a number of `*.tfrecord` files in your data directory.  The script 
may take up to an hour to run; you might want to grab a coffee while it's 
going.

We now need to copy these files to GCS so they are accessible to our TPU for
training.  We can use `gsutil` to copy the files over.  We also want to save the
annotation files: we use these to validate our model performance:

```
gsutil -m cp ./data/dir/coco/*.tfrecord ${GCS_BUCKET}/coco
gsutil cp ./data/dir/coco/raw-data/annotations/*.json ${GCS_BUCKET}/coco
```

## Installing extra packages

The RetinaNet trainer requires a few extra packages.  We can install them now:

```
sudo apt-get install -y python-tk
pip install Cython matplotlib
pip install 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
```

## Running the trainer

We're ready to run our trainer.  Let's first try running it for 100 steps to
make sure everything is working and we can write out checkpoints successfully:

```
RESNET_CHECKPOINT=gs://cloud-tpu-artifacts/resnet/resnet-nhwc-2018-02-07/model.ckpt-112603
MODEL_DIR=${GCS_BUCKET}/retinanet-model

python tpu/models/official/retinanet/retinanet_main.py \
 --tpu=${TPU_NAME} \
 --train_batch_size=64 \
 --training_file_pattern=${GCS_BUCKET}/coco/train-* \
 --resnet_checkpoint=${RESNET_CHECKPOINT} \
 --model_dir=${MODEL_DIR} \
 --hparams=image_size=640 \
 --num_examples_per_epoch=6400 \
 --num_epochs=1
```

Note the `--resnet_checkpoint` flag: RetinaNet requires a pre-trained image
classification model (like ResNet) as a _backbone network_.  We have provided
a pretrained checkpoint using the `resnet` demonstration model.  You can instead
train your own `resnet` model if desired: simply specify a checkpoint from your
`resnet` model directory.

## Evaluating a model while we train (optional)

We often want to measure the progress of our model on a validation set as it
trains.  As our evaluation code for RetinaNet does not currently run on the
TPU VM, we need to run it on a CPU or GPU machine.  Running through all of the
validation images is time-consuming, so we don't want to stop our training to
let it run.  Instead, we can run our validation in parallel on a different VM.
Our validation runner will scan our model directory for new checkpoints, and when
it finds one, will compute new evaluation metrics.

Let's start a VM for running the evalution.  We recommend using a GPU VM so
evaluations run quickly.  This requires a bit more setup:

### GPU Evaluation VM

Start the VM:

```
gcloud compute instances create eval-vm  \
 --machine-type=n1-highcpu-16  \
 --image-project=ubuntu-os-cloud  \
 --image-family=ubuntu-1604-lts  \
 --scopes=cloud-platform \
 --accelerator type=nvidia-tesla-p100 \
 --maintenance-policy TERMINATE \
 --restart-on-failure
```

After a minute, we should be able to connect:

`gcloud compute ssh eval-vm`

We need to setup CUDA so Tensorflow can use our image.  The following commands,
run on the evaluation VM, will install CUDA and Tensorflow on our GPU VM. After 
the installation finishes, we recommend you restart the VM.

```
cat > /tmp/setup.sh <<HERE
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb

dpkg -i ./cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt-get update
apt-get install -y cuda-9-0
bash -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list'
apt-get update
apt-get install -y --no-install-recommends libcudnn7=7.0.5.15-1+cuda9.0
apt install -y python-pip python-tk
pip install tensorflow-gpu==1.8
HERE

sudo bash /tmp/setup.sh
```

### CPU Evaluation VM (not recommended)

You can also use a CPU VM for evalution which requires a bit less setup, but
is significantly slower:

```
gcloud compute instances create\
  retinanet-eval-vm\
  --machine-type=n1-highcpu-64\
  --image-project=ml-images\
  --image-family=tf-1-6\
  --scopes=cloud-platform
```

We can now connect to the evaluation VM and start the evaluation loop.

### Installing packages and checking the RetinaNet Model

On either VM type, as before, we'll need to install our packages:

```
sudo apt-get install -y python-tk
pip install Cython matplotlib
pip install 'git+https://github.com/pdollar/coco.git#egg=pycocotools&subdirectory=PythonAPI'
```

We then need to grab the Retinanet model code so we can evaluate:

```
git clone https://github.com/tensorflow/tpu
```

### Running evaluation

We can now run the evaluation script.  Let's first try a quick evaluation to
test that we can read our model directory and validation files.

```
# export GCS_BUCKET as above

# Copy over the annotation file we created during preprocessing
gsutil cp ${GCS_BUCKET}/coco/instances_val2017.json .

python tpu/models/official/retinanet/retinanet_main.py  \
 --use_tpu=False \
 --validation_file_pattern=${GCS_BUCKET}/coco/val-* \
 --val_json_file=./instances_val2017.json \
 --model_dir=${GCS_BUCKET}/retinanet-model/ \
 --hparams=image_size=640 \
 --mode=eval \
 --num_epochs=1 \
 --num_examples_per_epoch=100 \
 --eval_steps=10
```

We specified `num_epochs=1` and `eval_steps=10` above to ensure our script
finished quickly.  We'll change those now to run over the full evaluation
dataset:

```
python tpu/models/official/retinanet/retinanet_main.py  \
 --use_tpu=False \
 --validation_file_pattern=${GCS_BUCKET}/coco/val-* \
 --val_json_file=./instances_val2017.json \
 --model_dir=${GCS_BUCKET}/retinanet-model/ \
 --hparams=image_size=640 \
 --num_epochs=15 \
 --mode=eval \
 --eval_steps=5000
```

It takes about 10 minutes to run through the 5000 evaluation steps.  After
finishing, the evaluator will continue waiting for new checkpoints from the
trainer for up to 1 hour.  We don't have to wait for the evaluation to finish
though: we can go ahead and kick off our full training run now.

## Running the trainer (again)

Back on our original VM, we're now ready to run our model on our preprocessed
COCO data.  Complete training takes less than 4 hours.

```
python tpu/models/official/retinanet/retinanet_main.py \
 --tpu=${TPU_NAME} \
 --train_batch_size=64 \
 --training_file_pattern=${GCS_BUCKET}/coco/train-* \
 --resnet_checkpoint=${RESNET_CHECKPOINT} \
 --model_dir=${GCS_BUCKET}/retinanet-model/ \
 --hparams=image_size=640 \
 --num_epochs=15
```

### Checking the status of our training

[Tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard)
lets us visualize the progress of our training.

If you setup an evaluation VM, it will continually read new checkpoints and 
output the  evaluation events to the `model_dir` directory.  You can view
the current status of the training and evaluation in Tensorboard:

```
tensorboard --logdir=${MODEL_DIR}
```

You will need to run this from your local desktop, [setup port forwarding](https://cloud.google.com/tpu/docs/tutorials/mnist)
to your VM to access the server.

## Where to go from here

### Training with Different Image Sizes

The instructions in this tutorial assume we want to train on a 640x640 pixel
image.  You can try changing the `image_size` hparam to train on a smaller
image, resulting in a faster but less precise model.

In addition, you can explore using a larger backbone network (e.g., ResNet-101
instead of ResNet-50). A larger input image and a more powerful backbone will
yield a slower but more precise model. You can specify the `image_size`
hparam to be 768, 896, or 1024; also, the `resnet_depth` parameter can be one of
50 or 101 (see sections below). When training the model with larger image size,
the model may run OOM on the TPU device; one way to address the issue is to use
_bfloat16_ by setting the `use_bfloat16=True`
hparam. With `image_size=896,resnet_depth=101`, the model is able to reach 37.7
AP.


### Different Basis

Alternatively, you can explore pre-training a Resnet model on your own dataset
and using it as a basis for your RetinaNet model.  With some more work, you can
also swap in an alternate _backbone_ network in place of ResNet.  Finally, if
you are interested in implementing your own object detection models, this
network may be a good basis for further experimentation.


### Larger Batch size on TPUv2-32

By using a TPU pod, you can reduce the training time by using a larger batch 
size. To train on a TPUv2-32, you need to change the batch size accordingly
(e.g., 64 on TPUv2-8, 256 on TPUv2-32). The model will linearly scale the
learning rate given the batch size, see `retinanet_model.py` for more details.

```
python tpu/models/official/retinanet/retinanet_main.py \
 --tpu=${TPU_NAME} \
 --train_batch_size=256 \
 --num_cores=32 \
 --training_file_pattern=${GCS_BUCKET}/coco/train-* \
 --resnet_checkpoint=${RESNET_CHECKPOINT} \
 --model_dir=${GCS_BUCKET}/retinanet-model/ \
 --hparams=image_size=640 \
 --num_epochs=15
```
