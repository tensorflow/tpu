# Methodology for ResNet-50 v1.5 Performance Comparison on Cloud TPUs and Google Cloud GPUs

Frank Chen, Toby Boyd, Jing Li *(Google Brain)*

## Our approach to performance measurement

Great care is required to construct performance benchmarks that fairly and reproducibly compare machine learning (ML) training performance across an increasing variety of different hardware configurations and software frameworks.

For this initial performance comparison, we chose to focus on two top-of-the-line hardware accelerators that are currently available on Google Cloud: NVIDIA’s V100 GPU and Google’s Cloud TPU v2 Pod. We ran our analysis on Google Cloud Platform (GCP) and used well-optimized, open-source TensorFlow 1.12 implementations to collect all performance measurements. To maximize performance, we use [`bfloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) (a half-precision, 16-bit data type explicitly designed for ML) on the Cloud TPUs and use mixed-precision [`float16`](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) to maximize the utilization of Tensor Cores on the NVIDIA Tesla V100s.

The scale of the largest ML training runs has [increased rapidly](https://blog.openai.com/ai-and-compute/) over the past few years, and we expect this trend to continue; we also expect rapid continued improvements in accelerator performance and capabilities.

## Model architecture: ResNet-50 v1.5

We chose to focus on training the ResNet-50 image recognition model on the ImageNet dataset because it is well-known and has been well-optimized on many platforms. There are actually several different variants of the ResNet-50 architecture and training procedure that have vastly different computational profiles and achieve different trained accuracies. In this study, we choose a variant of ResNet-50 that we informally call “**ResNet-50 v1.5**.”

ResNet-50 v1.5 is almost the same model architecture described by He, et. al. in the original ResNet paper, “[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)” (arXiv:1512.03385v1). However, stride 2 is used in the first 3x3 convolution of each block instead of in the first 1x1 convolution. This variation can be found in the [code](https://github.com/facebook/fb.resnet.torch) corresponding to the paper "[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)" (arXiv:1706.02677v2). We use the same input size (224x224) as the original ResNet paper.

### Implementation Details

We use the [tf_cnn_benchmarks implementation](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks) of ResNet-50 v1.5 training for the GPU benchmark. This version of ResNet-50 utilizes [mixed-precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html#mptrain) FP16 to maximize the utilization of Tensor Cores on the NVIDIA Tesla V100. [XLA](https://www.tensorflow.org/xla/) was used to optimize the graph for GPU execution to further improve the performance of the V100 GPUs.

We use the [standard Cloud TPU reference model](https://github.com/tensorflow/tpu/tree/r1.12/models/official/resnet) implementation of ResNet-50 v1.5 for Cloud TPU Pods. This implementation includes minor optimizations specific to TPUs (including using bfloat16 numerics on more variables, and transposing NCHW-formatted data to NHWC before sending it to the TPU for better performance).

A variety of alternative ResNet-50 training protocols have recently emerged that can accelerate convergence by reducing the amount of computation required for training. For example, progressively scaling up image sizes as training progresses and setting more aggressive learning rate schedules empirically lead to faster convergence on ImageNet. However, for the purposes of this benchmark comparison, we have chosen to stick with the most standard ResNet-50 training protocol, and we hold the amount of computation fixed and then compare the performance of different systems as they carry out the same logical operations.

### Target accuracy and reproducibility: 76% Top-1 accuracy across 5 runs

We set a top-1 accuracy target of **76.0%** on the ImageNet dataset as we believe this is near the top of the achievable range for ResNet-50 v1.5 with the standard training protocol. When training ML models, there are many ways to increase training throughput by sacrificing accuracy. However, those last few percentage points of accuracy are often the most valuable ones in real-world ML applications, so we chose the challenge of training a well-known model to the highest-achievable accuracy.

To make sure that our training results are reproducible, we performed five separate training runs for each hardware configuration, and we certified that all runs achieved at least 76.0% top-1 accuracy on the ImageNet validation dataset with no blacklists.

### Training epochs

Similar to the “[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)” paper, we measure the performance of training 90 epochs on each hardware configuration. Before beginning our measurements, we run a single “warm-up epoch” to exclude one-time setup costs and ensure that data caches are fully filled.

As training time decreases to the order of minutes on our largest Cloud TPU Pod configurations, initialization and compiler overhead becomes significant, per Amdahl’s law. While software optimizations will continue to reduce these overheads, we find that Cloud TPU Pod customers [such as eBay](https://www.ebayinc.com/stories/blogs/tech/large-scale-product-image-recognition-with-cloud-tpus/) typically train on much larger datasets than ImageNet, in which case these overheads are no longer significant, which is why we choose to exclude them here.

We determine the training time for 90 epochs by examining the TensorFlow summary file in the model directory after each training run. Specifically, we measure the training duration as the time between TensorFlow logging the training loss for the last step of the warmup epoch and the last step of the entire training run. We believe that this accurately captures the time taken to train ImageNet for 90 epochs and excludes overheads and cache warming latencies for both GPUs and Cloud TPUs.

### Optimizations for large-batch training: LARS and label smoothing

To enable efficient training on our largest Cloud TPU Pod configurations using batch sizes of 16,384 and larger, our open-source ResNet-50 implementation includes the following optimizations:

1. At large batch sizes, the Cloud TPU implementation switches to the Layer-wise Adaptive Rate Scaling (LARS) optimizer presented by You et al. in “[Large Batch Training of Convolutional Neural Networks](https://arxiv.org/abs/1708.03888)” (arXiv:1708.03888) rather than the conventional stochastic gradient descent optimizer with momentum.

2. At large batch sizes, the Cloud TPU implementation enables “label smoothing” as [described](https://www.tensorflow.org/api_docs/python/tf/losses/softmax_cross_entropy) in the TensorFlow documentation. Label smoothing becomes more important as the total number of gradient updates per training run decreases.

## Detailed Instructions to Reproduce All Experiments

### Training ResNet-50 v1.5 on V100 GPUs on GCP

#### ImageNet Data Preparation for GPUs

Instructions for generating the ImageNet image set.  These instructions result in the data being uploaded to a Google Storage Bucket.

1. Sign up for the ImageNet image database (image-net.org) and obtain a username and access key to download the training and evaluation data.

2. Utilize instructions for the [imagenet_to_gcs.py](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) tool to process the data and upload it to a [Google Storage Bucket](https://cloud.google.com/storage/docs/creating-buckets).

#### VM Sizes for V100 GPUs on GCP

GCP makes it possible to attach GPUs to VMs of different sizes. Since all input pipeline stages (such as JPEG decoding and image pre-processing) happen on the VM CPUs, it is important to rent a large enough VM to keep up with the attached GPUs. Our experiments show that each V100 GPU requires approximately 8 virtual CPU threads for full utilization when training ResNet-50 v1.5, so we chose the following configurations for our experiments to minimize total training costs while maximizing performance:

|Number of GPUs|GCE Machine Size|
|--------------|----------------|
|1 x V100|n1-standard-8|
|4 x V100|n1-standard-32|
|8 x V100|n1-standard-64|

#### GPU training instructions

1. Start the instance type to use for training based on the [Google Deep Learning Images](https://cloud.google.com/deep-learning-vm/docs/) optimized for TensorFlow.

```
export IMAGE_FAMILY="tf-1-12-cu100"
export ZONE=<Your Zone>
export PROJECT=<Your Project>
export INSTANCE_NAME=v100-training

# 1 GPU
gcloud compute instances create $INSTANCE_NAME \
  --machine-type=n1-standard-8 \
  --maintenance-policy=TERMINATE \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --tags=http-server,https-server \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --local-ssd interface=nvme \
  --metadata install-nvidia-driver=True

# 4 GPU
gcloud compute instances create $INSTANCE_NAME \
  --machine-type=n1-standard-32 \
  --maintenance-policy=TERMINATE \
  --accelerator=type=nvidia-tesla-v100,count=4 \
  --tags=http-server,https-server \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --local-ssd interface=nvme \
  --local-ssd interface=nvme \
  --metadata install-nvidia-driver=True

# 8 GPU
gcloud compute instances create $INSTANCE_NAME \
  --machine-type=n1-standard-64 \
  --maintenance-policy=TERMINATE \
  --accelerator=type=nvidia-tesla-v100,count=8 \
  --tags=http-server,https-server \
  --image-family=$IMAGE_FAMILY \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --local-ssd interface=nvme \
  --local-ssd interface=nvme \
  --local-ssd interface=nvme \
  --local-ssd interface=nvme \
  --metadata install-nvidia-driver=True
```

2. Set up local data storage using Google Cloud [local solid-state drives](https://cloud.google.com/compute/docs/disks/local-ssd) (SSD).

```
gcloud compute ssh $INSTANCE_NAME

### Instructions for 1, 4, and 8 GPUs are different.
### The difference is each setup has a different number of nvme drives.
### When more than one drive exist RAID is used to create a single drive.

## 4 and 8 GPU instances
# Installs raid management tool.
sudo apt-get update && sudo apt-get install mdadm --no-install-recommends

# Only run for 8 GPUs with 4x local nvme drives.
sudo mdadm --create /dev/md0 --level=0 --raid-devices=4 \
/dev/nvme0n1 /dev/nvme0n2 /dev/nvme0n3 /dev/nvme0n4 
# Only run for 4 GPUs with 2x local nvme drives.
sudo mdadm --create /dev/md0 --level=0 --raid-devices=2 \
/dev/nvme0n1 /dev/nvme0n2

# Formats and mounts the array.
sudo mkfs.ext4 -F /dev/md0
sudo mkdir -p /data && sudo mount /dev/md0 /data

## 1 GPU instances.
sudo mkfs.ext4 -F /dev/nvme0n1
sudo mkdir -p /data && sudo mount /dev/nvme0n1 /data
```

3. Copy data from your GCS bucket created earlier to the local drive.

```
### Copies data from your GCS bucket created earlier to the local drive.
sudo mkdir -p /data/imagenet && sudo chmod -R 777 /data
gsutil -m cp -r gs://<your bucket with imagenet>/imagenet/* /data/imagenet/
```

4. Install TensorFlow 1.12 compiled with CUDA 10.0, cuDNN 7.3, and AVX2.

```
### Install custom TensorFlow build.
pip install --upgrade --force-reinstall \
https://storage.googleapis.com/tf-performance/tf_binary/tensorflow-1.12.0.a6d8ffa.AVX2.CUDA10-cp27-cp27mu-linux_x86_64.whl
```

5. Start training with the commands below.   For this test, only the loss and the learning rate are recorded with their timestamps to calculate elapsed training time.  Summaries are recorded to disk asynchronously and have not shown to have a performance impact.

```
git clone https://github.com/tensorflow/benchmarks.git && cd benchmarks

git reset --hard 1e7d788042dfc6d5e5cd87410c57d5eccee5c664

# Using tmux keeps the test running if the connection drops
tmux new -s bench

# Execute the command for 1, 4, or 8 GPUs.
# 8 GPUs
python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--batch_size=312 \
--model=resnet50_v1.5 \
--optimizer=momentum \
--variable_update=replicated \
--nodistortions \
--gradient_repacking=2 \
--num_gpus=8 \
--num_epochs=91 \
--weight_decay=1e-4 \
--use_fp16 \
--all_reduce_spec=nccl \
--save_summaries_steps=1 \
--summary_verbosity=1 \
--num_warmup_batches=0 \
--data_dir=/data/imagenet/train \
--train_dir=$HOME/test00 \
--compute_lr_on_cpu=True \
--single_l2_loss_op=True \
--loss_type_to_report=base_loss \
--xla_compile=True

# 4 GPUs
python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--batch_size=312 \
--model=resnet50_v1.5 \
--optimizer=momentum \
--variable_update=replicated \
--nodistortions \
--gradient_repacking=2 \
--num_gpus=4 \
--num_epochs=91 \
--weight_decay=1e-4 \
--use_fp16 \
--all_reduce_spec=nccl \
--save_summaries_steps=1 \
--summary_verbosity=1 \
--num_warmup_batches=0 \
--data_dir=/data/imagenet/train \
--train_dir=$HOME/test00
--compute_lr_on_cpu=True \
--single_l2_loss_op=True \
--loss_type_to_report=base_loss \
--xla_compile=True


# 1 GPU
python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--batch_size=312 \
--model=resnet50_v1.5 \
--optimizer=momentum \
--nodistortions \
--num_gpus=1 \
--num_epochs=91 \
--weight_decay=1e-4 \
--data_dir=/data/imagenet/train \
--use_fp16 \
--train_dir=$HOME/test00 \
--save_summaries_steps=1 \
--summary_verbosity=1 \
--num_warmup_batches=0 \
--compute_lr_on_cpu=True \
--single_l2_loss_op=True \
--loss_type_to_report=base_loss \
--xla_compile=True
```

6. After training is complete, execute the evaluation with one of the commands below:

```
# 4 and 8 GPUs
python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--batch_size=250 \
--model=resnet50_v1.5 \
--variable_update=replicated \
--num_gpus=1 \
--num_batches=200 \
--use_fp16 \
--data_dir=/data/imagenet/validation \
--train_dir=$HOME/test00 \
--eval=True \
--xla_compile=True

# 1 GPU
python scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py \
--batch_size=250 \
--model=resnet50_v1.5 \
--nodistortions \
--num_gpus=1 \
--num_batches=200 \
--use_fp16 \
--data_dir=/data/imagenet/validation \
--train_dir=$HOME/test00 \
--eval=True \
--xla_compile=True
```

7. At the end of the evaluation, you should see the evaluation results printed out in the following format:

```
Accuracy @ 1 = 0.7649 Accuracy @ 5 = 0.9309 [50000 examples]
```

Calculate the training time.

```
# Get script to read event log
git clone https://github.com/tensorflow/tpu.git
cd tpu/models/official/resnet/benchmark

# 8 GPUs
python read_training_time.py --model_dir=$HOME/test00/  \
--warmup_steps=513 \
--end_step=46710  \
--event_name=base_loss

# 4 GPUs
python read_training_time.py --model_dir=$HOME/test00/  \
--warmup_steps=1026 \
--end_step=93419  \
--event_name=base_loss

# 1 GPU
python read_training_time.py --model_dir=$HOME/test00/  \
--warmup_steps=4107 \
--end_step=373674  \
--event_name=base_loss
```

### Training ResNet-50 v1.5 on Cloud TPUs on GCP 

#### ImageNet Data Preparation for Cloud TPUs

1. Create a new GCS bucket, making sure to select the “Regional” storage class and choose a region that is the same as the desired location of your Cloud TPUs.

2. Sign up for the ImageNet image database (image-net.org) and obtain a username and access key to download the training and evaluation data. 

3. Run the [imagenet_to_gcs.py](https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py) on a GCE VM or your local desktop to download, process, and re-upload the data into your GCS bucket.

#### VM Size for Cloud TPUs on GCP

Multiple Cloud TPU hardware configurations are available on GCP, and each one includes a balanced combination of host machines and TPU accelerators. These combinations are constructed automatically and made available over the network. In this study, training data was stored in Google Cloud Storage (GCS) and all input preprocessing happens on the [Cloud TPU server](https://cloud.google.com/tpu/docs/system-architecture), so only a tiny VM is required to orchestrate the computation. We used an n1-standard-2 in all of our Cloud TPU experiments.

#### Cloud TPU Training Instructions

1. Create a Cloud TPU or Cloud TPU Pod slice with TensorFlow version 1.12 via the [Google Cloud console](https://console.cloud.google.com/compute/tpus).

2. [Start](https://console.cloud.google.com/compute/tpus) a new n1-standard-2 GCE VM with Ubuntu 16.04 LTS and the “Allow full access to all Cloud APIs option”, and then [install TensorFlow 1.12](https://www.tensorflow.org/install/) on the VM.

3. Clone the Cloud TPU GitHub repository located at https://github.com/tensorflow/tpu using the command `git clone https://github.com/tensorflow/tpu`.

4. Go to the local copy of the repository and check out the `r1.12` branch using `git checkout r1.12`.

5. Start ResNet training using the following command line in the GCE VM. We recommend that you start a screen session so that training will be uninterrupted even if the SSH connection to your VM is temporarily lost.

```
export PYTHONPATH="$PYTHONPATH:~/tpu/models"

python tpu/models/official/resnet/resnet_main.py \
  --tpu=MY_TPU_NAME --tpu_zone=MY_TPU_ZONE --num_cores=TPU_CORES \
  --data_dir=gs://IMAGENET_DATA_BUCKET/DIRECTORY \
  --model_dir=gs://RESNET_CHECKPOINT_BUCKET/DIRECTORY \
  --train_batch_size=BATCH_SIZE --iterations_per_loop=ITERATIONS \
  --train_steps=ITERATIONS \
  --mode=train --eval_batch_size=1000
```

Use the following parameters for various Cloud TPU system sizes:

|TPU Type                   |`TPU_CORES`|`BATCH_SIZE`|`ITERATIONS`|
|---------------------------|---------|----------|----------|
|Cloud TPU v2               |8        |1024      |113854    |
|1/16 Cloud TPU Pod (v2-32) |32       |4096      |28464     |
|1/4 Cloud TPU Pod (v2-128) |128      |16384     |7116      |
|1/2 Cloud TPU Pod (v2-256) |256      |32768     |3558      |
|Full Cloud TPU Pod (v2-512)|512      |32768     |3558      |

In addition, add the following additional parameters for batch sizes >= 16384 to enable the Layer-wise Adaptive Rate Scaling optimizer and label smoothing changes needed for large-batch training: `--enable_lars=True --label_smoothing=0.1`.

This script runs a total of 91 epochs (one warm-up epoch and 90 training epochs). For each result, we run five complete training runs of the script and report the median elapsed time. All of the runs using the above `BATCH_SIZE` and `ITERATIONS` parameters were observed to reach 76% top-1 accuracy.

After training completes, we then measure and report the training time using the method specified in the “Training epochs” section of this methodology. To implement the methodology, we have provided a script for use here. The warmup_steps parameter (corresponding to one training epoch) used are as follows:

|TPU Type                   |`warmup_steps`|
|---------------------------|--------------|
|Cloud TPU v2               |1251          |
|1/16 Cloud TPU Pod (v2-32) |313           |
|1/4 Cloud TPU Pod (v2-128) |78            |
|1/2 Cloud TPU Pod (v2-256) |39            |
|Full Cloud TPU Pod (v2-512)|39            |

A sample command is as follows:

```
python ~/tpu/models/official/resnet/benchmark/read_training_time.py \
  --model_dir=gs://RESNET_CHECKPOINT_BUCKET/DIRECTORY
  --warmup_steps=WARMUP_STEPS
  --tpu=True
```

6. Start ResNet evaluation using the following command line on the GCE VM. Note that evaluation is only supported on a single Cloud TPU at present.

```
python ~/tpu/models/official/resnet/resnet_main.py \
  --tpu=MY_TPU_NAME --tpu_zone=MY_TPU_ZONE --num_cores=8 \
  --data_dir=gs://IMAGENET_DATA_BUCKET/DIRECTORY \
  --model_dir=gs://RESNET_CHECKPOINT_BUCKET/DIRECTORY \
  --train_batch_size=BATCH_SIZE --iterations_per_loop=ITERATIONS \
  --train_steps=ITERATIONS \
  --mode=eval --eval_batch_size=1000 
```

At the end of the evaluation, you should see the evaluation results printed out in the following format:

```
Eval results: {'loss': 2.2301836, 'top_1_accuracy': 0.76658, 'global_step': 28151, 'top_5_accuracy': 0.93422}. Elapsed seconds: 33
```
