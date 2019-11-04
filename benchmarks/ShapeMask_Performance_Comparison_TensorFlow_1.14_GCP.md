# Methodology for ShapeMask Performance Benchmark on Cloud TPUs

Weicheng Kuo, Anelia Angelova, Pengchong Jin, Zak Stone, Omkar Pathak, Tsungyi Lin *(Google Brain)* (order TBD).

## Performance Measurement
This study focuses on the scaling capability of ShapeMask training while maintaining the target accuracy. We measure training time as the time between "Init TPU system" and "Shutdown TPU system". This captures the whole time span that the TPU is on, but excludes the time of setting up the connection to TPU.

### Implementation Details
We choose COCO to test our instance segmentation model, as it is the standard dataset in the community. The default training schedule follows the 2X schedule of [Detectron Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md). To leverage the scaling capability of TPU, we scale up the batch size from 16 to 64 (4x), and initial learning rate from 0.02 to 0.08 (4x). We train for the same number of epochs as Detectron, which means we have 45k iterations as opposed to their 180k. The model architecture of choice is ResNet-101-FPN, consistent with what we reported in the ShapeMask [paper](https://arxiv.org/abs/1904.03239) as well. We resize all input images to 1024 on the longer side, which is comparable to the input sizes used in Detectron.

All experiments are performed on Google’s Cloud TPU v3-8 device (batch size = 64) and larger slices of Cloud TPU v3 pods (batch_size > 64). TPU v3 offers significant speedup over v2, so we use it to demonstrate the speed of our system. This implementation uses bfloat16 numerics and input data transpose to improve TPU performance.

## Benchmark Results
The results of ShapeMask scaling experiments are as follows

|Model|Batch Size|Number of Cores|Mask AP|Box AP|Training Time (mins)|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Mask R-CNN|64|8|37.3|42.1|730|
|ShapeMask|64|8|38.0|41.6|485|
|ShapeMask|256|32|37.9|41.5|187|
|ShapeMask|1024|128|35.1|37.9|51|
|ShapeMask|2048|256|34.7|37.1|36|

### Benchmark Configurations
Here are the optimization schedules for the experiments.

|Model|Batch Size|Number of Cores|Total Steps|Warmup Steps|Initial Learning Rate|First Decay Step|Second Decay Step|
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|Mask R-CNN|64|8|45000|500|0.08|30000|40000|
|ShapeMask|64|8|45000|500|0.08|30000|40000|
|ShapeMask|256|32|11250|1600|0.24|7500|10000|
|ShapeMask|1024|128|2813|1093|0.64|2188|2656|
|ShapeMask|2048|256|1800|600|0.64|1200|1600|

### Commands
Here are the commands to run the scaling experiments of ShapeMask.

Download code and dependencies:
```
# Install packages
sudo apt-get install -y python-tk && \
pip install --user Cython matplotlib opencv-python-headless pyyaml Pillow && \
pip install --user 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'

# Download the code base.
git clone https://github.com/tensorflow/tpu/
```

Download and prepare data:
```
export USER=weicheng  # Your user name.

# Export the bucket path to env.
export STORAGE_BUCKET=gs://${USER}-data

# Create storage bucket.
gsutil mb $STORAGE_BUCKET

# Download COCO data.
mkdir ~/data
mkdir ~/data/coco

cd ~/tpu/tools/datasets
bash download_and_preprocess_coco.sh ~/data/coco

# Create coco directory under the bucket.
mkdir coco
touch coco/empty.txt
gsutil cp -r coco $STORAGE_BUCKET

# Move data over to bucket.
gsutil -m cp data/coco/*.tfrecord gs://${USER}-data/coco
gsutil -m cp data/coco/raw-data/annotations/*.json gs://${USER}-data/coco

# Create shapemask directory under the bucket.
mkdir shapemask_exp
touch shapemask_exp/empty.txt
gsutil cp -r shapemask_exp gs://${USER}-data/

# Back to home directory.
cd ~
```

Setup environment variables:
```
export TPU_NAME=''  # Your tpu name.
export EVAL_TPU_NAME=''  # Your evaluation tpu name. tf.Estimator only supports 2x2 at the moment.
export EXP_NAME=shapemask_demo_run  # Your experiment name.
export MODEL_DIR=${STORAGE_BUCKET}/shapemask_exp/${EXP_NAME};  # You must have created shapemask directory under the bucket.
export RESNET_CHECKPOINT=gs://cloud-tpu-checkpoints/shapemask/retinanet/resnet101-checkpoint-2018-02-24;
export TRAIN_FILE_PATTERN=${STORAGE_BUCKET}/coco/train-*;  # Make sure coco directory exists under your bucket.
export EVAL_FILE_PATTERN=${STORAGE_BUCKET}/coco/val-*;
export VAL_JSON_FILE=${STORAGE_BUCKET}/coco/instances_val2017.json;
export SHAPE_PRIOR_PATH=gs://cloud-tpu-checkpoints/shapemask/kmeans_class_priors_91x20x32x32.npy
export PYTHONPATH="/home/${USER}/tpu/models"
```

Training commands:
```
export NUM_CORES=8  # Number of cores in your TPU. 8 for 2x2.
export BATCH_SIZE=64  # Your batch size.
export TOTAL_STEPS=45000  # Total number of steps.
export WARMUP_STEPS=500  # Warmup steps to use.
export INIT_LR=0.08  # Initial learning rate.
export FIRST_DECAY_LR=0.008  # First decay learning rate. Set to 0.1 of INIT_LR
export FIRST_DECAY_STEPS=30000  # First decay learning rate step.
export SECOND_DECAY_LR=0.0008  # Second decay learning rate. Set to 0.01 of INIT_LR
export SECOND_DECAY_STEPS=40000  # Second decay learning rate step.

python ~/tpu/models/official/detection/main.py --model shapemask --use_tpu=True \
--tpu=${TPU_NAME} --num_cores=${NUM_CORES} --model_dir="${MODEL_DIR}" --mode="train" \
--params_override="{ train: { \
iterations_per_loop: 1000, train_batch_size: ${BATCH_SIZE}, total_steps: ${TOTAL_STEPS}, \
learning_rate: {total_steps: ${TOTAL_STEPS}, warmup_learning_rate: 0.0067, warmup_steps: ${WARMUP_STEPS}, \
init_learning_rate: ${INIT_LR},learning_rate_levels: [${FIRST_DECAY_LR}, ${SECOND_DECAY_LR}], \
learning_rate_steps: [${FIRST_DECAY_STEPS}, ${SECOND_DECAY_STEPS}]}, \
checkpoint: { path: ${RESNET_CHECKPOINT}, prefix: resnet101/ }, \
train_file_pattern: ${TRAIN_FILE_PATTERN} }, \
resnet: {resnet_depth: 101}, \
eval: { val_json_file: ${VAL_JSON_FILE}, eval_file_pattern: ${EVAL_FILE_PATTERN}, eval_samples: 5000 }, \
shapemask_head: {use_category_for_mask: true, shape_prior_path: ${SHAPE_PRIOR_PATH}}, \
shapemask_parser: {output_size: [1024, 1024]}, \
retinanet_loss: {focal_loss_alpha: 0.4, huber_loss_delta: 0.15}}" \
2>&1 | tee ${EXP_NAME}_log_training.txt
```

Evaluation Commands:
```
python ~/tpu/models/official/detection/main.py \
--model shapemask --use_tpu=True --tpu=${EVAL_TPU_NAME} \
--num_cores=8 --model_dir="${MODEL_DIR}" --mode="eval" \
--params_override="{resnet: {resnet_depth: 101}, \
eval: { val_json_file: ${VAL_JSON_FILE}, eval_file_pattern: ${EVAL_FILE_PATTERN}, eval_samples: 5000 }, \
shapemask_head: {use_category_for_mask: true, shape_prior_path: ${SHAPE_PRIOR_PATH}}, \
shapemask_parser: {output_size: [1024, 1024]}}"
```

At the end of the evaluation, you should see the evaluation results printed out in the following format:

```
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=94.92s).
Accumulating evaluation results...
DONE (t=12.52s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.606
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.447
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.341
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.542
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.385
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.618
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.730
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=101.21s).
Accumulating evaluation results...
DONE (t=13.03s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.380
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.580
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.409
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.171
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.413
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.555
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.505
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.533
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.582
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.718
```
