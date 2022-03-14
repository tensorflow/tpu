# About
This folder demonstrates a simple, **experimental** way you can run a
TensorFlow TPU VM pod with custom dependencies by starting `grpc_tpu_worker.py`
yourself.

TPU VM TF pod versions (versions ending in `-pod`, e.g.
`tpu-vm-tf-2.8.0-pod`) are the same as the non-pod versions except they contain
the `tpu-runtime` container for convenience that starts `grpc_tpu_worker.py`.

## Step-by step
Create the TPU VM. Check [Cloud TPU VM user's guide](https://cloud.google.com/tpu/docs/users-guide-tpu-vm)
for more info. For example:

```
gcloud alpha compute tpus tpu-vm create $TPU_NAME \
  --accelerator-type=v2-32 --version=tpu-vm-tf-2.8.0
```

Stop the existing `tpu-runtime` container on all workers:
```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=all \
  --command="sudo systemctl stop tpu-runtime"
```

Get this code
```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=all \
  --command="wget https://raw.githubusercontent.com/tensorflow/tpu/master/tools/grpc_tpu_worker/grpc_tpu_worker.py"
```

Start the `grpc_tpu_worker.py` on all workers:
```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME --worker=all \
  --command="python3 grpc_tpu_worker.py"
```

Run a sample ResNet workload:
```
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME \
  --command="python3 /usr/share/tpu/tensorflow/resnet50_keras/resnet50.py --tpu=$TPU_NAME --data=gs://cloud-tpu-test-datasets/fake_imagenet"
```
