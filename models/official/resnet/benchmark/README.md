# ResNet-50 Benchmark on Cloud TPU pods

Submission for [DAWNBench](https://dawn.cs.stanford.edu/benchmark/index.html).

This subdirectory contains the code needed to replicate the DAWNBench results
for ResNet-50 on a Cloud TPU pod. The model used here is identical to the model
in the parent directory. The only difference is that `resnet_benchmark.py` will
generate checkpoints at every epoch and evaluate in a separate job.

## Instructions for training on single Cloud TPU

1. Add the top-level `/models` folder to the Python path with the command

```
export PYTHONPATH="$PYTHONPATH:/path/to/models"
```

1. Train the model (roughly 90 epochs, 1 checkpoint per epoch):
```
python resnet_benchmark.py \
  --tpu=[TPU NAME] \
  --mode=train \
  --data_dir=[PATH TO DATA] \
  --model_dir=[PATH TO MODEL] \
  --train_batch_size=1024 \
  --train_steps=112590 \
  --iterations_per_loop=1251
```

1. Evaluate the model (run after train completes):
```
python resnet_benchmark.py \
  --tpu=[TPU NAME] \
  --mode=eval \
  --data_dir=[PATH TO DATA] \
  --model_dir=[PATH TO MODEL]
```

## Instructions for training on a half TPU Pod

Not yet available due to TPU Pod availability in Cloud.

