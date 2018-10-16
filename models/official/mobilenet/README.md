# Cloud TPU Port of the MobileNet v1 model

This is a straightforward port of the [MobileNet v1 model](https://arxiv.org/pdf/1704.04861.pdf).  The code was based on the original version from the [tensorflow/models](https://github.com/tensorflow/models/tree/master/research/slim/nets) repository.

The only adjustments have been to add the required code to enable using the 
TPUEstimator interface, along with the data processing pipeline for ImageNet.

## Running the model

Assuming you have a version of ImageNet converted to the tfrecord format located
at `gs://my-cloud-bucket/data/imagenet/`, you can run this model with the 
following command:

```
python mobilenet.py\ 
  --alsologtostderr\
  --master=$TPU_WORKER\
  --data_dir=gs://my-cloud-bucket/data/imagenet\
  --model_dir=gs://my-cloud-bucket/models/mobilenet/v0\
  --num_shards=8\
  --batch_size=1024\
  --use_tpu=1\
```

Note that the mobilenet network requires a large number of epochs to converge
completely.
