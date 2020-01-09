# About
This folder contains a suite of tools that builds upon [tensorflow/datasets](https://www.tensorflow.org/datasets)
that can be used to easily convert raw data into the TFRecord format on GCS.
This is helpful because data must be stored in [TFRecords](https://www.tensorflow.org/tutorials/load_data/tf_records)
on [GCS](https://cloud.google.com/storage/) to run with TPU models.

# High-Level Overview
The folder is divided by task and each task has specific fields that are required
"essential inputs" for each task.

For example, image classification requires an image and a label. However, models
may require more features, and this tool both facilitates the extraction of
these extra features and converts the data into TFRecords.

Currently supported tasks:
- Image Classification

# Usage
To use the tool, create an implementation of one of the abstract BuilderConfigs.

For example:
```
class MyBuilderConfig(ImageClassificationDataConfig):
  ...

config = MyBuilderConfig(name="MyBuilderConfig",
                         description="MyBuilderConfig")
ds = ImageClassificationData(config)
ds.download_and_prepare()

```

In each folder are also simple examples for further reference.
