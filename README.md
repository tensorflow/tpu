# Cloud TPUs #

This repository is a collection of reference models and tools used with
[Cloud TPUs](https://cloud.google.com/tpu/).

The fastest way to get started training a model on a Cloud TPU is by following
the tutorial. Click the button below to launch the tutorial using Google Cloud
Shell.

[![Open in Cloud Shell](http://gstatic.com/cloudssh/images/open-btn.svg)](https://console.cloud.google.com/cloudshell/open?git_repo=https%3A%2F%2Fgithub.com%2Ftensorflow%2Ftpu&page=shell&tutorial=tools%2Fctpu%2Ftutorial.md)

_Note:_ This repository is a public mirror, pull requests will not be accepted.
Please file an issue if you have a feature or bug request.

## Running Models

To run models in the `models` subdirectory, you may need to add the top-level
`/models` folder to the Python path with the command:

```
export PYTHONPATH="$PYTHONPATH:/path/to/models"
```
