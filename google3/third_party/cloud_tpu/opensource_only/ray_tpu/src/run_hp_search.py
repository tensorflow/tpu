# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Hyperparameter search FLAX MNIST with Ray backend."""
import getpass
import os
from typing import Any, Mapping

from absl import app
from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from ray import tune
from ray_tpu_controller import RayRuntimeEnv
from ray_tpu_controller import RayTpuController
import tensorflow_datasets as tfds
from tpu_api import get_default_gcp_project


NUM_TRIALS = 3
NUM_SAMPLES = 3


@jax.jit
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params):
    logits = state.apply_fn({"params": params}, images)
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""

  train_ds_size = len(train_ds["image"])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds["image"]))
  perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_loss = []
  epoch_accuracy = []

  for perm in perms:
    batch_images = train_ds["image"][perm, ...]
    batch_labels = train_ds["label"][perm, ...]
    grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
    state = update_model(state, grads)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder("mnist")
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
  train_ds["image"] = jnp.float32(train_ds["image"]) / 255.0
  test_ds["image"] = jnp.float32(test_ds["image"]) / 255.0
  return train_ds, test_ds


def create_train_state(rng, config):
  """Creates initial `TrainState`."""

  class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
      x = nn.Conv(features=32, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = nn.Conv(features=64, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
      x = x.reshape((x.shape[0], -1))  # flatten
      x = nn.Dense(features=256)(x)
      x = nn.relu(x)
      x = nn.Dense(features=10)(x)
      return x

  cnn = CNN()
  params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))["params"]
  tx = optax.sgd(config.learning_rate, config.momentum)
  return train_state.TrainState.create(apply_fn=cnn.apply, params=params, tx=tx)


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> float:
  """Executes model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The final test accuracy.
  """
  train_ds, test_ds = get_datasets()
  rng = jax.random.PRNGKey(0)

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, config)
  final_accuracy = 0

  for epoch in range(1, config.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_accuracy = train_epoch(
        state, train_ds, config.batch_size, input_rng
    )
    _, test_loss, test_accuracy = apply_model(
        state, test_ds["image"], test_ds["label"]
    )

    logging.info(
        (
            "epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss:"
            " %.4f, test_accuracy: %.2f"
        ),
        epoch,
        train_loss,
        train_accuracy * 100,
        test_loss,
        test_accuracy * 100,
    )

    summary_writer.scalar("train_loss", train_loss, epoch)
    summary_writer.scalar("train_accuracy", train_accuracy, epoch)
    summary_writer.scalar("test_loss", test_loss, epoch)
    summary_writer.scalar("test_accuracy", test_accuracy, epoch)
    final_accuracy = np.array(test_accuracy)

  summary_writer.flush()
  return final_accuracy


def get_default_run_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  config.learning_rate = 0.1
  config.momentum = 0.9
  config.batch_size = 128
  config.num_epochs = 10
  return config


def hp_search_mnist(tuner_config: Mapping[str, Any]):
  """Runs hyperparameter search given a Ray Tune config."""
  os.environ["TPU_MIN_LOG_LEVEL"] = "0"
  os.environ["TPU_STDERR_LOG_LEVEL"] = "0"
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  logging.set_verbosity(logging.INFO)

  root_workdir = os.path.join("/tmp", "hp_search_mnist")
  os.makedirs(root_workdir, exist_ok=True)

  for trial_number in range(NUM_TRIALS):
    run_config = get_default_run_config()
    run_config.learning_rate = tuner_config["learning_rate"]
    run_config.learning_rate = tuner_config["momentum"]
    workdir = os.path.join(root_workdir, "trial-{}".format(trial_number))
    accuracy = train_and_evaluate(run_config, workdir)
    logging.info("Accuracy: %.2f", accuracy)
    tune.report(mean_accuracy=accuracy)


def main(_):
  tpu_name = getpass.getuser() + "-ray-test"
  project = get_default_gcp_project()
  print(project)

  pip_installs = [
      "absl-py==1.0.0",
      "clu==0.0.6",
      "jax[tpu]==0.3.4",
      "-f https://storage.googleapis.com/jax-releases/libtpu_releases.html",
      "ray[tune]",
      "flax==0.4.1",
      "tensorflow-cpu",
      "tensorflow-datasets",
      "ml-collections==0.1.0",
      "protobuf==3.19.0",
      "pandas",
  ]

  controller = RayTpuController(
      tpu_name=tpu_name,
      project=project,
      zone="us-central2-b",
      accelerator_type="V4",
      accelerator_topology="2x2x1",
      version="tpu-vm-v4-base",
      runtime_env=RayRuntimeEnv(
          pip=pip_installs, working_dir=os.path.expanduser("~/src")
      ),
  )
  controller.maybe_create_and_wait_for_ready()

  search_space = {
      "learning_rate": tune.sample_from(
          lambda spec: 10 ** (-10 * np.random.rand())
      ),
      "momentum": tune.uniform(0.1, 0.9),
  }
  trainable_with_resources = tune.with_resources(
      hp_search_mnist, tune.PlacementGroupFactory([{"CPU": 240, "tpu_host": 1}])
  )
  tuner = tune.Tuner(
      trainable_with_resources,
      param_space=search_space,
      tune_config=tune.TuneConfig(
          metric="mean_accuracy",
          mode="max",
          num_samples=NUM_SAMPLES,
      ),
  )
  results = tuner.fit()
  logging.info(results.get_dataframe())
  controller.delete_tpu()


if __name__ == "__main__":
  logging.set_verbosity(logging.INFO)
  app.run(main)
