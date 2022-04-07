#  Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""An Example of a custom TPUEstimator for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import iris_data_tpu as iris_data
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be the name used when "
    "creating the Cloud TPU. To find out hte name of TPU, either use command "
    "'gcloud compute tpus list --zone=<zone-name>', or use "
    "'ctpu status --details' if you have created Cloud TPU using 'ctpu up'.")

# Model specific parameters
tf.flags.DEFINE_string(
    "model_dir", default="",
    help="This should be the path of GCS bucket which will be used as "
    "model_directory to export the checkpoints during training.")
tf.flags.DEFINE_integer(
    "batch_size", default=128,
    help="This is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer(
    "train_steps", default=1000,
    help="Total number of training steps.")
tf.flags.DEFINE_integer(
    "eval_steps", default=4,
    help="Total number of evaluation steps. If `0`, evaluation "
    "after training is skipped.")

# TPU specific parameters.
tf.flags.DEFINE_bool(
    "use_tpu", default=True,
    help="True, if want to run the model on TPU. False, otherwise.")
tf.flags.DEFINE_integer(
    "iterations", default=500,
    help="Number of iterations per TPU training loop.")

FLAGS = tf.flags.FLAGS


def metric_fn(labels, logits):
  """Function to return metrics for evaluation."""

  predicted_classes = tf.argmax(logits, 1)
  accuracy = tf.metrics.accuracy(labels=labels,
                                 predictions=predicted_classes,
                                 name="acc_op")
  return {"accuracy": accuracy}


def my_model(features, labels, mode, params):
  """Deep Neural Network(DNN) model.

  This is a DNN Model with 3 hidden layers. First 2 hidden layers are having
  10 neurons in each. And number of neurons in the last layer is equal to the
  number of output classes. This is a densely connected network where each
  neuron of previous layer is connected to each neuron of next layer.

  Args:
    features: Feature values for input samples.
    labels: label/class assigned to the corresponding input sample.
    mode: "TRAIN"/"EVAL"/"PREDICT"
    params: Dictionary used to pass extra parameters to model function from
      the main function.

  Returns:
    TPUEstimator object.

  """

  # Create three fully connected layers.
  net = tf.feature_column.input_layer(features, params["feature_columns"])
  for units in params["hidden_units"]:
    net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

  # Compute logits (1 per class).
  logits = tf.layers.dense(net, params["n_classes"], activation=None)

  # Compute predictions.
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf_estimator.ModeKeys.PREDICT:
    predictions = {
        "class_ids": predicted_classes[:, tf.newaxis],
        "probabilities": tf.nn.softmax(logits),
        "logits": logits,
    }
    return contrib_tpu.TPUEstimatorSpec(mode, predictions=predictions)

  # Compute loss.
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                logits=logits)

  if mode == tf_estimator.ModeKeys.EVAL:
    return contrib_tpu.TPUEstimatorSpec(
        mode=mode, loss=loss, eval_metrics=(metric_fn, [labels, logits]))

  # Create training op.
  if mode == tf_estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    if FLAGS.use_tpu:
      optimizer = contrib_tpu.CrossShardOptimizer(optimizer)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return contrib_tpu.TPUEstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
  del argv  # Unused
  if FLAGS.use_tpu:
    assert FLAGS.model_dir.startswith("gs://"), ("'model_dir' should be a "
                                                 "GCS bucket path!")

  # Fetch the data
  (train_x, train_y), (test_x, test_y) = iris_data.load_data()

  # Feature columns describe how to use the input.
  my_feature_columns = []
  for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

  # Resolve TPU cluster and runconfig for this.
  tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(FLAGS.tpu)

  run_config = contrib_tpu.RunConfig(
      model_dir=FLAGS.model_dir,
      cluster=tpu_cluster_resolver,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=contrib_tpu.TPUConfig(FLAGS.iterations),
  )

  # Build 2 hidden layer DNN with 10, 10 units respectively.
  classifier = contrib_tpu.TPUEstimator(
      model_fn=my_model,
      use_tpu=FLAGS.use_tpu,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      predict_batch_size=FLAGS.batch_size,
      config=run_config,
      params={
          # Name of the feature columns in the input data.
          "feature_columns": my_feature_columns,
          # Two hidden layers of 10 nodes each.
          "hidden_units": [10, 10],
          # The model must choose between 3 classes.
          "n_classes": 3,
          "use_tpu": FLAGS.use_tpu,
      })

  # Train the Model.
  classifier.train(
      input_fn=lambda params: iris_data.train_input_fn(
          train_x, train_y, params["batch_size"]),
      max_steps=FLAGS.train_steps)

  # Evaluate the model.
  eval_result = classifier.evaluate(
      input_fn=lambda params: iris_data.eval_input_fn(
          test_x, test_y, params["batch_size"]),
      steps=FLAGS.eval_steps)

  print("\nTest set accuracy: {accuracy:0.3f}\n".format(**eval_result))

  # Generate predictions from the model
  predictions = classifier.predict(
      input_fn=lambda params: iris_data.predict_input_fn(
          iris_data.PREDICTION_INPUT_DATA, params["batch_size"]))

  for pred_dict, expec in zip(predictions, iris_data.PREDICTION_OUTPUT_DATA):
    template = ("\nPrediction is \"{}\" ({:.1f}%), expected \"{}\"")

    class_id = pred_dict["class_ids"][0]
    probability = pred_dict["probabilities"][class_id]

    print(template.format(iris_data.SPECIES[class_id],
                          100 * probability, expec))

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main)
