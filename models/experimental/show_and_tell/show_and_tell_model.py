# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Standard Imports
import tensorflow as tf

import image_embedding
import image_processing
import inputs as input_ops


class ShowAndTellModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def __init__(self, config, mode, train_inception=False):
    """Basic setup.

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Collection of variables from the inception submodel.
    self.inception_variables = []

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def load_image(self, encoded_image, thread_id=0):
    """Decodes and processes an image string.

    Args:
      encoded_image: A scalar string Tensor; the encoded image.
      thread_id: Preprocessing thread id used to select the ordering of color
        distortions.

    Returns:
      A float32 Tensor of shape [height, width, 3]; the processed image.
    """
    return image_processing.process_image(
        encoded_image,
        is_training=self.is_training(),
        height=self.config.image_height,
        width=self.config.image_width,
        thread_id=thread_id,
        image_format=self.config.image_format)

  def distort_images(self, images, seed):
    """Distort a batch of images.

    (Processing a batch allows us to easily switch between TPU and CPU
    execution).
    """
    if self.mode == "train":
      images = image_processing.distort_image(images, seed)

    # Rescale to [-1,1] instead of [0, 1]
    images = tf.subtract(images, 0.5)
    images = tf.multiply(images, 2.0)
    return images

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.

    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.string, shape=[], name="image_feed")
      input_feed = tf.placeholder(
          dtype=tf.int64,
          shape=[None],  # batch_size
          name="input_feed")

      # Process image and insert batch dimensions.
      images = tf.expand_dims(self.load_image(image_feed), 0)
      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None
    else:

      def _load_example(serialized_example):
        encoded_image, caption = input_ops.parse_example(
            serialized_example,
            image_feature=self.config.image_feature_name,
            caption_feature=self.config.caption_feature_name)
        image = self.load_image(encoded_image)
        # strings.split expects a batch
        words = tf.strings.split(tf.reshape(caption, [1]), sep=" ")
        words = tf.sparse_tensor_to_dense(words, default_value="")[0]
        word_idx = tf.strings.to_hash_bucket(words, self.config.vocab_size)
        input_seqs, target_seqs, input_mask = input_ops.pad_caption_to_input(
            word_idx)
        return image, input_seqs, target_seqs, input_mask

      def _load_dataset(filename):
        return tf.data.TFRecordDataset(filename, buffer_size=16 * 1024 * 1024)

      df = tf.data.Dataset.list_files(
          self.config.input_file_pattern, shuffle=self.mode == "train")
      df = df.apply(
          tf.data.experimental.parallel_interleave(
              _load_dataset, cycle_length=64, sloppy=True))

      if self.mode == "train":
        df = df.repeat()
        df = df.shuffle(1024)

      df = df.apply(
          tf.data.experimental.map_and_batch(
              _load_example,
              self.config.batch_size,
              num_parallel_batches=8,
              drop_remainder=True))
      df = df.prefetch(8)
      images, input_seqs, target_seqs, input_mask = df.make_one_shot_iterator(
      ).get_next()

    self.images = images
    self.input_seqs = input_seqs
    self.target_seqs = target_seqs
    self.input_mask = input_mask

  def build_image_embeddings(self, images):
    """Builds the image model subgraph and generates image embeddings.

    Inputs:
      images

    Outputs:
      self.image_embeddings
    """
    images = self.distort_images(images, tf.train.get_or_create_global_step())
    inception_output = image_embedding.inception_v3(
        images,
        trainable=self.train_inception,
        is_training=self.is_training(),
        add_summaries=False)

    self.inception_variables = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="InceptionV3")

    # Map inception output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=inception_output,
          num_outputs=self.config.embedding_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          biases_initializer=None,
          scope=scope)

    # Save the embedding size in the graph.
    tf.constant(self.config.embedding_size, name="embedding_size")

    return image_embeddings

  def build_seq_embeddings(self, input_seqs):
    """Builds the input sequence embeddings.

    Inputs:
      input_seqs

    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

    return seq_embeddings

  def build_model(self):
    """Builds the model.

    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)

    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
    if self.mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)

    with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      zero_state = lstm_cell.zero_state(
          batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
      _, initial_state = lstm_cell(self.image_embeddings, zero_state)

      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()

      if self.mode == "inference":
        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
        tf.concat(initial_state, 1, name="initial_state")

        # Placeholder for feeding a batch of concatenated states.
        state_feed = tf.placeholder(
            dtype=tf.float32,
            shape=[None, sum(lstm_cell.state_size)],
            name="state_feed")
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

        # Run a single LSTM step.
        lstm_outputs, state_tuple = lstm_cell(
            inputs=tf.squeeze(self.seq_embeddings, squeeze_dims=[1]),
            state=state_tuple)

        # Concatentate the resulting state.
        tf.concat(state_tuple, 1, name="state")
      else:
        # Run the batch of sequence embeddings through the LSTM.
        sequence_length = tf.reduce_sum(self.input_mask, 1)
        lstm_outputs, _ = tf.nn.dynamic_rnn(
            cell=lstm_cell,
            inputs=self.seq_embeddings,
            sequence_length=sequence_length,
            initial_state=initial_state,
            dtype=tf.float32,
            scope=lstm_scope)

    # Stack batches vertically.
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    with tf.variable_scope("logits") as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=lstm_outputs,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_scope)

    if self.mode == "inference":
      tf.nn.softmax(logits, name="softmax")
    else:
      targets = tf.reshape(self.target_seqs, [-1])
      weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

      # Compute losses.
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=targets, logits=logits)
      batch_loss = tf.div(
          tf.reduce_sum(tf.multiply(losses, weights)),
          tf.reduce_sum(weights),
          name="batch_loss")
      tf.losses.add_loss(batch_loss)
      total_loss = tf.losses.get_total_loss()

      self.total_loss = total_loss
      self.target_cross_entropy_losses = losses  # Used in evaluation.
      self.target_cross_entropy_loss_weights = weights  # Used in evaluation.

  def setup_inception_initializer(self):
    """Sets up the function to restore inception variables from checkpoint."""
    if self.mode != "inference":
      # Restore inception variables only.
      saver = tf.train.Saver(self.inception_variables)

      def restore_fn(sess):
        tf.logging.info("Restoring Inception variables from checkpoint file %s",
                        self.config.inception_checkpoint_file)
        saver.restore(sess, self.config.inception_checkpoint_file)

      self.init_fn = restore_fn

  def setup_global_step(self):
    """Sets up the global step Tensor."""
    self.global_step = tf.train.get_or_create_global_step()

  def build_model_for_tpu(self, images, input_seqs, target_seqs, input_mask):
    self.image_embeddings = self.build_image_embeddings(images)
    self.seq_embeddings = self.build_seq_embeddings(target_seqs)
    self.target_seqs = target_seqs
    self.input_mask = input_mask
    self.build_model()

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.image_embeddings = self.build_image_embeddings(self.images)
    self.seq_embeddings = self.build_seq_embeddings(self.input_seqs)
    self.build_model()
    self.setup_inception_initializer()
    self.setup_global_step()
