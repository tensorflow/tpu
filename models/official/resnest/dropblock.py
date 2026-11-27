import tensorflow as tf
from tensorflow.python.keras import backend as K


def _bernoulli(shape, mean):
  return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))


class DropBlock2D(tf.keras.layers.Layer):
  def __init__(self, keep_prob, block_size, scale=True, **kwargs):
    super(DropBlock2D, self).__init__(**kwargs)
    self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
    self.block_size = int(block_size)
    self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

  def compute_output_shape(self, input_shape):
    return input_shape

  def build(self, input_shape):
    assert len(input_shape) == 4
    _, self.h, self.w, self.channel = input_shape.as_list()
    # pad the mask
    p1 = (self.block_size - 1) // 2
    p0 = (self.block_size - 1) - p1
    self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
    self.set_keep_prob()
    super(DropBlock2D, self).build(input_shape)

  def call(self, inputs, training=None, **kwargs):
    def drop():
      mask = self._create_mask(tf.shape(inputs))
      output = inputs * mask
      output = tf.cond(self.scale,
                        true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                        false_fn=lambda: output)
      return output

    if training is None:
      training = K.learning_phase()
    output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                      true_fn=lambda: inputs,
                      false_fn=drop)
    return output

  def set_keep_prob(self, keep_prob=None):
    """This method only supports Eager Execution"""
    if keep_prob is not None:
      self.keep_prob = keep_prob
    w, h = tf.to_float(self.w), tf.to_float(self.h)
    self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                 ((w - self.block_size + 1) * (h - self.block_size + 1))

  def _create_mask(self, input_shape):
    sampling_mask_shape = tf.stack([input_shape[0],
                                    self.h - self.block_size + 1,
                                    self.w - self.block_size + 1,
                                    self.channel])
    mask = _bernoulli(sampling_mask_shape, self.gamma)
    mask = tf.pad(mask, self.padding)
    mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
    mask = 1 - mask
    return mask
