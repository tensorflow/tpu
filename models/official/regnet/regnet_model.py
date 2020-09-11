from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.tpu import tpu_function


def get_stages_from_blocks(ws, rs):
  """Gets ws/ds of network at each stage from per block values."""
  ts_temp = zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
  ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
  s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
  s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
  return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
  """Generates per block ws from RegNet parameters."""
  assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
  ws_cont = np.arange(d) * w_a + w_0
  ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
  ws = w_0 * np.power(w_m, ks)
  ws = np.round(np.divide(ws, q)) * q
  num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
  ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
  return ws, num_stages, max_stage, ws_cont


def quantize_float(f, q):
  """Converts a float to closest non-zero int divisible by q."""
  return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
  """Adjusts the compatibility of widths and groups."""
  ws_bot = [int(w * b) for w, b in zip(ws, bms)]
  gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
  ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
  ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
  return ws, gs


def _get_conv2d(filters, kernel_size, **kwargs):
  """A helper function to create Conv2D layer."""
  return tf.keras.layers.Conv2D(
      filters=filters, kernel_size=kernel_size, **kwargs)


def _split_channels(total_filters, num_groups):
  split = [total_filters // num_groups for _ in range(num_groups)]
  split[0] += total_filters - sum(split)
  return split


class TpuBatchNormalization(tf.keras.layers.BatchNormalization):
  """Cross replica batch normalization."""

  def __init__(self, fused=False, **kwargs):
    if fused in (True, None):
      raise ValueError('TpuBatchNormalization does not support fused=True.')
    super(TpuBatchNormalization, self).__init__(fused=fused, **kwargs)

  def _cross_replica_average(self, t, num_shards_per_group):
    """Calculates the average value of input tensor across TPU replicas."""
    num_shards = tpu_function.get_tpu_context().number_of_shards
    group_assignment = None
    if num_shards_per_group > 1:
      if num_shards % num_shards_per_group != 0:
        raise ValueError('num_shards: %d mod shards_per_group: %d, should be 0'
                         % (num_shards, num_shards_per_group))
      num_groups = num_shards // num_shards_per_group
      group_assignment = [[
          x for x in range(num_shards) if x // num_shards_per_group == y
      ] for y in range(num_groups)]
    return tf.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
        num_shards_per_group, t.dtype)

  def _moments(self, inputs, reduction_axes, keep_dims):
    """Compute the mean and variance: it overrides the original _moments."""
    shard_mean, shard_variance = super(TpuBatchNormalization, self)._moments(
        inputs, reduction_axes, keep_dims=keep_dims)

    num_shards = tpu_function.get_tpu_context().number_of_shards or 1
    if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
      num_shards_per_group = 1
    else:
      num_shards_per_group = max(8, num_shards // 8)
    #logging.info('TpuBatchNormalization with num_shards_per_group %s',
    #             num_shards_per_group)
    if num_shards_per_group > 1:
      # Compute variance using: Var[X]= E[X^2] - E[X]^2.
      shard_square_of_mean = tf.math.square(shard_mean)
      shard_mean_of_square = shard_variance + shard_square_of_mean
      group_mean = self._cross_replica_average(
          shard_mean, num_shards_per_group)
      group_mean_of_square = self._cross_replica_average(
          shard_mean_of_square, num_shards_per_group)
      group_variance = group_mean_of_square - tf.math.square(group_mean)
      return (group_mean, group_variance)
    else:
      return (shard_mean, shard_variance)


class GroupedConv2D(object):
  """Grouped convolution.
  Currently tf.keras and tf.layers don't support group convolution, so here we
  use split/concat to implement this op. It reuses kernel_size for group
  definition, where len(kernel_size) is number of groups. Notably, it allows
  different group has different kernel size.
  """

  def __init__(self, filters, kernel_size, **kwargs):
    """Initialize the layer.
    Args:
      filters: Integer, the dimensionality of the output space.
      kernel_size: An integer or a list. If it is a single integer, then it is
        same as the original Conv2D. If it is a list, then we split the channels
        and perform different kernel for each group.
      **kwargs: other parameters passed to the original conv2d layer.
    """
    self._groups = len(kernel_size)
    self._channel_axis = -1

    self._convs = []
    splits = _split_channels(filters, self._groups)
    for i in range(self._groups):
      self._convs.append(
          _get_conv2d(splits[i], kernel_size[i], **kwargs))

  def __call__(self, inputs):
    if len(self._convs) == 1:
      return self._convs[0](inputs)

    if tf.__version__ < "2.0.0":
      filters = inputs.shape[self._channel_axis].value
    else:
      filters = inputs.shape[self._channel_axis]
    splits = _split_channels(filters, len(self._convs))
    x_splits = tf.split(inputs, splits, self._channel_axis)
    x_outputs = [c(x) for x, c in zip(x_splits, self._convs)]
    x = tf.concat(x_outputs, self._channel_axis)
    return x


class Stem(tf.keras.Model):
  def __init__(self, filters, input_shape, use_tpu):
    """Initialize a simple stem for AnyNet for ImageNet: Conv3x3, BN, ReLU
    
    Args:
      filters: number of filters to use for the conv layer.
      
    """
    super(Stem, self).__init__()
    self.use_tpu = use_tpu
    self.conv = tf.keras.layers.Conv2D(
        filters=filters, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
        use_bias=False, input_shape=input_shape)
    if use_tpu:
      self.bn = TpuBatchNormalization(axis=-1, epsilon=1.001e-5, momentum=0.9)
    else:
      self.bn = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5, momentum=0.9)
    self.relu = tf.keras.layers.Activation('relu')

  def call(self, inputs):
    """Implementation of call() for Stem.
    
    Args:
      inputs: input tensor.

    Returns:
      output tensor.
    
    """
    out = self.conv(inputs)
    out = self.bn(out)
    if self.use_tpu:
      out = tf.cast(out, tf.bfloat16)
    out = self.relu(out)
    return out


class SE(tf.keras.Model):
  """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

  def __init__(self, w_in, w_se):
    super(SE, self).__init__()
    self.channels = w_in
    self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')
    self.f_ex = tf.keras.Sequential([
      tf.keras.layers.Conv2D(
          filters=w_se, kernel_size=1, padding='valid', kernel_initializer='he_normal',
          activation='relu', use_bias=True),
      tf.keras.layers.Conv2D(
          filters=w_in, kernel_size=1, padding='valid', kernel_initializer='he_normal',
          activation='sigmoid', use_bias=True),
    ])

  def call(self, inputs):
    """Implementation of call() for Squeeze-Excitation block.
    
    Args:
      inputs: input tensor.

    Returns:
      output tensor.
    
    """
    # Adaptive average pooling.
    out = self.avg_pool(inputs)
    out = tf.reshape(out, [-1, 1, 1, self.channels])

    out = self.f_ex(out)
    out = inputs * out

    return out


class BottleneckTransform(tf.keras.Model):
  """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

  def __init__(self, w_in, w_out, strides, bm, gw, se_r, use_tpu):
    super(BottleneckTransform, self).__init__()
    self.use_tpu = use_tpu

    w_b = int(round(w_out * bm))
    g = w_b // gw
    
    self.conv1 = tf.keras.layers.Conv2D(
        filters=w_b, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
        use_bias=False)
    if use_tpu:
      self.bn = TpuBatchNormalization(axis=-1, epsilon=1.001e-5, momentum=0.9)
    else:
      self.bn = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5, momentum=0.9)
    self.relu = tf.keras.layers.Activation('relu')

    self.group_conv = GroupedConv2D(
        filters=w_b, kernel_size=[3]*g, strides=strides, padding='same',
        kernel_initializer='he_normal', use_bias=False)

    self.se = SE(w_in=w_b, w_se=int(round(w_in * se_r)))
    self.conv2 = tf.keras.layers.Conv2D(
        filters=w_out, kernel_size=1, strides=1, padding='valid', kernel_initializer='he_normal',
        use_bias=False)

  def call(self, inputs):
    """Implementation of call() for BottleneckTransform.
    
    Args:
      inputs: input tensor.

    Returns:
      output tensor.
    
    """
    out = self.conv1(inputs)
    out = self.bn(out)
    if self.use_tpu:
      out = tf.cast(out, tf.bfloat16)
    out = self.relu(out)

    out = self.group_conv(out)
    out = self.bn(out)
    if self.use_tpu:
      out = tf.cast(out, tf.bfloat16)
    out = self.relu(out)

    out = self.se(out)
    out = self.conv2(out)
    out = self.bn(out)
    if self.use_tpu:
      out = tf.cast(out, tf.bfloat16)
    return out
        

class ResBottleneckBlock(tf.keras.Model):
  """Residual bottleneck block (i.e. "X block" from paper): x + F(x), F = bottleneck transform."""

  def __init__(self, w_in, w_out, strides, bm, gw=1, se_r=None, use_tpu=False):
    """Initialize a stage for AnyNet.

    Uses Bottleneck block.

    Args:
      w_in: number of input channels.
      w_out: number of desired output channels for each block.
      strides: convolutional strides.
      bm: bottleneck multiplier for this stage.
      gw: group width for this stage.
      se_r: squeeze-excitation ratio.

    """
    super(ResBottleneckBlock, self).__init__()
    self.use_tpu = use_tpu

    # Use skip connection with projection if shape changes
    self.proj_block = (w_in != w_out) or (strides != 1)
    if self.proj_block:
      self.proj = tf.keras.layers.Conv2D(
        filters=w_out, kernel_size=1, strides=strides, padding='valid',
        kernel_initializer='he_normal', use_bias=False)
      if use_tpu:
        self.bn = TpuBatchNormalization(axis=-1, epsilon=1.001e-5, momentum=0.9)
      else:
        self.bn = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1.001e-5, momentum=0.9)

    self.transform = BottleneckTransform(w_in, w_out, strides, bm, gw, se_r, use_tpu)
    self.relu = tf.keras.layers.Activation('relu')

  def call(self, inputs):
    """Implementation of call() for ResBottleneckBlock.
    
    Args:
      inputs: input tensor.

    Returns:
      output tensor.
    
    """
    if self.proj_block:
      out = self.bn(self.proj(inputs))
      if self.use_tpu:
        out = tf.cast(out, tf.bfloat16)
      out += self.transform(inputs)
    else:
      out = inputs + self.transform(inputs)
    out = self.relu(out)
    return out


class AnyStage(tf.keras.Model):
  def __init__(self, w_in, w_out, strides, d, bm, gw, se_r, use_tpu, name):
    """Initialize a stage for AnyNet.

    Uses Bottleneck block.

    Args:
      w_in: number of input channels.
      w_out: number of desired output channels for each block.
      strides: convolutional strides.
      d: stage depth.
      bm: bottleneck multiplier for this stage.
      gw: group width for this stage.
      se_r: squeeze-excitation ratio.
      name: name for this stage.

    """
    super(AnyStage, self).__init__()
    self.stage = tf.keras.Sequential(name=name)
    for i in range(d):
      b_strides = strides if i == 0 else 1
      b_w_in = w_in if i == 0 else w_out
      self.stage.add(ResBottleneckBlock(b_w_in, w_out, b_strides, bm, gw, se_r, use_tpu))

  def call(self, inputs):
    """Implementation of call() for AnyStage.
    
    Args:
      inputs: input tensor.

    Returns:
      output tensor.
    
    """
    return self.stage(inputs)

  
class AnyHead(tf.keras.Model):
  """AnyNet head: AvgPool, 1x1."""

  def __init__(self, w_in, nc):
    """Initialize Head for AnyNet.
    
    Args:
      w_in: number of input channels.
      nc: number of classes.
    
    """
    super(AnyHead, self).__init__()
    self.channels = w_in
    # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    self.avg_pool = tf.keras.layers.GlobalAveragePooling2D(data_format='channels_last')
    # self.fc = nn.Linear(w_in, nc, bias=True)
    self.fc = tf.keras.layers.Dense(units=nc, kernel_initializer='he_normal', use_bias=True)

  def call(self, inputs):
    """Implementation of call() for AnyHead.
    
    Args:
      inputs: input tensor.

    Returns:
      output tensor.
    
    """
    out = self.avg_pool(inputs)
    out = tf.reshape(out, [-1, 1, 1, self.channels])
    out = self.fc(out)
    return out


class AnyNet(tf.keras.Model):
  def __init__(self, stem_w, ds, ws, ss, bms, gws, se_r, nc, input_shape, use_tpu):
    """Initialize an AnyNet model.

    By default uses a simple stem for ImageNet and a residual bottleneck block.

    Args:
      stem_w: width of stem.
      ds: depth for each stage (number of blocks in the stage).
      wd: width for each stage (width of each block in the stage).
      ss: strides for each stage (applies to the first block of each stage).
      bms: bottleneck multipliers for each stage (applies to bottleneck block).
      gws: group widths for each stage (applies to bottleneck block).
      se_r: squeeze-excitation ratio.
      nc: number of classes for the dataset.

    """
    super(AnyNet, self).__init__()
    stage_params = list(zip(ds, ws, ss, bms, gws))
    self.stem = Stem(filters=stem_w, input_shape=input_shape, use_tpu=use_tpu)
    prev_w = stem_w
    self.stage = tf.keras.Sequential()
    for i, (d, w, s, bm, gw) in enumerate(stage_params):
      name = 's{}'.format(i+1)
      self.stage.add(AnyStage(prev_w, w, s, d, bm, gw, se_r, use_tpu, name))
      prev_w = w
    self.head = AnyHead(w_in=prev_w, nc=nc)

  def call(self, inputs):
    """Implementation of call() for AnyNet.
    
    Args:
      inputs: input tensor.

    Returns:
      output tensor.
    
    """
    out = self.stem(inputs)
    out = self.stage(out)
    out = self.head(out)
    return out


class RegNet(AnyNet):
  @staticmethod
  def get_args(stem_w, w_a, w_0, w_m, d, se_r, nc):
    """Convert RegNet to AnyNet param format.
    
    Args:
      w_a: slope.
      w_0: initial width.
      w_m: quantization.
      d: depth.
      
    Returns:
      dictionary of AnyNet arguments defining the RegNet.
    """
    # Generate RegNet ws per block.
    ws, num_stages, _, _ = generate_regnet(w_a, w_0, w_m, d)

    # Convert to per stage format.
    s_ws, s_ds = get_stages_from_blocks(ws, ws)

    # Use the same gw, bm and ss for each stage.
    s_gs = [16 for _ in range(num_stages)] # group widths
    s_bs = [1.0 for _ in range(num_stages)] # bottleneck multipliers
    s_ss = [2 for _ in range(num_stages)] # strides

    # Adjust the compatibility of ws and gws.
    s_ws, s_gs = adjust_ws_gs_comp(s_ws, s_bs, s_gs)

    # Get AnyNet arguments defining the RegNet.
    return {
      'stem_w': stem_w,
      'ds': s_ds,
      'ws': s_ws,
      'ss': s_ss,
      'bms': s_bs,
      'gws': s_gs,
      'se_r': se_r,
      'nc': nc,
    }

  def __init__(self, stem_w, w_a, w_0, w_m, d, se_r, nc, input_shape, use_tpu):
    """Initialize a RegNet model.

    By default uses a simple stem for ImageNet and a residual bottleneck block.

    Args:
      stem_w: width of stem.
      ds: depth for each stage (number of blocks in the stage).
      wd: width for each stage (width of each block in the stage).
      ss: strides for each stage (applies to the first block of each stage).
      bms: bottleneck multipliers for each stage (applies to bottleneck block).
      gws: group widths for each stage (applies to bottleneck block).
      se_r: squeeze-excitation ratio.
      nc: number of classes for the dataset.

    """
    kwargs = RegNet.get_args(stem_w, w_a, w_0, w_m, d, se_r, nc)
    super(RegNet, self).__init__(**kwargs, input_shape=input_shape, use_tpu=use_tpu)

