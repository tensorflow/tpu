"""Contains definition for ResNeSt."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from dropblock import DropBlock2D
from splat import SplAtConv2D

from utils import TpuBatchNormalization


class Bottleneck(tf.keras.Model):
  """ResNet Bottleneck"""
  expansion = 4
  def __init__(
      self, filters, strides=1, downsample=None, radix=2, cardinality=1, bottleneck_width=64,
      avd=False, avd_first=False, dilation=1, is_first=False, norm_layer=None,
      dropblock_prob=0.0, use_tpu=False):
    """Initializes a Bottleneck block.
    
    Args:
      filters: number of filters (i.e. output channels) for conv layer.
      strides: convolutional stride.
      downsample: a tf.keras.Sequential of downsampling layers.
      radix: number of splits within a cardinal group.
      cardinality: number of cardinal groups (i.e. feature-map groups).
      bottleneck_width: default 64 so that number of filters is multiplied just by cardinality.
      avd: boolean to use average downsampling.
      avd_first: boolean to use average pooling layer before conv (used in fast setting).
      dilation: default 1 for classification tasks, >1 for segmentation tasks.
      is_first: previous dilation != current dilation.
      norm_layer: normalization layer used in backbone network.
      dropblock_prob: DropBlock keep probability.
    """
    super(Bottleneck, self).__init__()
    group_width = int(filters * (bottleneck_width / 64.)) * cardinality
    self.conv1 = tf.keras.layers.Conv2D(
        filters=group_width, kernel_size=1, strides=1, padding='same',
        kernel_initializer='he_normal', use_bias=False)
    self.bn1 = norm_layer(axis=-1, epsilon=1.001e-5)
    self.dropblock_prob = dropblock_prob
    self.radix = radix
    self.avd = avd and (strides > 1 or is_first)
    self.avd_first = avd_first
    self.use_tpu = use_tpu

    if self.avd:
      self.avd_layer = tf.keras.layers.AveragePooling2D(
          pool_size=3, strides=strides, padding='same')
      strides = 1

    if dropblock_prob > 0.0:
      self.dropblock1 = DropBlock2D(keep_prob=dropblock_prob, block_size=3)
      self.dropblock2 = DropBlock2D(keep_prob=dropblock_prob, block_size=3)
      self.dropblock3 = DropBlock2D(keep_prob=dropblock_prob, block_size=3)

    if radix >= 1:
      # using split-attention
      self.conv2 = SplAtConv2D(
          in_channels=group_width, channels=group_width, kernel_size=3,
          padding='same', dilation=dilation, groups=cardinality, use_bias=False, radix=radix,
          norm_layer=norm_layer, dropblock_prob=dropblock_prob, use_tpu=use_tpu)
    else:
      self.conv2 = tf.keras.layers.Conv2D(
          filters=group_width, kernel_size=3, strides=strides, padding='same',
          dilation_rate=dilation, use_bias=False)
      self.bn2 = norm_layer(axis=-1, epsilon=1.001e-5)

    self.conv3 = tf.keras.layers.Conv2D(
        filters=filters*4, kernel_size=1, strides=1, padding="same",
        kernel_initializer='he_normal', dilation_rate=dilation, use_bias=False)
    self.bn3 = norm_layer(axis=-1, epsilon=1.001e-5)

    self.relu = tf.keras.layers.Activation('relu')
    self.downsample = downsample
    self.dilation = dilation
    self.strides = strides

  def call(self, inputs):
    """Implementation of call() for Bottleneck.
    
    Args:
      inputs: input tensor.

    Returns:
      output tensor.
    
    """
    residual = inputs

    out = self.conv1(inputs)
    out = self.bn1(out)
    if self.use_tpu:
      out = tf.cast(out, tf.bfloat16)
    if self.dropblock_prob > 0.0:
      out = self.dropblock1(out)
    out = self.relu(out)

    if self.avd and self.avd_first:
      out = self.avd_layer(out)

    out = self.conv2(out)
    if self.radix >= 1:
      # using split-attention
      if self.dropblock_prob > 0.0:
        out = self.dropblock2(out)
    else:
      out = self.bn2(out)
      if self.use_tpu:
        out = tf.cast(out, tf.bfloat16)
      if self.dropblock_prob > 0.0:
        out = self.dropblock2(out)
      out = self.relu(out)

    if self.avd and not self.avd_first:
      out = self.avd_layer(out)

    out = self.conv3(out)
    out = self.bn3(out)
    if self.use_tpu:
      out = tf.cast(out, tf.bfloat16)
    if self.dropblock_prob > 0.0:
      out = self.dropblock3(out)

    if self.downsample is not None:
      residual = self.downsample(inputs)

    out += residual
    out = self.relu(out)

    return out


class ResNet(tf.keras.Model):
  def __init__(
      self, block, layers, input_shape=(224, 224, 3), radix=2, groups=1, bottleneck_width=64,
      num_classes=1000, dilated=False, dilation=1, deep_stem=True, stem_width=64,
      avg_down=True, avd=True, avd_first=False, final_drop=0.2, dropblock_prob=0, use_tpu=False):
    """Initializes a ResNet variant (default: ResNeSt).

    Args:
      block: class for residual block (e.g. Bottleneck).
      layers: list of 4 integers indicating number of blocks in each layer. 
      input_shape: (H, W, C) of input.
      radix: number of splits within a cardinal group.
      groups: number of cardinal groups (i.e. feature-map groups).
      bottleneck_width: default 64 so that number of filters is multiplied just by cardinality.
      num_classes: number of classification buckets in dataset.
      dilated: boolean if applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation (default: False).
      dilation: default 1 for classification tasks, >1 for segmentation tasks.
      deep_stem: boolean to replace the usual single conv7x7 (64 filters, stride 2) stem with:
        conv3x3 (stride 2, stem_width filters),
        batchnorm,
        relu,
        conv3x3 (stem_width filters, stride 1),
        batchnorm,
        relu,
        conv3x3 (stem_width*2 filters, stride 1)
      stem_width: width of beginning of deep stem.
      avg_down: boolean to use average downsampling in shortcut connection (default: True).
      avd: boolean to use average downsampling in residual block (default: True).
      avd_first: boolean to use average pooling layer before conv (default: False).
        Used in fast setting.
      final_drop: dropout layer keep probability.
      dropblock_prob: DropBlock keep probability.
      use_tpu: boolean if running on TPU.
    """
    self.cardinality = groups
    self.bottleneck_width = bottleneck_width
    # ResNet-D params
    self.in_channels = stem_width*2 if deep_stem else 64
    self.deep_stem = deep_stem
    self.avg_down = avg_down
    # ResNeSt params
    self.radix = radix
    self.avd = avd
    self.avd_first = avd_first
    self.use_tpu = use_tpu
    super(ResNet, self).__init__()

    conv_layer = tf.keras.layers.Conv2D
    norm_layer = TpuBatchNormalization if use_tpu else tf.keras.layers.BatchNormalization
    if deep_stem:
      self.conv_stem_1 = conv_layer(
          filters=stem_width, kernel_size=3, strides=2, padding='same',
          kernel_initializer='he_normal', use_bias=False, input_shape=input_shape)
      self.bn_stem_1 = norm_layer(axis=-1, epsilon=1.001e-5)
      self.relu_stem_1 = tf.keras.layers.Activation('relu')
      self.conv_stem_2 = conv_layer(
          filters=stem_width, kernel_size=3, strides=1, padding='same',
          kernel_initializer='he_normal', use_bias=False)
      self.bn_stem_2 = norm_layer(axis=-1, epsilon=1.001e-5)
      self.relu_stem_2 = tf.keras.layers.Activation('relu')
      self.conv_stem_3 = conv_layer(
          filters=stem_width*2, kernel_size=3, strides=1, padding='same',
          kernel_initializer='he_normal', use_bias=False)                
    else:
      self.conv_stem_1 = conv_layer(
          filters=64, kernel_size=7, strides=2, padding='same', use_bias=False,
          input_shape=input_shape)

    self.bn1 = norm_layer(axis=-1, epsilon=1.001e-5)
    self.relu = tf.keras.layers.Activation('relu')
    self.maxpool = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')

    self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
    self.layer2 = self._make_layer(block, 128, layers[1], strides=2, norm_layer=norm_layer)
    if dilated or dilation == 4:
      self.layer3 = self._make_layer(
          block=block, filters=256, num_blocks=layers[2], strides=1, dilation=2,
          norm_layer=norm_layer, dropblock_prob=dropblock_prob)
      self.layer4 = self._make_layer(
          block=block, filters=512, num_blocks=layers[3], strides=1, dilation=4,
          norm_layer=norm_layer, dropblock_prob=dropblock_prob)
    elif dilation==2:
      self.layer3 = self._make_layer(
          block=block, filters=256, num_blocks=layers[2], strides=2, dilation=1,
          norm_layer=norm_layer, dropblock_prob=dropblock_prob)
      self.layer4 = self._make_layer(
          block=block, filters=512, num_blocks=layers[3], strides=1, dilation=2,
          norm_layer=norm_layer, dropblock_prob=dropblock_prob)
    else:
      self.layer3 = self._make_layer(
          block=block, filters=256, num_blocks=layers[2], strides=2, norm_layer=norm_layer,
          dropblock_prob=dropblock_prob)
      self.layer4 = self._make_layer(
          block=block, filters=512, num_blocks=layers[3], strides=2, norm_layer=norm_layer,
          dropblock_prob=dropblock_prob)

    self.avgpool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')
    self.flatten = tf.keras.layers.Flatten()
    self.drop = tf.keras.layers.Dropout(
        final_drop, noise_shape=None) if final_drop > 0.0 else None
    self.fc = tf.keras.layers.Dense(
        units=num_classes, kernel_initializer="he_normal", use_bias=False, name="fc")

  def _make_layer(
      self, block, filters, num_blocks, strides=1, dilation=1, norm_layer=None,
      dropblock_prob=0.0, is_first=True):
    """Creates a layer of blocks for the ResNet.
    
    Args:
      block: class for residual block (e.g. Bottleneck).
      filters: number of filters (i.e. output channels) for conv layer.
      num_blocks: number of blocks to be used in this layer.
      strides: convolutional stride.
      dilation: default 1 for classification tasks, >1 for segmentation tasks.
      norm_layer: normalization layer used in backbone network.
      dropblock_prob: DropBlock keep probability.
      is_first: previous dilation != current dilation.

    Returns:
      a tf.keras.Sequential of blocks.
    
    """
    downsample = None
    if strides != 1 or self.in_channels != filters * block.expansion:
        down_layers = []
        if self.avg_down:
          if dilation == 1:
            down_layers.append(tf.keras.layers.AveragePooling2D(
                pool_size=strides, strides=strides, padding='same'))
          else:
              down_layers.append(tf.keras.layers.AveragePooling2D(
                  pool_size=1, strides=1, padding='same'))
          down_layers.append(tf.keras.layers.Conv2D(
              filters * block.expansion, kernel_size=1, strides=1, padding='same',
              kernel_initializer='he_normal', use_bias=False))
        else:
          down_layers.append(tf.keras.layers.Conv2D(
              filters * block.expansion, kernel_size=1, strides=stride, padding='same',
              kernel_initializer='he_normal', use_bias=False))
        down_layers.append(norm_layer(
            axis=-1, epsilon=1.001e-5))
        downsample = tf.keras.Sequential(down_layers)

    blocks = []
    if dilation == 1 or dilation == 2:
      blocks.append(block(
          filters=filters, strides=strides, downsample=downsample, radix=self.radix,
          cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd,
          avd_first=self.avd_first, dilation=1, is_first=is_first, norm_layer=norm_layer,
          dropblock_prob=dropblock_prob))
    elif dilation == 4:
      blocks.append(block(
          filters=filters, strides=strides, downsample=downsample, radix=self.radix,
          cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd,
          avd_first=self.avd_first, dilation=2, is_first=is_first, norm_layer=norm_layer,
          dropblock_prob=dropblock_prob))
    else:
      raise RuntimeError("=> unknown dilation size: {}".format(dilation))

    for i in range(1, num_blocks):
      blocks.append(block(
          filters=filters, strides=1, downsample=None, radix=self.radix,
          cardinality=self.cardinality, bottleneck_width=self.bottleneck_width, avd=self.avd,
          avd_first=self.avd_first, dilation=dilation, norm_layer=norm_layer,
          dropblock_prob=dropblock_prob))

    return tf.keras.Sequential(blocks)

  def call(self, inputs):
    """Implementation of call() for ResNet.
    
    Args:
      inputs: input tensor.

    Returns:
      output tensor.
    
    """
    if self.deep_stem:
      out = self.conv_stem_1(inputs)
      out = self.bn_stem_1(out)
      if self.use_tpu:
        out = tf.cast(out, tf.bfloat16)
      out = self.relu_stem_1(out)
      out = self.conv_stem_2(out)
      out = self.bn_stem_2(out)
      if self.use_tpu:
        out = tf.cast(out, tf.bfloat16)
      out = self.relu_stem_2(out)
      out = self.conv_stem_3(out)
    else:
      out = self.conv_stem_1(inputs)
    
    out = self.bn1(out)
    if self.use_tpu:
      out = tf.cast(out, tf.bfloat16)
    out = self.relu(out)
    out = self.maxpool(out)

    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)

    out = self.avgpool(out)
    out = self.flatten(out)
    if self.drop:
      out = self.drop(out)
    out = self.fc(out)

    return out


def resnest50(input_shape=(224, 224, 3), num_classes=1000, use_tpu=False, **kwargs):
    """ResNeSt-50 config."""
    model = ResNet(Bottleneck, layers=[3, 4, 6, 3], input_shape=input_shape,
                   radix=2, groups=1, bottleneck_width=64, num_classes=1000,
                   deep_stem=True, stem_width=32, avg_down=True,
                   avd=True, avd_first=False, use_tpu=use_tpu, **kwargs)
    return model

def resnest101(input_shape=(224, 224, 3), num_classes=1000, use_tpu=False, **kwargs):
    """ResNeSt-101 config."""
    model = ResNet(Bottleneck, layers=[3, 4, 23, 3], input_shape=input_shape,
                   radix=2, groups=1, bottleneck_width=64, num_classes=1000,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, use_tpu=use_tpu, **kwargs)
    return model

def resnest200(input_shape=(224, 224, 3), num_classes=1000, use_tpu=False, **kwargs):
    """ResNeSt-200 config."""
    model = ResNet(Bottleneck, layers=[3, 24, 36, 3], input_shape=input_shape,
                   radix=2, groups=1, bottleneck_width=64, num_classes=1000,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, use_tpu=use_tpu, **kwargs)
    return model

def resnest269(input_shape=(224, 224, 3), num_classes=1000, use_tpu=False, **kwargs):
    """ResNeSt-269 config."""
    model = ResNet(Bottleneck, layers=[3, 30, 48, 8], input_shape=input_shape,
                   radix=2, groups=1, bottleneck_width=64, num_classes=1000,
                   deep_stem=True, stem_width=64, avg_down=True,
                   avd=True, avd_first=False, use_tpu=use_tpu, **kwargs)
    return model
