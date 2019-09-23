# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Implements the QANet question answering model for SQuAD."""
import os
# Standard Imports
import numpy as np
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf
import data
import utils


def build_config(model_dir, data_path):
  return utils.Config(
      # Practicalities
      model_dir=model_dir,
      master='',

      # Data
      num_epochs=30,
      # If 0, run until OurOfRange is raised by dataset.
      # Must be nonzero on TPU.
      steps_per_epoch=0,
      dataset=utils.Config(
          train_split='train',
          eval_split='dev',
          load_tfrecord=False,  # If false, generate on the fly
          # Number of times to repeat dataset per call.
          # If 0, repeat indefinitely.
          num_repeats=1,
          train_shuffle=True,
          cache=True,
          max_length=256,
          data_path=data_path,
          vocab_path=os.path.join(data_path, 'vocab.vec'),
          train_batch_size=32,
          eval_batch_size=16,
          resample_too_long=True,
      ),

      # Optimizer
      opt=utils.Config(
          lr=0.001,
          warmup_steps=1000,
          beta1=0.9,
          beta2=0.999,
          epsilon=1e-7,
          grad_clip_norm=10.0,
          l2_reg=5e-8,
          ema_decay=0.999),
      # Encoder structure
      encoder_emb=utils.Config(
          layers=1,
          kernel_size=7,
          hidden_size=128,
          ffn_multiplier=2.0,
          attention_heads=4,
          attention_dropout=0.1,
          layer_dropout=0.25,
          attention_type='dot_product',
          structure='conv,conv,conv,conv,att,ffn',
          separable_conv=True,
          timing_signal=True,
      ),
      encoder_model=utils.Config(
          layers=1,
          hidden_size=128,
          kernel_size=7,
          ffn_multiplier=2.0,
          attention_heads=4,
          attention_dropout=0.1,
          layer_dropout=0.25,
          attention_type='dot_product',
          structure='conv,conv,att,ffn',
          separable_conv=True,
          timing_signal=True,
      ),
      output_dropout_rate=0.3,
      embedding=utils.Config(dropout_rate=0.15),
      tpu=utils.Config(enable=False,),
  )


def get_loss(answer_start,
             answer_end,
             logits_start,
             logits_end,
             label_smoothing=0.0):
  """Get loss given answer and logits.

  Args:
    answer_start: [batch_size, num_answers] shaped tensor
    answer_end: Similar to `answer_start` but for end.
    logits_start: [batch_size, context_size]-shaped tensor for answer start
      logits.
    logits_end: Similar to `logits_start`, but for end. This tensor can be also
      [batch_size, context_size, context_size], in which case the true answer
      start is used to index on dim 1 (context_size).
    label_smoothing: whether to use label smoothing or not.
  Returns:
    Float loss tensor.
  """
  length = logits_start.shape.as_list()[1] or tf.shape(logits_start)[1]
  start = tf.one_hot(answer_start, length)
  end = tf.one_hot(answer_end, length)
  loss_start = tf.losses.softmax_cross_entropy(
      onehot_labels=start, logits=logits_start, label_smoothing=label_smoothing)
  loss_end = tf.losses.softmax_cross_entropy(
      onehot_labels=end, logits=logits_end, label_smoothing=label_smoothing)

  return tf.reduce_mean(loss_start) + tf.reduce_mean(loss_end)


def bi_attention_memory_efficient_dcn(
    a,
    b,
    mask_a=None,
    # TODO(ddohan): Should the other mask be used?
    mask_b=None,  # pylint: disable=unused-argument
):
  """Biattention between question (a) and document (b)."""
  logits = tf.transpose(
      trilinear_memory_efficient(a, b), perm=[0, 2, 1])  # [bs,len_b,len_a]
  b2a = b2a_attention(logits, a, mask_a)
  a2b = a2b_attention_dcn(logits, b)
  return b2a, a2b


def trilinear_memory_efficient(a, b, use_activation=False):
  """W1a + W2b + aW3b."""
  d = a.shape.as_list()[-1]
  n = tf.shape(a)[0]

  len_a = tf.shape(a)[1]
  len_b = tf.shape(b)[1]

  w1 = tf.get_variable('w1', shape=[d, 1], dtype=tf.float32)
  w2 = tf.get_variable('w2', shape=[d, 1], dtype=tf.float32)
  w3 = tf.get_variable('w3', shape=[1, 1, d], dtype=tf.float32)

  a_reshape = tf.reshape(a, [-1, d])  # [bs*len_a, d]
  b_reshape = tf.reshape(b, [-1, d])  # [bs*len_b, d]

  part_1 = tf.reshape(tf.matmul(a_reshape, w1), [n, len_a])  # [bs, len_a]
  part_1 = tf.tile(tf.expand_dims(part_1, 2),
                   [1, 1, len_b])  # [bs, len_a, len_b]

  part_2 = tf.reshape(tf.matmul(b_reshape, w2), [n, len_b])  # [bs, len_b]
  part_2 = tf.tile(tf.expand_dims(part_2, 1),
                   [1, len_a, 1])  # [bs, len_a, len_b]

  a_w3 = a * w3  # [bs, len_a, d]
  part_3 = tf.matmul(a_w3, tf.transpose(b, perm=[0, 2, 1]))  # [bs,len_a,len_b]

  ## return the unnormalized logits matrix : [bs,len_a,len_b]
  if use_activation:
    return tf.nn.relu(part_1 + part_2 + part_3)
  return part_1 + part_2 + part_3


def b2a_attention(b, a, mask_a=None):
  """Attention of document (b) over question (a).

  Args:
    b: [bs, len_b, depth]
    a: [bs, len_a, depth]
    mask_a: Mask over elem

  Returns:
    logits: [batch, len_b, len_a]

  """
  if len(mask_a.get_shape()) == 1:
    mask_a = tf.sequence_mask(mask_a, tf.shape(a)[1])
  if len(mask_a.get_shape()) == 2:
    mask_a = tf.expand_dims(mask_a, 1)
  logits = exp_mask(b, mask_a, mask_is_length=False)
  probabilities = tf.nn.softmax(logits)  # [bs,len_b,len_a]
  b2a = tf.matmul(probabilities, a)  # [bs, len_b, d]
  return b2a


def a2b_attention_dcn(logits, b):
  """Attention of question (a) over document (b).

  Args:
    logits: [bs, len_b, len_a] tensor
    b: [bs, len_b, depth] tensor

  Returns:
    logits: [batch, len_b, depth]

  """
  prob1 = tf.nn.softmax(logits)  # [bs,len_b,len_a]
  prob2 = tf.nn.softmax(tf.transpose(logits, perm=[0, 2,
                                                   1]))  # [bs,len_a,len_b]
  a2b = tf.matmul(tf.matmul(prob1, prob2), b)  # [bs,len_b,d]
  return a2b


VERY_LARGE_NEGATIVE_VALUE = -1e12


def exp_mask(logits, mask, mask_is_length=True):
  """Exponential mask for logits.

  Logits cannot be masked with 0 (i.e. multiplying boolean mask)
  because expnentiating 0 becomes 1. `exp_mask` adds very large negative value
  to `False` portion of `mask` so that the portion is effectively ignored
  when exponentiated, e.g. softmaxed.

  Args:
    logits: Arbitrary-rank logits tensor to be masked.
    mask: `boolean` type mask tensor.
      Could be same shape as logits (`mask_is_length=False`)
      or could be length tensor of the logits (`mask_is_length=True`).
    mask_is_length: `bool` value. whether `mask` is boolean mask.
  Returns:
    Masked logits with the same shape of `logits`.
  """
  if mask_is_length:
    mask = tf.sequence_mask(mask, maxlen=tf.shape(logits)[-1])
  return logits + (1.0 - tf.cast(mask, 'float')) * VERY_LARGE_NEGATIVE_VALUE


# Metrics


def get_answer_op(context, context_words, answer_start, answer_end):
  return tf.py_func(
      data.enum_fn(data.get_answer_tokens),
      [context, context_words, answer_start, answer_end], 'string')


# token level - Empirically 870 in train set, 700 in dev
MAX_CONTEXT_SIZE = 900


def get_predictions(context,
                    context_tokens,
                    logits_start,
                    logits_end,
                    max_answer_size=30):
  """Get prediction op dictionary given start & end logits.

  This dictionary will contain predictions as well as everything needed
  to produce the nominal answer and identifier (ids).

  Args:
    context: The original context string
    context_tokens: Tokens to index into
    logits_start: [batch_size, context_size]-shaped tensor of start logits
    logits_end: [batch_size, context_size]-shaped tensor of end logits
    max_answer_size: Maximum length (in tokens) of a valid answer.
  Returns:
    A dictionary of prediction tensors.
  """
  prob_start = tf.nn.softmax(logits_start)
  prob_end = tf.nn.softmax(logits_end)

  max_x_len = tf.shape(context_tokens)[1]
  # This is only for computing span accuracy and not used for training.
  # Masking with `upper_triangular_matrix` only allows valid spans,
  # i.e. `answer_pred_start` <= `answer_pred_end`.
  upper_tri_mat = tf.slice(
      np.triu(
          np.ones([MAX_CONTEXT_SIZE, MAX_CONTEXT_SIZE], dtype='float32') -
          np.triu(
              np.ones([MAX_CONTEXT_SIZE, MAX_CONTEXT_SIZE], dtype='float32'),
              k=max_answer_size)), [0, 0], [max_x_len, max_x_len])
  # Outer product
  prob_mat = tf.expand_dims(prob_start, -1) * tf.expand_dims(prob_end, 1)

  # Mask out
  prob_mat *= tf.expand_dims(upper_tri_mat, 0)

  answer_pred_start = tf.argmax(tf.reduce_max(prob_mat, 2), 1)
  answer_pred_end = tf.argmax(tf.reduce_max(prob_mat, 1), 1)
  answer_prob = tf.reduce_max(prob_mat, [1, 2])

  predictions = {
      'yp1': answer_pred_start,
      'yp2': answer_pred_end,
      'p1': prob_start,
      'p2': prob_end,
      'answer_prob': answer_prob,
  }

  answer = get_answer_op(
      context,
      context_tokens,
      answer_pred_start,
      answer_pred_end,
  )
  predictions['answer'] = answer
  return predictions


def get_attention_bias(sequence_length, maxlen=None):
  """Create attention bias so attention is not applied at padding position."""
  # attention_bias: [batch, 1, 1, memory_length]
  mask = tf.sequence_mask(sequence_length, maxlen=maxlen)
  nonpadding = tf.to_float(mask)
  invert_sequence_mask = tf.to_float(tf.logical_not(mask))
  attention_bias = common_attention.attention_bias_ignore_padding(
      invert_sequence_mask)
  return nonpadding, attention_bias


def separable_conv(x, filters, kernel_size, activation):
  """Apply a depthwise separable 1d convolution."""
  tf.assert_rank(x, 3)
  net = tf.expand_dims(x, 2)
  net = tf.layers.separable_conv2d(
      net,
      filters=filters,
      kernel_size=(kernel_size, 1),
      padding='same',
      activation=activation)
  net = tf.squeeze(net, axis=2)
  return net


def sequence_encoder(inputs, length, is_training, cfg):
  """Encode a sequence using self attention, convolutions, and dense layers.

  Args:
    inputs: [batch x length x depth] tensor to encode
    length: [batch] tensor containing length of each sequence as an int
    is_training: bool indicating whether we are training
    cfg: Layer configuration

  Returns:
    Encoded sequence

  Raises:
    ValueError: If cfg.structure is invalid.
  """
  cfg = utils.Config(cfg)
  assert length is not None
  assert is_training in [False, True]

  # Turn off dropout at test time.
  if not is_training:
    for k in cfg:
      if 'dropout' in k:
        cfg[k] = 0.0

  # Mask out padding tokens during attention
  maxlen = None
  if is_training:
    # All dimensions must be static on a TPU
    maxlen = inputs.shape.as_list()[1]
  _, attention_bias = get_attention_bias(length, maxlen=maxlen)

  if inputs.shape.as_list()[-1] != cfg.hidden_size:
    # Project to internal size
    inputs = common_layers.conv1d(
        inputs=inputs,
        filters=cfg.hidden_size,
        kernel_size=1,
        activation=None,
        padding='SAME')
  net = inputs
  if cfg.timing_signal:
    net = common_attention.add_timing_signal_nd(net)
  structure = cfg.structure.split(',') * cfg.layers
  for layer_id, layer_type in enumerate(structure):
    with tf.variable_scope('%s_%d' % (layer_type, layer_id)):
      layer_input = net
      net = common_layers.layer_norm(net)
      if layer_type == 'att':
        net = common_attention.multihead_attention(
            query_antecedent=net,
            memory_antecedent=None,
            bias=attention_bias,
            total_key_depth=cfg.hidden_size,
            total_value_depth=cfg.hidden_size,
            output_depth=cfg.hidden_size,
            num_heads=cfg.attention_heads,
            dropout_rate=cfg.attention_dropout,
            attention_type=cfg.attention_type,
            make_image_summary=False)
      elif layer_type == 'conv':
        if cfg.separable_conv:
          net = separable_conv(
              net,
              filters=cfg.hidden_size,
              kernel_size=cfg.kernel_size,
              activation=tf.nn.relu)
        else:
          net = common_layers.conv1d(
              inputs=net,
              filters=cfg.hidden_size,
              kernel_size=cfg.kernel_size,
              activation=tf.nn.relu,
              padding='SAME')
      elif layer_type == 'ffn':
        # TODO(ddohan): See how expert_utils used to do the dense layer
        net = tf.layers.dense(
            net,
            units=int(cfg.ffn_multiplier * cfg.hidden_size),
            activation=tf.nn.relu)
        net = tf.layers.dense(net, units=cfg.hidden_size, activation=None)
      else:
        raise ValueError('Unknown layer type %s' % layer_type)

      if cfg.layer_dropout:
        net = tf.nn.dropout(net, keep_prob=1.0 - cfg.layer_dropout)
      net += layer_input
  net = common_layers.layer_norm(net)
  return net


def create_eval_scaffold_fn(ema_decay, model_dir):
  """Returns scaffold function for evaluation step."""

  def scaffold_fn():
    """Scaffold function for Estimator, used for restoring EMA vars."""
    # If we used EMA for training, load the exponential moving
    # average variables for evaluations.  Otherwise, load the
    # normal variables.
    ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
    var_dict = ema.variables_to_restore(tf.trainable_variables())
    tf.logging.info(model_dir)
    if model_dir is None:
      saver_filename = None
    else:
      saver_filename = os.path.join(model_dir, 'model_saver_file')
    saver = tf.train.Saver(var_dict,
                           max_to_keep=2,
                           filename=saver_filename)
    return tf.train.Scaffold(saver=saver)

  return scaffold_fn


def build_train_op(loss, is_tpu, opt_cfg, trainable_vars=None):
  """Build the optimizer, compute gradients, and build training op."""
  trainable_vars = trainable_vars or tf.trainable_variables()

  def decay_fn(learning_rate, global_step):
    # Linear warmup from 0.

    lr_decay = tf.minimum(1.0, tf.to_float(global_step) / opt_cfg.warmup_steps)
    return lr_decay * learning_rate

  def optimizer(lr):
    opt = tf.train.AdamOptimizer(
        learning_rate=lr,
        beta1=opt_cfg.beta1,
        beta2=opt_cfg.beta2,
        epsilon=opt_cfg.epsilon)
    if is_tpu:
      opt = tf.contrib.tpu.CrossShardOptimizer(opt)
    return opt

  if opt_cfg.l2_reg:
    tf.logging.info('Applying l2 regularization of %s', opt_cfg.l2_reg)
    decay_costs = []
    for var in trainable_vars:
      decay_costs.append(tf.nn.l2_loss(var))

    loss += tf.multiply(opt_cfg.l2_reg, tf.add_n(decay_costs))

  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.train.get_global_step(),
      learning_rate=opt_cfg.lr,
      learning_rate_decay_fn=decay_fn,
      clip_gradients=opt_cfg.grad_clip_norm,
      optimizer=optimizer,
      variables=trainable_vars,
      gradient_noise_scale=None,
      summaries=[] if is_tpu else [
          'learning_rate',
          'loss',
      ])

  if opt_cfg.ema_decay < 1.0:
    ema = tf.train.ExponentialMovingAverage(decay=opt_cfg.ema_decay)
    maintain_average_op = ema.apply(trainable_vars)
    with tf.control_dependencies([train_op]):
      train_op = tf.group(maintain_average_op)

  return train_op


def model_fn(
    features,  # This is batch_features from input_fn
    labels,    # This is batch_labels from input_fn
    mode,      # An instance of tf.estimator.ModeKeys
    params):   # Additional configuration
  """Model function defining a QANet model."""
  is_training = mode == 'train'
  cfg = utils.Config(params)

  def make_encoder(encoder_cfg, name):

    def call_encoder(inputs, length):
      return sequence_encoder(
          inputs=inputs,
          length=length,
          is_training=is_training,
          cfg=encoder_cfg)

    return tf.make_template(name, call_encoder)

  xl = features['context_length']
  ql = features['question_length']

  x = features['context_vecs']
  q = features['question_vecs']

  def emb_dropout(net):
    return tf.layers.dropout(
        net, rate=cfg.embedding.dropout_rate, training=is_training)

  def output_dropout(net):
    return tf.layers.dropout(
        net, rate=cfg.output_dropout_rate, training=is_training)

  encoder_emb = make_encoder(cfg.encoder_emb, 'encoder_emb')
  encoder_model = make_encoder(cfg.encoder_model, 'encoder_model')

  def encode_emb(net, length):
    net = emb_dropout(net)
    net = encoder_emb(net, length)
    net = emb_dropout(net)
    return net

  x = encode_emb(x, xl)
  q = encode_emb(q, ql)

  with tf.variable_scope('attention'):
    xq, qx = bi_attention_memory_efficient_dcn(a=q, b=x, mask_a=ql, mask_b=xl)

  net = tf.concat([x, xq, x * xq, x * qx], 2)
  net = output_dropout(encoder_model(net, xl))
  x_enc = output_dropout(encoder_model(net, xl))
  start = output_dropout(encoder_model(x_enc, xl))
  end = output_dropout(encoder_model(start, xl))

  logits_start = exp_mask(
      tf.squeeze(
          tf.layers.dense(tf.concat([x_enc, start], 2), 1, name='logits_start'),
          2), xl)
  logits_end = exp_mask(
      tf.squeeze(
          tf.layers.dense(tf.concat([x_enc, end], 2), 1, name='logits_end'), 2),
      xl)
  predictions = {
      'logits_start': logits_start,
      'logits_end': logits_end,
  }

  starts = features['answers_start_token']
  ends = features['answers_end_token']
  loss = get_loss(
      answer_start=starts[:, 0],
      answer_end=ends[:, 0],
      logits_start=logits_start,
      logits_end=logits_end)

  # Eval never runs on TPU
  if not is_training:
    orig_context = features['context_tokens']
    predictions['id'] = features['id']
    predictions.update(
        get_predictions(
            context=features['context'],
            context_tokens=orig_context,
            logits_start=logits_start,
            logits_end=logits_end))

  train_op = None
  scaffold_fn = tf.train.Scaffold
  if is_training:
    train_op = build_train_op(loss, is_tpu=cfg.tpu.enable, opt_cfg=cfg.opt)

  eval_metrics = None
  if mode == 'eval':
    if cfg.opt.ema_decay < 1.0:
      scaffold_fn = create_eval_scaffold_fn(cfg.opt.ema_decay, cfg.model_dir)
    eval_metrics = (data.metric_fn,
                    dict(
                        answers=labels['answers'],
                        prediction=predictions['answer'],
                        start=starts,
                        end=ends,
                        yp1=predictions['yp1'],
                        yp2=predictions['yp2'],
                        num_answers=labels['num_answers']))

  # Run eval on CPU/GPU always
  if cfg.tpu.enable and is_training:
    return tf.contrib.tpu.TPUEstimatorSpec(
        mode,
        loss=loss,
        predictions=predictions,
        train_op=train_op,
        eval_metrics=eval_metrics,
        scaffold_fn=scaffold_fn)
  else:
    eval_metric_ops = None
    if mode == 'eval':
      eval_metric_ops = eval_metrics[0](**eval_metrics[1])

    return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        predictions=predictions,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        scaffold=scaffold_fn())


def get_estimator(**kwargs):
  """Construct an estimator."""
  cfg = utils.Config(kwargs)

  if cfg.tpu.get('name'):
    tf.logging.info('Using cluster resolver.')
    cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        cfg.tpu.name, zone=cfg.tpu.zone, project=cfg.tpu.gcp_project)
    master = None
  else:
    cluster_resolver = None
    master = cfg.master

  tf.logging.info('Config:\n %s' % cfg)
  if cfg.tpu.enable:
    if not cfg.steps_per_epoch:
      raise ValueError('steps_per_epoch must be nonzero on TPU.')
    exp = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        config=tf.contrib.tpu.RunConfig(
            cluster=cluster_resolver,
            master=master,
            model_dir=cfg.model_dir,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=cfg.steps_per_epoch)),
        use_tpu=True,
        eval_on_tpu=False,
        # TPU requires these args, but they are ignored inside the input
        # function, which directly get train_batch_size or eval_batch_size.
        train_batch_size=cfg.dataset.train_batch_size,
        eval_batch_size=cfg.dataset.eval_batch_size,
        params=cfg,
    )
  else:
    exp = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=cfg.model_dir, params=cfg)

  return exp
