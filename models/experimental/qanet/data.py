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

"""Implements data loaders and metrics for the SQuAD dataset."""
import collections
import json
import os
import re
import string
# Standard Imports
import numpy as np
import tensorflow.compat.v1 as tf


def build_dataset(cfg, is_tpu):
  """Construct train and eval inputs_fn."""
  load_tfrecord = cfg.load_tfrecord
  if is_tpu:
    load_tfrecord = True
  # TODO(ddohan): Share the common args more clearly
  train_input = get_input_fn(
      split=cfg.train_split,
      max_length=cfg.max_length,
      # TPUs don't handle OutOfRange exceptions from data pipelines, so we
      # repeat indefinitely and handle setting number of training steps
      # manually. This is handled by the tpu.steps_per_epoch setting.
      # On a GPU, we are able to be more exact about the exact boundary between
      # epochs and avoid reasoning in terms of step counts.
      # If 0, repeat indefinitely. Otherwise repeat N times.
      num_repeats=0 if is_tpu else cfg.num_repeats,
      shuffle=cfg.train_shuffle,
      cache=cfg.cache,
      limit=None,
      data_path=cfg.data_path,
      vocab_path=cfg.vocab_path,
      is_tpu=is_tpu,
      use_generator=not load_tfrecord,
      resample_too_long=cfg.resample_too_long,
      is_training=True)
  eval_input = get_input_fn(
      split=cfg.eval_split,
      max_length=None,  # Never do any filtering at eval
      limit=None,
      num_repeats=1,
      shuffle=False,
      cache=cfg.cache,
      data_path=cfg.data_path,
      vocab_path=cfg.vocab_path,
      is_tpu=False,  # Never eval on TPU because of py_func
      use_generator=not load_tfrecord,
      is_training=False)
  return train_input, eval_input


def word_tokenize(text):
  """Split on whitespace and punctuation."""
  return re.findall(r'\w+|[^\w\s]', text, re.UNICODE)


def utf_encode_list(text):
  """utf encode every element of a list."""
  return [x.encode('utf-8') for x in text]


def convert_to_spans(raw_text, tokens):
  """Convert tokenized version of `raw_text` to character spans into it.

  Args:
    raw_text: The raw string
    tokens: The tokenized version of the string


  Returns:
    [list of (start, end) tuples] mapping each token to corresponding indices
    in the text.
  """
  cur_idx = 0
  spans = []
  for token in tokens:
    tmp = raw_text.find(token, cur_idx)
    l = len(token)
    cur_idx = tmp
    spans.append((cur_idx, cur_idx + l))
    cur_idx += l
  return spans


def get_answer_tokens(context, context_tokens, answer_start, answer_end):
  """Get answer given context, context_words, and span.

  Args:
    context: A list of bytes, to be decoded with utf-8.
    context_tokens: A list of a list of bytes, to be decoded with utf-8.
    answer_start: An int for answer start.
    answer_end: An int for answer end.
  Returns:
    A list of bytes, encoded with utf-8, for the answer.
  """
  # Word level
  context = context.decode('utf-8')
  spans = convert_to_spans(
      context, [token.decode('utf-8') for token in context_tokens])
  start_char = spans[answer_start][0]
  end_char = spans[answer_end][1]
  result = context[start_char:end_char]
  return result


def get_embedding_map(embeddings_path, draft=False, word_subset=None):
  """Get an `OrderedDict` that maps word to vector.

  Args:
    embeddings_path: `str` value,
      path to the embeddings file (e.g. `glove.6B.50d.txt`)
    draft: `bool` value, whether to only load first 99 for draft mode.
  Returns:
    `OrderedDict` object, mapping word to vector.
  """
  embeddings = collections.OrderedDict()
  with tf.gfile.GFile(embeddings_path, 'r') as fp:
    for idx, line in enumerate(fp):
      if len(line) < 30 and idx == 0:
        # In fasttext, the first line is the # of vocab words.
        continue
      line = line.decode('utf-8')
      tokens = line.strip().split(u' ')
      word = tokens[0]
      vec = list(map(float, tokens[1:]))
      if draft and idx > 99:
        break
      if word_subset:
        if word not in word_subset:
          continue
      embeddings[word] = vec
  return embeddings


# Global state to do some basic caching - avoid reloading
_GLOBAL_VOCAB_CACHE = dict()


def get_pretrained_embeddings_cache(embeddings_path):
  """Get pretrained vocab embeddings."""
  if embeddings_path in _GLOBAL_VOCAB_CACHE:
    return _GLOBAL_VOCAB_CACHE[embeddings_path]
  else:
    tf.logging.info('Loading pretrained embeddings from %s', embeddings_path)
    embeddings = get_embedding_map(embeddings_path)
    # OrderedDict, so keys and values are ordered.
    assert isinstance(embeddings, collections.OrderedDict)
    words = embeddings.keys()
    embeddings_as_arr = np.array(embeddings.values())
  _GLOBAL_VOCAB_CACHE[embeddings_path] = (words, embeddings_as_arr)
  return _GLOBAL_VOCAB_CACHE[embeddings_path]


def squad_generator(path, tokenizer_fn=word_tokenize, as_np=False):
  """Generate SQuAD data from the raw json file."""

  with tf.gfile.GFile(path) as f:
    squad = json.load(f)
  for article in squad['data']:

    for paragraph in article['paragraphs']:
      context = paragraph['context'].strip()
      context_enc = context.encode('utf-8')
      context_tokens = tokenizer_fn(context)
      for qa in paragraph['qas']:
        question = qa['question'].strip()
        id_ = qa['id']

        answer_starts = [answer['answer_start'] for answer in qa['answers']]
        answers = [answer['text'].strip() for answer in qa['answers']]
        answer_ends = [
            start + len(answer)
            for start, answer in zip(answer_starts, answers)
        ]

        feats = {}
        feats['id'] = id_
        feats['answers'] = utf_encode_list(answers)
        feats['num_answers'] = len(answers)

        feats['context'] = context_enc
        feats['context_tokens'] = context_tokens
        feats['context_length'] = len(context_tokens)

        feats['question'] = question.encode('utf-8')
        feats['question_tokens'] = utf_encode_list(tokenizer_fn(question))
        feats['question_length'] = len(feats['question_tokens'])

        spans = convert_to_spans(context, feats['context_tokens'])
        starts = []
        ends = []
        for answer_start, answer_end in zip(answer_starts, answer_ends):
          start, end = get_span(spans, answer_start, answer_end)
          starts.append(start)
          ends.append(end)

        feats['answers_start_token'] = starts
        feats['answers_end_token'] = ends
        feats['context_tokens'] = utf_encode_list(feats['context_tokens'])
        if as_np:
          out = dict()
          for feat in feats:
            out[feat] = np.array(feats[feat])
          yield out
        else:
          yield feats


def get_span(spans, answer_start, answer_end):
  """Get the start/end index that contains the (start,end) interval.

  Args:
    spans: [List of (start, end) tuples]
    answer_start: Start index
    answer_end: End index

  Returns:
    tuple of (start, end) indices into spans such that
    spans[start][0] <= answer_start <= answer_end <= spans[end][1]

  Raises:
    ValueError: if either the start or end position is not found.
  """
  word_answer_start = None
  word_answer_end = None
  for word_idx, span in enumerate(spans):
    if span[0] <= answer_start <= span[1]:
      word_answer_start = word_idx
    if span[0] <= answer_end <= span[1]:
      word_answer_end = word_idx
    if word_answer_start and word_answer_end:
      break
  if word_answer_end is None and word_answer_start is not None:
    # TODO(ddohan): Figure out why this is sometimes necessary
    if answer_end > spans[-1][-1]:
      word_answer_end = len(spans) - 1
  if word_answer_end is None or word_answer_start is None:
    raise ValueError
  assert word_answer_end >= word_answer_start
  return word_answer_start, word_answer_end


FIELDS = ['context', 'question', 'answers']


def build_tfrecord_pipeline(filenames):
  """Read TFRecords from disk to create data pipeline."""
  sequence_feature = tf.FixedLenSequenceFeature(
      [], tf.int64, allow_missing=True)
  str_sequence_feature = tf.FixedLenSequenceFeature(
      [], tf.string, allow_missing=True)
  int_feature = tf.FixedLenFeature([], tf.int64)
  str_feature = tf.FixedLenFeature([], tf.string)
  features = {
      'id': str_feature,
      'num_answers': int_feature,
      'answers': str_sequence_feature,
      'answers_start_token': sequence_feature,
      'answers_end_token': sequence_feature,
      'context': str_feature,
      'context_length': int_feature,
      'context_tokens': str_sequence_feature,
      'question': str_feature,
      'question_length': int_feature,
      'question_tokens': str_sequence_feature,
  }

  def _parse(proto):
    return tf.parse_single_example(proto, features=features)

  ds = tf.data.TFRecordDataset(
      filenames,
      # 1 GB
      buffer_size=1024 * 1024 * 1024,
      num_parallel_reads=8)

  ds = ds.map(_parse, num_parallel_calls=16)
  return ds


def build_generator_pipeline(data_path,
                             split,
                             tokenizer_fn=word_tokenize):
  """Build a data pipeline from raw json SQuAD file."""
  shapes, types = get_shapes_and_types(is_tpu=False, max_length=None)

  def generator():
    path = os.path.join(data_path, '%s-v1.1.json' % split)
    return squad_generator(path=path, tokenizer_fn=tokenizer_fn)

  ds = tf.data.Dataset.from_generator(
      generator, output_types=types, output_shapes=shapes)
  return ds


def get_shapes_and_types(is_tpu=False, max_length=None):
  """Build tuple of (shapes, types) dictionaries specifying the dataset."""
  # TODO(ddohan): Explicitly list types & shapes instead of creating in a loop
  types = {}
  shapes = {}
  length = None
  if is_tpu:
    assert max_length
    length = max_length

  for k in FIELDS:
    if not is_tpu:
      types[k] = tf.string
      types['%s_tokens' % k] = tf.string
      shapes[k] = []
      shapes['%s_tokens' % k] = [length]
    types['%s_length' % k] = tf.int64
    shapes['%s_length' % k] = []
  for k in ['answers_tokens', 'answers_length']:
    if k in types:
      del types[k]
      del shapes[k]

  types['num_answers'] = tf.int64
  types['answers_start_token'] = tf.int64
  types['answers_end_token'] = tf.int64

  shapes['num_answers'] = []
  shapes['answers_start_token'] = []
  shapes['answers_end_token'] = []

  if not is_tpu:
    types['id'] = tf.string
    shapes['id'] = []

  for k in shapes:
    if k.startswith('answer'):
      # TODO(ddohan): Handle multiple answers
      shapes[k] = [1 if is_tpu else None] + shapes[k]

  return shapes, types


def resample_example(example, max_length=256):
  """Given an example and max length, resample the context to that length.

  Start position randomly chosen from [0, answer_start]. Assumes a single
    answer per context, which is true for the SQuAD training set.

  Args:
    example: A single example containing at least these fields:
      ['answers_start_token', 'answers_end_token', 'context_tokens',
      'context_length']
    max_length: Maximum length. Contexts are resampled to this length.

  Returns:
    Resampled example.
  """

  # TODO(ddohan): Consider randomly cropping to shorter lengths
  # TODO(ddohan): Figure out how to resample the raw text as well. Not necessary
  # for training
  def _resample():
    """Helper method for resampling inside cond."""
    x = example
    ans_start = tf.to_int64(x['answers_start_token'][0])
    ans_end = tf.to_int64(x['answers_end_token'][0])
    min_start = tf.maximum(tf.to_int64(0), ans_end - max_length + 1)
    max_start = ans_start
    start_idx = tf.random_uniform([],
                                  min_start,
                                  max_start + 1, dtype=tf.int64)
    for k in ['answers_start_token', 'answers_end_token']:
      x[k] -= start_idx
    x['context_tokens'] = x['context_tokens'][start_idx:start_idx + max_length]
    x['context_length'] = tf.to_int64(tf.shape(x['context_tokens'])[0])
    return x

  def identity():
    return example

  return tf.cond(
      tf.greater_equal(
          tf.to_int32(max_length), tf.to_int32(example['context_length'])),
      true_fn=identity,
      false_fn=_resample)


def get_input_fn(split='dev',
                 shuffle=False,
                 num_repeats=False,
                 limit=None,
                 do_embedding=True,
                 cache=True,
                 max_length=None,
                 resample_too_long=True,
                 data_path=None,
                 vocab_path=None,
                 is_tpu=False,
                 use_generator=True,
                 is_training=False):
  """Build input function."""
  if is_tpu:
    assert max_length

  # Do the GLOVE embedding lookups in the data loader
  if do_embedding:
    # Load and package into the graph directly
    # Vocab is about ~200MB total once filtered down
    embedding_words, embedding_vectors = get_pretrained_embeddings_cache(
        embeddings_path=vocab_path)

  def _input_fn(params=None):
    """Input function compatible with `Experiment` object.

    Args:
      params: Params passed to the estimator. Contains 'batch_size'.

    Returns:
      A tuple of feature tensors and target tensors.

    Raises:
      ValueError: If filtering by length is set during eval mode.
    """
    if not is_training:
      assert not is_tpu
    tf.logging.info('Data pipeline given params:\n%s' % params)
    if is_training:
      batch_size = params.dataset.train_batch_size
    else:
      batch_size = params.dataset.eval_batch_size

    if use_generator:
      tf.logging.info('Building generator data pipeline.')
      ds = build_generator_pipeline(
          data_path=data_path,
          split=split,
          tokenizer_fn=word_tokenize)
    else:
      tf.logging.info('Loading TFRecords from %s' % data_path)
      filenames = tf.gfile.Glob(os.path.join(data_path, '%s_*' % split))
      tf.logging.info(filenames)
      ds = build_tfrecord_pipeline(filenames=filenames)

    if max_length:
      if not is_training:
        raise ValueError('Unable to filter or resample examples at eval time.')
      if resample_too_long:

        tf.logging.info('Resampling with max length %s', max_length)
        def _resample(x):
          return resample_example(x, max_length=max_length)

        ds = ds.map(_resample, num_parallel_calls=16)
      else:
        # Filter out examples over our max length to avoid an error downstream.
        tf.logging.info('Filtering out examples over max length %s', max_length)
        def _not_too_long(x):
          return tf.greater_equal(
              tf.to_int32(max_length), tf.to_int32(x['context_length']))

        ds = ds.filter(_not_too_long)

    if limit:
      # Take the first N examples
      ds = ds.take(limit)

    if cache:
      # Cache dataset to avoid hitting the python generator after first epoch
      ds = ds.cache()

    # Subset that we should actually pass back to the caller
    # This is required to filter out tf.string fields which are not TPU
    # compatible
    # Specifically: id, context, question, context_tokens and question_tokens
    # are all string fields that will be removed.
    shapes, _ = get_shapes_and_types(is_tpu=is_tpu, max_length=max_length)

    if do_embedding:
      # Embed tokens with pretrained word vectors

      # Add in shape info before batching
      shapes['context_vecs'] = [max_length if is_tpu else None, 300]
      shapes['question_vecs'] = [max_length if is_tpu else None, 300]

      vocab_table = tf.contrib.lookup.index_table_from_tensor(
          embedding_words, default_value=0)
      vocab_vectors = tf.constant(embedding_vectors, dtype=tf.float32)

      def lookup(words):
        ids = vocab_table.lookup(words)
        result = tf.nn.embedding_lookup(params=vocab_vectors, ids=ids)
        return result

      def lookup_fields(d):
        d['context_vecs'] = lookup(d['context_tokens'])
        d['question_vecs'] = lookup(d['question_tokens'])
        return d

      ds = ds.map(lookup_fields, num_parallel_calls=16)

    buffer_size = 5000  # Magic number TUNE ME
    repeats = num_repeats if num_repeats else None
    if shuffle and repeats != 1:
      tf.logging.info('Shuffle and repeat size: %s' % buffer_size)
      ds = ds.apply(
          tf.contrib.data.shuffle_and_repeat(
              buffer_size=buffer_size,
              count=repeats))
    elif repeats != 1:
      tf.logging.info('Repeating')
      ds = ds.repeat(count=repeats)
    elif shuffle:
      tf.logging.info('Shuffle size: %s' % buffer_size)
      ds = ds.shuffle(buffer_size=buffer_size)

    def filter_fields(example):
      out = {}
      for k in shapes:
        out[k] = example[k]
      return out

    ds = ds.map(filter_fields, num_parallel_calls=16)

    if is_training:
      ds = ds.apply(
          tf.contrib.data.padded_batch_and_drop_remainder(
              batch_size, padded_shapes=shapes))
    else:
      # Never want to ignore values at eval time
      ds = ds.padded_batch(batch_size, padded_shapes=shapes)
    ds = ds.prefetch(tf.contrib.data.AUTOTUNE)  # Buffer a few batches ahead
    if do_embedding:
      iterator = ds.make_initializable_iterator()
      # Must be initialized when the graph is initialized and before the
      # dataset tensors are evaluated.
      # Run `tf.tables_initializer()` before getting first batch
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           iterator.initializer)
    else:
      iterator = ds.make_one_shot_iterator()
    batch = iterator.get_next()
    return batch, batch

  return _input_fn

# Metrics


def metric_fn(answers, prediction, start, end, yp1, yp2, num_answers):
  """Compute span accuracies and token F1/EM scores."""

  yp1 = tf.expand_dims(yp1, -1)
  yp2 = tf.expand_dims(yp2, -1)
  answer_mask = tf.sequence_mask(num_answers)

  start = tf.to_int64(start)
  end = tf.to_int64(end)
  start_correct = tf.reduce_any(tf.equal(start, yp1) & answer_mask, 1)
  end_correct = tf.reduce_any(tf.equal(end, yp2) & answer_mask, 1)
  correct = start_correct & end_correct

  em = tf.py_func(
      enum_fn(_exact_match_score, dtype='float32'),
      [prediction, answers, answer_mask], 'float32')
  f1 = tf.py_func(
      enum_fn(_f1_score, dtype='float32'), [prediction, answers, answer_mask],
      'float32')

  eval_metric_ops = {
      # TODO(ddohan): Add other useful metrics
      'acc_start':
          tf.metrics.mean(tf.cast(start_correct, 'float')),
      'acc_end':
          tf.metrics.mean(tf.cast(end_correct, 'float')),
      'acc_span':
          tf.metrics.mean(tf.cast(correct, 'float')),
      'em':
          tf.metrics.mean(em),
      'f1':
          tf.metrics.mean(f1),
      # Number of questions processed
      'num_question':
          tf.metrics.true_positives(
              tf.ones([tf.shape(prediction)][0]),
              tf.ones([tf.shape(prediction)][0]))
  }
  return eval_metric_ops


def _f1_score(prediction, ground_truths, answer_mask=None):
  prediction = prediction.decode('utf-8', errors='ignore')
  ground_truths = [
      ground_truth.decode('utf-8', errors='ignore')
      for ground_truth in ground_truths
  ]
  scores = np.array(
      [_f1_score_(prediction, ground_truth) for ground_truth in ground_truths])
  return max(scores * answer_mask.astype(float))


def _exact_match_score(prediction, ground_truths, answer_mask=None):
  prediction = prediction.decode('utf-8', errors='ignore')
  ground_truths = [
      ground_truth.decode('utf-8') for ground_truth in ground_truths
  ]
  scores = np.array([
      float(_exact_match_score_(prediction, ground_truth))
      for ground_truth in ground_truths
  ])
  return max(scores * answer_mask.astype(float))


def enum_fn(fn, dtype='object'):
  # Map function across a batch

  def new_fn(*args):
    return np.array([fn(*each_args) for each_args in zip(*args)], dtype=dtype)

  return new_fn


# Functions below are copied from official SQuAD eval script and SHOULD NOT
# BE MODIFIED.


def _normalize_answer(s):
  """Lower text and remove punctuation, articles and extra whitespace.

  Directly copied from official SQuAD eval script, SHOULD NOT BE MODIFIED.

  Args:
    s: Input text.
  Returns:
    Normalized text.
  """

  def remove_articles(text):
    return re.sub(r'\b(a|an|the)\b', ' ', text)

  def white_space_fix(text):
    return ' '.join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return ''.join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def _f1_score_(prediction, ground_truth):
  """Directly copied from official SQuAD eval script, SHOULD NOT BE MODIFIED."""
  # tf.logging.info('%s | %s' % (prediction, ground_truth))
  prediction_tokens = _normalize_answer(prediction).split()
  ground_truth_tokens = _normalize_answer(ground_truth).split()
  common = collections.Counter(prediction_tokens) & collections.Counter(
      ground_truth_tokens)
  num_same = sum(common.values())
  if num_same == 0:
    return 0
  precision = 1.0 * num_same / len(prediction_tokens)
  recall = 1.0 * num_same / len(ground_truth_tokens)
  f1 = (2 * precision * recall) / (precision + recall)
  return f1


def _exact_match_score_(prediction, ground_truth):
  """Directly copied from official SQuAD eval script, SHOULD NOT BE MODIFIED."""
  return _normalize_answer(prediction) == _normalize_answer(ground_truth)
