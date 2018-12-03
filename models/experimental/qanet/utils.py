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

"""Config utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import pprint

import six
from six import text_type


# TODO(ddohan): FrozenConfig type
class Config(dict):
  """a dictionary that supports dot and dict notation.

  Create:
    d = Config()
    d = Config({'val1':'first'})

  Get:
    d.val2
    d['val2']

  Set:
    d.val2 = 'second'
    d['val2'] = 'second'
  """
  __getattr__ = dict.__getitem__
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __str__(self):
    return pprint.pformat(self)

  def __deepcopy__(self, memo):
    return self.__class__([(copy.deepcopy(k, memo), copy.deepcopy(v, memo))
                           for k, v in self.items()])


def to_config(mapping):
  out = Config(copy.deepcopy(mapping))
  for k, v in six.iteritems(out):
    if isinstance(v, dict):
      out[k] = to_config(v)
  return out


def unflatten_dict(flat_dict):
  """Convert a flattened dict to a nested dict.

  Inverse of flatten_config.

  Args:
    flat_dict: A dictionary to unflatten.

  Returns:
    A dictionary with all keys containing `.` split into nested dicts.
    {'a.b.c': 1} --> {'a': {'b': {'c': 1}}}
  """
  result = {}
  for key, val in six.iteritems(flat_dict):
    parts = key.split('.')
    cur = result
    for part in parts[:-1]:
      if part not in cur:
        cur[part] = Config()
      cur = cur[part]
    cur[parts[-1]] = val
  return Config(result)


def parse_config_string(string):
  """Parse a config string such as one produced by `config_to_string`.

  example:

  A config of:
  ```
  {
    'model': {
      'fn': 'RNN'
    }
    'train_steps': 500
  }
  ```

  Yields a serialized string of: `model.fn=RNN,train_steps=500`

  Args:
   string: String to parse.

  Returns:
    dict resulting from parsing the string. Keys are split on `.`s.

  """
  result = {}
  for entry in string.split(','):
    try:
      key, val = entry.split('=')
    except ValueError:
      raise ValueError('Error parsing entry %s' % entry)
    val = _try_numeric(val)
    result[key] = val
  return unflatten_dict(result)


def _try_numeric(string):
  """Attempt to convert a string to an int then a float.


  Args:
    string: String to convert

  Returns:
    Attempts, in order, to return an integer, a float, and finally the original
    string.
  """
  try:
    float_val = float(string)
    if math.floor(float_val) == float_val:
      return int(float_val)
    return float_val
  except ValueError:
    return string


def _convert_type(val, tp):
  """Attempt to convert given value to type.

  This is used when trying to convert an input value to fit the desired type.

  Args:
    val: Value to convert.
    tp: Type to convert to.

  Returns:
    Value after type conversion.

  Raises:
    ValueError: If the conversion fails.
  """
  if tp in [int, float, str, text_type, bool, tuple, list]:
    in_type = type(val)
    cast = tp(val)
    if in_type(cast) != val:
      raise TypeError(
          'Type conversion between %s (%s) and %s (%s) loses information.' %
          (val, type(val), cast, tp))
    return cast
  raise ValueError(
      'Cannot convert %s (type %s) to type %s' % (val, type(val), tp))


def merge_fixed_structure(*args, **kwargs):
  kwargs['_merge_validate'] = True
  return merge(*args, **kwargs)


def merge(*args, **kwargs):
  """Merge together an iterable of configs in order.

  The first instance of a key determines its type. The last instance of a key
  determines its value.

  Example:

  args[0] = {
    'layers': 1,
    'dropout': 0.5
  }
  kwargs = {
    dropout': 0
  }

  Final result:
  {
    'layers': 1,
    'dropout': 0.0  # note the type change
  }

  Args:
    *args: List of dict-like objects to merge in order.
    **kwargs: Any additional values to add. Treated like as a final additional
      dict to merge.

  Returns:
    dict resulting from merging all configs together.

  Raises:
    TypeError: if there is a type mismatch between the same key across dicts.
    ValueError: If _merge_validate is specified and there is a type mismatch.
  """
  assert args
  validate = False
  if '_merge_validate' in kwargs:
    validate = kwargs['_merge_validate']
    del kwargs['_merge_validate']
  config = copy.deepcopy(args[0])
  configs = list(args)
  configs.append(kwargs)
  for c in configs[1:]:
    for k, v in six.iteritems(c):
      if isinstance(v, dict):
        v = copy.deepcopy(v)
      if k in config:
        value_type = type(config[k])

        if config[k] is not None and v is not None and not isinstance(
            v, value_type):
          v = value_type(v)

        if isinstance(v, dict):
          config[k] = merge(config[k], v, _merge_validate=validate)
        else:
          config[k] = v
      else:
        if validate:
          raise ValueError('Target did not contain key %s' % k)
        config[k] = v

  return copy.deepcopy(config)
