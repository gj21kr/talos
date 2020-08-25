# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Built-in loss functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.losses import util as tf_losses_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls

class Loss(object):

  def __init__(self, reduction=losses_utils.Reduction.AUTO, name=None):
    losses_utils.Reduction.validate(reduction)
    self.reduction = reduction
    self.name = name
    # SUM_OVER_BATCH is only allowed in losses managed by `fit` or
    # CannedEstimators.
    self._allow_sum_over_batch_size = False
    self._set_name_scope()

  def _set_name_scope(self):
    if self.name is None:
      self._name_scope = self.__class__.__name__
    elif self.name == '<lambda>':
      self._name_scope = 'lambda'
    else:
      # E.g. '_my_loss' => 'my_loss'
      self._name_scope = self.name.strip('_')

  def __call__(self, y_true, y_pred, sample_weight=None):
    # If we are wrapping a lambda function strip '<>' from the name as it is not
    # accepted in scope name.
    graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
        y_true, y_pred, sample_weight)
    with K.name_scope(self._name_scope), graph_ctx:
      ag_call = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
      losses = ag_call(y_true, y_pred)
      return losses_utils.compute_weighted_loss(
          losses, sample_weight, reduction=self._get_reduction())

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    return {'reduction': self.reduction, 'name': self.name}

  def call(self, y_true, y_pred):
    NotImplementedError('Must be implemented in subclasses.')

  def _get_reduction(self):
    if (not self._allow_sum_over_batch_size and
        distribution_strategy_context.has_strategy() and
        (self.reduction == losses_utils.Reduction.AUTO or
         self.reduction == losses_utils.Reduction.SUM_OVER_BATCH_SIZE)):
      raise ValueError(
          'Please use `tf.keras.losses.Reduction.SUM` or '
          '`tf.keras.losses.Reduction.NONE` for loss reduction when losses are '
          'used with `tf.distribute.Strategy` outside of the built-in training '
          'loops. You can implement '
          '`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch '
          'size like:\n```\nwith strategy.scope():\n'
          '    loss_obj = tf.keras.losses.CategoricalCrossentropy('
          'reduction=tf.keras.losses.Reduction.NONE)\n....\n'
          '    loss = tf.reduce_sum(loss_obj(labels, predictions)) * '
          '(1. / global_batch_size)\n```\nPlease see '
          'https://www.tensorflow.org/tutorials/distribute/custom_training'
          ' for more details.')

    if self.reduction == losses_utils.Reduction.AUTO:
      return losses_utils.Reduction.SUM_OVER_BATCH_SIZE
    return self.reduction


class LossFunctionWrapper(Loss):
  def __init__(self,
               fn,
               reduction=losses_utils.Reduction.AUTO,
               name=None,
               **kwargs):
    super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
      y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(
          y_pred, y_true)
    ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
    return ag_fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
    base_config = super(LossFunctionWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class MeanSquaredError(LossFunctionWrapper):
  def __init__(self,
               reduction=losses_utils.Reduction.AUTO,
               name='mean_squared_error'):
    super(MeanSquaredError, self).__init__(
        mean_squared_error, name=name, reduction=reduction)

class MeanAbsoluteError(LossFunctionWrapper):
  def __init__(self,
               reduction=losses_utils.Reduction.AUTO,
               name='mean_absolute_error'):
    super(MeanAbsoluteError, self).__init__(
        mean_absolute_error, name=name, reduction=reduction)

class MeanAbsolutePercentageError(LossFunctionWrapper):
  def __init__(self,
               reduction=losses_utils.Reduction.AUTO,
               name='mean_absolute_percentage_error'):
    super(MeanAbsolutePercentageError, self).__init__(
        mean_absolute_percentage_error, name=name, reduction=reduction)

class MeanSquaredLogarithmicError(LossFunctionWrapper):
  def __init__(self,
               reduction=losses_utils.Reduction.AUTO,
               name='mean_squared_logarithmic_error'):
    super(MeanSquaredLogarithmicError, self).__init__(
        mean_squared_logarithmic_error, name=name, reduction=reduction)
        
class DiceCoefLoss(LossFunctionWrapper):
  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_utils.Reduction.AUTO,
               name='binary_crossentropy'):
    super(DiceCoefLoss, self).__init__(
        dice_coef_loss,
        name=name,
        label_smoothing=label_smoothing)
    self.from_logits = from_logits

class BinaryCrossentropy(LossFunctionWrapper):
  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_utils.Reduction.AUTO,
               name='binary_crossentropy'):
    super(BinaryCrossentropy, self).__init__(
        binary_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing)
    self.from_logits = from_logits

class CategoricalCrossentropy(LossFunctionWrapper):
  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_utils.Reduction.AUTO,
               name='categorical_crossentropy'):
    super(CategoricalCrossentropy, self).__init__(
        categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing)

class SparseCategoricalCrossentropy(LossFunctionWrapper):
  def __init__(self,
               from_logits=False,
               reduction=losses_utils.Reduction.AUTO,
               name='sparse_categorical_crossentropy'):
    super(SparseCategoricalCrossentropy, self).__init__(
        sparse_categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits)

class Hinge(LossFunctionWrapper):
  def __init__(self, reduction=losses_utils.Reduction.AUTO, name='hinge'):
    super(Hinge, self).__init__(hinge, name=name, reduction=reduction)

class SquaredHinge(LossFunctionWrapper):
  def __init__(self,
               reduction=losses_utils.Reduction.AUTO,
               name='squared_hinge'):
    super(SquaredHinge, self).__init__(
        squared_hinge, name=name, reduction=reduction)

class CategoricalHinge(LossFunctionWrapper):
  def __init__(self,
               reduction=losses_utils.Reduction.AUTO,
               name='categorical_hinge'):
    super(CategoricalHinge, self).__init__(
        categorical_hinge, name=name, reduction=reduction)

class Poisson(LossFunctionWrapper):
  def __init__(self, reduction=losses_utils.Reduction.AUTO, name='poisson'):
    super(Poisson, self).__init__(poisson, name=name, reduction=reduction)

class LogCosh(LossFunctionWrapper):
  def __init__(self, reduction=losses_utils.Reduction.AUTO, name='log_cosh'):
    super(LogCosh, self).__init__(log_cosh, name=name, reduction=reduction)

class KLDivergence(LossFunctionWrapper):
  def __init__(self,
               reduction=losses_utils.Reduction.AUTO,
               name='kl_divergence'):
    super(KLDivergence, self).__init__(
        kl_divergence, name=name, reduction=reduction)

class Huber(LossFunctionWrapper):
  def __init__(self,
               delta=1.0,
               reduction=losses_utils.Reduction.AUTO,
               name='huber_loss'):
    super(Huber, self).__init__(
        huber, name=name, reduction=reduction, delta=delta)

def mean_squared_error(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)

def mean_absolute_error(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  diff = math_ops.abs(
      (y_true - y_pred) / K.maximum(math_ops.abs(y_true), K.epsilon()))
  return 100. * K.mean(diff, axis=-1)

def mean_squared_logarithmic_error(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  first_log = math_ops.log(K.maximum(y_pred, K.epsilon()) + 1.)
  second_log = math_ops.log(K.maximum(y_true, K.epsilon()) + 1.)
  return K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)


def _maybe_convert_labels(y_true):
  are_zeros = math_ops.equal(y_true, 0)
  are_ones = math_ops.equal(y_true, 1)
  is_binary = math_ops.reduce_all(math_ops.logical_or(are_zeros, are_ones))

  def _convert_binary_labels():
    # Convert the binary labels to -1 or 1.
    return 2. * y_true - 1.

  updated_y_true = smart_cond.smart_cond(is_binary,
                                         _convert_binary_labels, lambda: y_true)
  return updated_y_true

def squared_hinge(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = _maybe_convert_labels(y_true)
  return K.mean(
      math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.)), axis=-1)

def hinge(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = _maybe_convert_labels(y_true)
  return K.mean(math_ops.maximum(1. - y_true * y_pred, 0.), axis=-1)

def categorical_hinge(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  pos = math_ops.reduce_sum(y_true * y_pred, axis=-1)
  neg = math_ops.reduce_max((1. - y_true) * y_pred, axis=-1)
  zero = math_ops.cast(0., y_pred.dtype)
  return math_ops.maximum(neg - pos + 1., zero)

def huber(y_true, y_pred, delta=1.0):
  y_pred = math_ops.cast(y_pred, dtype=K.floatx())
  y_true = math_ops.cast(y_true, dtype=K.floatx())
  delta = math_ops.cast(delta, dtype=K.floatx())
  error = math_ops.subtract(y_pred, y_true)
  abs_error = math_ops.abs(error)
  quadratic = math_ops.minimum(abs_error, delta)
  linear = math_ops.subtract(abs_error, quadratic)
  return K.mean(
      math_ops.add(
          math_ops.multiply(
              ops.convert_to_tensor(0.5, dtype=quadratic.dtype),
              math_ops.multiply(quadratic, quadratic)),
          math_ops.multiply(delta, linear)),
      axis=-1)

def log_cosh(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)

  def _logcosh(x):
    return x + nn.softplus(-2. * x) - math_ops.cast(math_ops.log(2.), x.dtype)

  return K.mean(_logcosh(y_pred - y_true), axis=-1)

def categorical_crossentropy(y_true,
                             y_pred,
                             from_logits=False,
                             label_smoothing=0):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  label_smoothing = ops.convert_to_tensor(label_smoothing, dtype=K.floatx())

  def _smooth_labels():
    num_classes = math_ops.cast(array_ops.shape(y_true)[-1], y_pred.dtype)
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

  y_true = smart_cond.smart_cond(label_smoothing,
                                 _smooth_labels, lambda: y_true)
  return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)

def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.sparse_categorical_crossentropy(
      y_true, y_pred, from_logits=from_logits, axis=axis)

def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  label_smoothing = ops.convert_to_tensor(label_smoothing, dtype=K.floatx())

  def _smooth_labels():
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

  y_true = smart_cond.smart_cond(label_smoothing,
                                 _smooth_labels, lambda: y_true)
  return K.mean(
      K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)

def dice_coef_loss(y_true, y_pred, smooth = 1e-07, label_smoothing=0):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  label_smoothing = ops.convert_to_tensor(label_smoothing, dtype=K.floatx())

  def _smooth_labels():
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

  y_true = smart_cond.smart_cond(label_smoothing,
                                 _smooth_labels, lambda: y_true)
  return (2.*K.sum(K.abs(y_true * y_pred), axis=-1)+smooth)/(K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
  
def kl_divergence(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = K.clip(y_true, K.epsilon(), 1)
  y_pred = K.clip(y_pred, K.epsilon(), 1)
  return math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)

def poisson(y_true, y_pred):
  y_pred = ops.convert_to_tensor(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)


def cosine_similarity(y_true, y_pred, axis=-1):
  y_true = nn.l2_normalize(y_true, axis=axis)
  y_pred = nn.l2_normalize(y_pred, axis=axis)
  return -math_ops.reduce_sum(y_true * y_pred, axis=axis)


class CosineSimilarity(LossFunctionWrapper):
  def __init__(self,
               axis=-1,
               reduction=losses_utils.Reduction.AUTO,
               name='cosine_similarity'):
    super(CosineSimilarity, self).__init__(
        cosine_similarity, reduction=reduction, name=name, axis=axis)


# Aliases.

bce = BCE = binary_crossentropy
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence = kl_divergence
logcosh = log_cosh
huber_loss = huber


def is_categorical_crossentropy(loss):
  result = ((isinstance(loss, CategoricalCrossentropy) or
             (isinstance(loss, LossFunctionWrapper) and
              loss.fn == categorical_crossentropy) or
             (hasattr(loss, '__name__') and
              loss.__name__ == 'categorical_crossentropy') or
             (loss == 'categorical_crossentropy')))
  return result

def serialize(loss):
  return serialize_keras_object(loss)

def deserialize(name, custom_objects=None):
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='loss function')

def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError(
        'Could not interpret loss function identifier: {}'.format(identifier))


LABEL_DTYPES_FOR_LOSSES = {
    losses_impl.sparse_softmax_cross_entropy: 'int32',
    sparse_categorical_crossentropy: 'int32'
}