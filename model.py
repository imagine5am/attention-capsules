# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Functions to build the Attention OCR model.

Usage example:
  ocr_model = model.Model(num_char_classes, seq_length, num_of_views)

  data = ... # create namedtuple InputEndpoints
  endpoints = model.create_base(data.images, data.labels_one_hot)
  # endpoints.predicted_chars is a tensor with predicted character codes.
  total_loss = model.create_loss(data, endpoints)
"""
import sys
import collections
import logging
import tensorflow as tf
from tensorflow.contrib import slim
#from tensorflow.contrib.slim.nets import inception

import metrics
import sequence_layers
import utils
from caps_layers import create_prim_conv3d_caps, create_dense_caps, layer_shape, create_conv3d_caps
#import config
from tensorflow.python import pywrap_tensorflow
import time
import numpy as np
#import mayurinference

OutputEndpoints = collections.namedtuple('OutputEndpoints', [
  'chars_logit', 'chars_log_prob', 'predicted_chars', 'predicted_scores',
  'predicted_text'
])

# TODO(gorban): replace with tf.HParams when it is released.
ModelParams = collections.namedtuple('ModelParams', [
  'num_char_classes', 'seq_length', 'num_views', 'null_code'
])

ConvTowerParams = collections.namedtuple('ConvTowerParams', ['final_endpoint'])

SequenceLogitsParams = collections.namedtuple('SequenceLogitsParams', [
  'use_attention', 'use_autoregression', 'num_lstm_units', 'weight_decay',
  'lstm_state_clip_value'
])

SequenceLossParams = collections.namedtuple('SequenceLossParams', [
  'label_smoothing', 'ignore_nulls', 'average_across_timesteps'
])

EncodeCoordinatesParams = collections.namedtuple('EncodeCoordinatesParams', [
  'enabled'
])


def _dict_to_array(id_to_char, default_character):
  num_char_classes = max(id_to_char.keys()) + 1
  array = [default_character] * num_char_classes
  for k, v in id_to_char.items():
    array[k] = v
  return array


class CharsetMapper(object):
  """A simple class to map tensor ids into strings.

    It works only when the character set is 1:1 mapping between individual
    characters and individual ids.

    Make sure you call tf.tables_initializer().run() as part of the init op.
    """

  def __init__(self, charset, default_character='?'):
    """Creates a lookup table.

    Args:
      charset: a dictionary with id-to-character mapping.
    """
    mapping_strings = tf.constant(_dict_to_array(charset, default_character))
    self.table = tf.contrib.lookup.index_to_string_table_from_tensor(
      mapping=mapping_strings, default_value=default_character)

  def get_text(self, ids):
    """Returns a string corresponding to a sequence of character ids.

        Args:
          ids: a tensor with shape [batch_size, max_sequence_length]
        """
    return tf.reduce_join(
      self.table.lookup(tf.to_int64(ids)), reduction_indices=1)


def create_skip_connection(in_caps_layer, n_units, kernel_size, strides=(1, 1, 1), 
                            padding='VALID', name='skip', activation=tf.nn.relu):
    '''
    skip_connection1 = create_skip_connection(sec_caps, 128, kernel_size=[1, 3, 3], 
                                                    strides=[1, 1, 1], padding='SAME', 
                                                    name='skip_1')
    '''

    in_caps_layer = in_caps_layer[0]
    batch_size = tf.shape(in_caps_layer)[0]
    _, d, h, w, ch, _ = in_caps_layer.get_shape()
    d, h, w, ch = map(int, [d, h, w, ch])

    in_caps_res = tf.reshape(in_caps_layer, [batch_size, d, h, w, ch * 16])

    return tf.layers.conv3d_transpose(in_caps_res, n_units, kernel_size=kernel_size, 
                                        strides=strides, padding=padding, use_bias=False, 
                                        activation=activation, name=name)


def get_softmax_loss_fn(label_smoothing):
  """Returns sparse or dense loss function depending on the label_smoothing.

    Args:
      label_smoothing: weight for label smoothing

    Returns:
      a function which takes labels and predictions as arguments and returns
      a softmax loss for the selected type of labels (sparse or dense).
    """
  if label_smoothing > 0:

    def loss_fn(labels, logits):
      return (tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
  else:

    def loss_fn(labels, logits):
      return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

  return loss_fn


class Model(object):
  """Class to create the Attention OCR Model."""

  def __init__(self,
               num_char_classes,
               seq_length,
               num_views,
               null_code,
               mparams=None,
               charset=None):
    """Initialized model parameters.

    Args:
      num_char_classes: size of character set.
      seq_length: number of characters in a sequence.
      num_views: Number of views (conv towers) to use.
      null_code: A character code corresponding to a character which
        indicates end of a sequence.
      mparams: a dictionary with hyper parameters for methods,  keys -
        function names, values - corresponding namedtuples.
      charset: an optional dictionary with a mapping between character ids and
        utf8 strings. If specified the OutputEndpoints.predicted_text will
        utf8 encoded strings corresponding to the character ids returned by
        OutputEndpoints.predicted_chars (by default the predicted_text contains
        an empty vector).
        NOTE: Make sure you call tf.tables_initializer().run() if the charset
        specified.
    """
    super(Model, self).__init__()
    self.myweight = None
    self._params = ModelParams(
      num_char_classes=num_char_classes,
      seq_length=seq_length,
      num_views=num_views,
      null_code=null_code)
    self._mparams = self.default_mparams()
    if mparams:
      self._mparams.update(mparams)
    self._charset = charset
    self.w_and_b = {'none': None,'zero': tf.zeros_initializer()}


  def my_init(self):
      caps_graph = tf.get_default_graph()
      conv1_kernel = caps_graph.get_tensor_by_name('conv1/kernel:0')
      with tf.Session() as sess:
            # Initializing the variables
            val = conv1_kernel.eval(session=sess)
      return val
    
    
  def init_network(self,x_input):
    n_classes = 12
    print('Building Caps3d Model')
    with tf.variable_scope('caps_fn/CAPS'):
      # creates the video encoder
      conv1 = tf.layers.conv3d(x_input, 64, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                   activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                   bias_initializer=self.w_and_b['zero'], name='conv1')
        
      conv2 = tf.layers.conv3d(conv1, 128, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 2, 2],
                   activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                   bias_initializer=self.w_and_b['zero'], name='conv2')
        
      conv3 = tf.layers.conv3d(conv2, 64, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                   activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                   bias_initializer=self.w_and_b['zero'], name='conv3')
  
      conv4 = tf.layers.conv3d(conv3, 128, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 2, 2],
                   activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                   bias_initializer=self.w_and_b['zero'], name='conv4')
  
      conv5 = tf.layers.conv3d(conv4, 256, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                   activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                   bias_initializer=self.w_and_b['zero'], name='conv5')
  
      conv6 = tf.layers.conv3d(conv5, 256, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 2, 2],
                   activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                   bias_initializer=self.w_and_b['zero'], name='conv6')
  
      conv7 = tf.layers.conv3d(conv6, 512, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                   activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                   bias_initializer=self.w_and_b['zero'], name='conv7')
  
      conv8 = tf.layers.conv3d(conv7, 512, kernel_size=[3, 3, 3], padding='SAME', strides=[1, 1, 1],
                   activation=tf.nn.relu, kernel_initializer=self.w_and_b['none'],
                   bias_initializer=self.w_and_b['zero'], name='conv8')
  
      # config.print_layers:
      print('Conv1:', conv1.get_shape())
      print('Conv2:', conv2.get_shape())
      print('Conv3:', conv3.get_shape())
      print('Conv4:', conv4.get_shape())
      print('Conv5:', conv5.get_shape())
      print('Conv6:', conv6.get_shape())
      print('Conv7:', conv7.get_shape())
      print('Conv8:', conv8.get_shape())
      
      # with tf.device('/gpu:0'):
      # creates the primary capsule layer: conv caps1
      prim_caps = create_prim_conv3d_caps(conv8, 32, kernel_size=[3, 9, 9], strides=[1, 1, 1],             
                        padding='VALID', name='prim_caps')
      # with tf.device('/gpu:0'):
      # creates the secondary capsule layer: conv caps2
      sec_caps = create_conv3d_caps(prim_caps, 32, kernel_size=[3, 5, 5], strides=[1, 2, 2],
                     padding='VALID', name='sec_caps', route_mean=True)
      #with tf.device('/gpu:0'):
      # creates the final capsule layer: class caps
      
      pred_caps = create_dense_caps(sec_caps, 12, subset_routing=-1, route_min=0.0,
                     name='pred_caps', coord_add=True, ch_same_w=True)
  
      #config.print_layers:
      print('Primary Caps:', layer_shape(prim_caps))
      print('Second Caps:', layer_shape(sec_caps))
      print('Prediction Caps:', layer_shape(pred_caps))
  
      # obtains the activations of the class caps layer and gets the class prediction
      digit_preds = tf.reshape(pred_caps[1], (-1, n_classes))
      predictions = tf.cast(tf.argmax(input=digit_preds, axis=1), tf.int32)
  
      pred_caps_poses = pred_caps[0]
      batch_size = tf.shape(pred_caps_poses)[0]
      _, n_classes, dim = pred_caps_poses.get_shape()
      n_classes, dim = map(int, [n_classes, dim])
  
      # masks the capsules that are not the ground truth (training) or the prediction (testing)
      vec_to_use = predictions
      # vec_to_use = tf.cond(self.is_train, lambda: self.y_input, lambda: self.predictions)
      vec_to_use = digit_preds#tf.one_hot(vec_to_use, depth=n_classes)
      vec_to_use = tf.tile(tf.reshape(vec_to_use, (batch_size, n_classes, 1)), multiples=[1, 1, dim])
      masked_caps = pred_caps_poses * tf.cast(vec_to_use, dtype=tf.float32)
      masked_caps = tf.reshape(masked_caps, (batch_size, n_classes * dim))
  
      # creates the decoder network
      recon_fc1 = tf.layers.dense(masked_caps, 4 * 10 * 24 * 1, activation=tf.nn.relu, name='recon_fc1')
      recon_fc1 = tf.reshape(recon_fc1, (batch_size, 4, 10, 24, 1))

      deconv1 = tf.layers.conv3d_transpose(recon_fc1, 128, kernel_size=[1, 3, 3], 
                        strides=[1, 1, 1], padding='SAME', 
                        use_bias=False, activation=tf.nn.relu, 
                        name='deconv1')

      skip_connection1 = create_skip_connection(sec_caps, 128, kernel_size=[1, 3, 3], 
                            strides=[1, 1, 1], padding='SAME', 
                            name='skip_1')

      deconv1 = tf.concat([deconv1, skip_connection1], axis=-1)

      deconv2 = tf.layers.conv3d_transpose(deconv1, 128, kernel_size=[3, 6, 6], strides=[1, 2, 2],
                         padding='VALID', use_bias=False, activation=tf.nn.relu, name='deconv2')
        
      skip_connection2 = create_skip_connection(prim_caps, 128, kernel_size=[1, 3, 3], strides=[1, 1, 1],
                           padding='SAME', name='skip_2')

      print('deconv1:', deconv1.get_shape())                     
      print('deconv2:', deconv2.get_shape())
      print('skip_connection2:', skip_connection2.get_shape())
      deconv2 = tf.concat([deconv2, skip_connection2], axis=-1)

      deconv3 = tf.layers.conv3d_transpose(deconv2, 256, kernel_size=[3, 9, 9], strides=[1, 1, 1],
                         padding='VALID',
                         use_bias=False, activation=tf.nn.relu, name='deconv3')
    return deconv3, digit_preds


  def default_mparams(self):
    return {
      'conv_tower_fn':
        ConvTowerParams(final_endpoint='Mixed_5d'),
      'sequence_logit_fn':
        SequenceLogitsParams(
          use_attention=True,
          use_autoregression=True,
          num_lstm_units=256,
          weight_decay=0.00004,
          lstm_state_clip_value=10.0),
      'sequence_loss_fn':
        SequenceLossParams(
          label_smoothing=0.1,
          ignore_nulls=True,
          average_across_timesteps=False),
      'encode_coordinates_fn': EncodeCoordinatesParams(enabled=True)
    }

  def set_mparam(self, function, **kwargs):
    self._mparams[function] = self._mparams[function]._replace(**kwargs)

  '''def conv_tower_fn(self, images, is_training=True, reuse=None):
    "#""Computes convolutional features using the InceptionV3 model.

    Args:
      images: A tensor of shape [batch_size, height, width, channels].
      is_training: whether is training or not.
      reuse: whether or not the network and its variables should be reused. To
        be able to reuse 'scope' must be given.

    Returns:
      A tensor of shape [batch_size, OH, OW, N], where OWxOH is resolution of
      output feature map and N is number of output features (depends on the
      network architecture).
    "#""
    mparams = self._mparams['conv_tower_fn']
    logging.debug('Using final_endpoint=%s', mparams.final_endpoint)
    with tf.variable_scope('conv_tower_fn/INCE'):
      if reuse:
        tf.get_variable_scope().reuse_variables()
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
          net, _ = inception.inception_v3_base(
            images, final_endpoint=mparams.final_endpoint)
      return net'''

  def _create_lstm_inputs(self, net):
    """Splits an input tensor into a list of tensors (features).

    Args:
      net: A feature map of shape [batch_size, num_features, feature_size].

    Raises:
      AssertionError: if num_features is less than seq_length.

    Returns:
      A list with seq_length tensors of shape [batch_size, feature_size]
    """
    num_features = net.get_shape().dims[1].value
    if num_features < self._params.seq_length:
      raise AssertionError('Incorrect dimension #1 of input tensor'
                           ' %d should be bigger than %d (shape=%s)' %
                           (num_features, self._params.seq_length,
                            net.get_shape()))
    elif num_features > self._params.seq_length:
      logging.warning('Ignoring some features: use %d of %d (shape=%s)',
                      self._params.seq_length, num_features, net.get_shape())
      net = tf.slice(net, [0, 0, 0], [-1, self._params.seq_length, -1])

    return tf.unstack(net, axis=1)

  def sequence_logit_fn(self, net, labels_one_hot): #, preds):
    mparams = self._mparams['sequence_logit_fn']
    # TODO(gorban): remove /alias suffixes from the scopes.
    with tf.variable_scope('sequence_logit_fn/SQLR'):
      layer_class = sequence_layers.get_layer_class(mparams.use_attention,
                                                    mparams.use_autoregression)
      # layer = layer_class(net, labels_one_hot, preds, self._params, mparams)
      layer = layer_class(net, labels_one_hot, self._params, mparams)
      return layer.create_logits()

  def max_pool_views(self, nets_list):
    """Max pool across all nets in spatial dimensions.

    Args:
      nets_list: A list of 4D tensors with identical size.

    Returns:
      A tensor with the same size as any input tensors.
    """
    batch_size, height, width, num_features = [
      d.value for d in nets_list[0].get_shape().dims
    ]
    xy_flat_shape = (batch_size, 1, height * width, num_features)
    nets_for_merge = []
    with tf.variable_scope('max_pool_views', values=nets_list):
      for net in nets_list:
        nets_for_merge.append(tf.reshape(net, xy_flat_shape))
      merged_net = tf.concat(nets_for_merge, 1)
      net = slim.max_pool2d(
        merged_net, kerel_size=[len(nets_list), 1], stride=1)
      net = tf.reshape(net, (batch_size, height, width, num_features))
    return net

  def pool_views_fn(self, nets):
    """Combines output of multiple convolutional towers into a single tensor.

    It stacks towers one on top another (in height dim) in a 4x1 grid.
    The order is arbitrary design choice and shouldn't matter mu.

    Args:
      nets: list of tensors of shape=[batch_size, height, width, num_features].

    Returns:
      A tensor of shape [batch_size, seq_length, features_size].
    """
    with tf.variable_scope('pool_views_fn/STCK'):
      net = tf.concat(nets, 1)
      batch_size = net.get_shape().dims[0].value
      feature_size = net.get_shape().dims[3].value
      return tf.reshape(net, [batch_size, -1, feature_size])

  def char_predictions(self, chars_logit):
    """Returns confidence scores (softmax values) for predicted characters.

    Args:
      chars_logit: chars logits, a tensor with shape
        [batch_size x seq_length x num_char_classes]

    Returns:
      A tuple (ids, log_prob, scores), where:
        ids - predicted characters, a int32 tensor with shape
          [batch_size x seq_length];
        log_prob - a log probability of all characters, a float tensor with
          shape [batch_size, seq_length, num_char_classes];
        scores - corresponding confidence scores for characters, a float
        tensor
          with shape [batch_size x seq_length].
    """
    log_prob = utils.logits_to_log_prob(chars_logit)
    ids = tf.to_int32(tf.argmax(log_prob, axis=2), name='predicted_chars')
    mask = tf.cast(
      slim.one_hot_encoding(ids, self._params.num_char_classes), tf.bool)
    all_scores = tf.nn.softmax(chars_logit)
    selected_scores = tf.boolean_mask(all_scores, mask, name='char_scores')
    scores = tf.reshape(selected_scores, shape=(-1, self._params.seq_length))
    return ids, log_prob, scores

  def encode_coordinates_fn(self, net):
    """Adds one-hot encoding of coordinates to different views in the networks.

    For each "pixel" of a feature map it adds a onehot encoded x and y
    coordinates.

    Args:
      net: a tensor of shape=[batch_size, height, width, num_features]

    Returns:
      a tensor with the same height and width, but altered feature_size.
    """
    mparams = self._mparams['encode_coordinates_fn']
    if mparams.enabled:
      print("net", net)
      batch_size, h, w, _ = net.shape.as_list()
      x, y = tf.meshgrid(tf.range(w), tf.range(h))
      w_loc = slim.one_hot_encoding(x, num_classes=w)
      h_loc = slim.one_hot_encoding(y, num_classes=h)
      loc = tf.concat([h_loc, w_loc], 2)
      loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1])
      return tf.concat([net, loc], 3)
    else:
      return net

  def encode_coordinates_temporal_fn(self, net):
    """Adds one-hot encoding of coordinates to different views in the networks.

    For each "pixel" of a feature map it adds a onehot encoded x and y
    coordinates.

    Args:
      net: a tensor of shape=[batch_size, height, 8*width, num_features]#1 X 8 x 32 x 60 x 256

    Returns:
      a tensor with the same height and width, but altered feature_size.
    """
    mparams = self._mparams['encode_coordinates_fn']
    if mparams.enabled:
      print("net", net)#1, 8, 14, 28, 1088
      batch_size, t, h, w, _ = net.shape.as_list()
      x, y, t1  = tf.meshgrid(tf.range(w),tf.range(h),tf.range(t))#1, 8, 14, 28, 1088
      print(t1)#14, 8, 28
      w_loc = slim.one_hot_encoding(x, num_classes=w)
      h_loc = slim.one_hot_encoding(y, num_classes=h)
      t_loc = slim.one_hot_encoding(t1, num_classes=t)
      loc = tf.concat([t_loc, h_loc, w_loc], 3)#w,h,t,w+h+t
      loc = tf.tile(tf.expand_dims(loc, 0), [batch_size, 1, 1, 1, 1])#bXhXwXtXsum
      loc = tf.transpose(loc, [0, 3, 1, 2, 4])#1X8XHXwX3
      return tf.concat([net, loc], 4)
    else:
      return net

  def create_base(self,
                  images,
                  labels_one_hot,
                  scope='AttentionOcr_v1',
                  reuse=None):
    """Creates a base part of the Model (no gradients, losses or summaries).

    Args:
      images: A tensor of shape [batch_size, height, width, channels].
      labels_one_hot: Optional (can be None) one-hot encoding for ground truth
        labels. If provided the function will create a model for training.
      scope: Optional variable_scope.
      reuse: whether or not the network and its variables should be reused. To
        be able to reuse 'scope' must be given.

    Returns:
      A named tuple OutputEndpoints.
    """
    logging.debug('images: %s', images)
    is_training = labels_one_hot is not None
    with tf.variable_scope(scope, reuse=reuse):
      views = tf.split(value=images, num_or_size_splits=self._params.num_views, axis=2)
      #b X H x 8W x 3 -> 8XBXHXwX3
      logging.debug('Views=%d single view: %s', len(views), views[0])
      views1 = tf.stack(views)#8X1X3X256X480
      views2 = tf.transpose(views1, [1, 0, 2, 3, 4])#1X8XHXwX3
      net1, preds = self.init_network(views2)#1 X 8 x 32 x 60 x 256
      #nets = [
      #  self.conv_tower_fn(v, is_training, reuse=(i != 0))
      #  for i, v in enumerate(views)
      #]
      #logging.debug('Conv tower: %s', nets[0])

      #nets = [self.encode_coordinates_fn(net) for net in net1]
      #logging.debug('Conv tower w/ encoded coordinates: %s', nets[0])

      #net = self.pool_views_fn(nets)
      #logging.debug('Pooled views: %s', net)
      '''
      with tf.Session() as sess:
            print(sess)#self.xinput = views[0].eval(session=tf.get_default_session())#sess.run(views)
      print("*************************************************************************")
      '''
      #print(self.myweight)#1X256X480
      
      #net2 = self.encode_coordinates_temporal_fn(net1)#1, 8, 14, 28, 1088
      net1a = tf.unstack(net1, axis=1)
      #net2 = [self.encode_coordinates_fn(net) for net in net1a]

      net3 = self.pool_views_fn(net1a)
      # chars_logit = self.sequence_logit_fn(net3, labels_one_hot, preds)
      chars_logit = self.sequence_logit_fn(net3, labels_one_hot)
      logging.debug('chars_logit: %s', chars_logit)

      predicted_chars, chars_log_prob, predicted_scores = (self.char_predictions(chars_logit))
      if self._charset:
        character_mapper = CharsetMapper(self._charset)
        predicted_text = character_mapper.get_text(predicted_chars)
      else:
        predicted_text = tf.constant([])
      #net2 = mayurinference.inference(nets, nets[0], net5, views,net6) #, net4)
      # net3 = tf.stack(views)
      #net3 = net2[0]
      # print(nets[0])
      # net = tf.Print(net,[nets[0]])
      # net = tf.Print(net, [tf.shape(nets[0])])
      # , message="This is test: ")
      #net3 = tf.convert_to_tensor(net3)

        
    return OutputEndpoints(
      chars_logit=chars_logit,
      chars_log_prob=chars_log_prob,
      predicted_chars=predicted_chars,
      predicted_scores=predicted_scores,
      predicted_text=predicted_text)

  def create_loss(self, data, endpoints):
    """Creates all losses required to train the model.

    Args:
      data: InputEndpoints namedtuple.
      endpoints: Model namedtuple.

    Returns:
      Total loss.
    """
    # NOTE: the return value of ModelLoss is not used directly for the
    # gradient computation because under the hood it calls slim.losses.AddLoss,
    # which registers the loss in an internal collection and later returns it
    # as part of GetTotalLoss. We need to use total loss because model may have
    # multiple losses including regularization losses.
    self.sequence_loss_fn(endpoints.chars_logit, data.labels)
    total_loss = slim.losses.get_total_loss()
    tf.summary.scalar('TotalLoss', total_loss)
    return total_loss

  def label_smoothing_regularization(self, chars_labels, weight=0.1):
    """Applies a label smoothing regularization.

    Uses the same method as in https://arxiv.org/abs/1512.00567.

    Args:
      chars_labels: ground truth ids of charactes,
        shape=[batch_size, seq_length];
      weight: label-smoothing regularization weight.

    Returns:
      A sensor with the same shape as the input.
    """
    one_hot_labels = tf.one_hot(
      chars_labels, depth=self._params.num_char_classes, axis=-1)
    pos_weight = 1.0 - weight
    neg_weight = weight / self._params.num_char_classes
    return one_hot_labels * pos_weight + neg_weight

  def sequence_loss_fn(self, chars_logits, chars_labels):
    """Loss function for char sequence.

    Depending on values of hyper parameters it applies label smoothing and can
    also ignore all null chars after the first one.

    Args:
      chars_logits: logits for predicted characters,
        shape=[batch_size, seq_length, num_char_classes];
      chars_labels: ground truth ids of characters,
        shape=[batch_size, seq_length];
      mparams: method hyper parameters.

    Returns:
      A Tensor with shape [batch_size] - the log-perplexity for each sequence.
    """
    mparams = self._mparams['sequence_loss_fn']
    with tf.variable_scope('sequence_loss_fn/SLF'):
      if mparams.label_smoothing > 0:
        smoothed_one_hot_labels = self.label_smoothing_regularization(
          chars_labels, mparams.label_smoothing)
        labels_list = tf.unstack(smoothed_one_hot_labels, axis=1)
      else:        # NOTE: in case of sparse softma we are not using one-hot
        # encoding.
        labels_list = tf.unstack(chars_labels, axis=1)

      batch_size, seq_length, _ = chars_logits.shape.as_list()
      if mparams.ignore_nulls:
        weights = tf.ones((batch_size, seq_length), dtype=tf.float32)
      else:
        # Suppose that reject character is the last in the charset.
        reject_char = tf.constant(
          self._params.num_char_classes - 1,
          shape=(batch_size, seq_length),
          dtype=tf.int64)
        known_char = tf.not_equal(chars_labels, reject_char)
        weights = tf.to_float(known_char)

      logits_list = tf.unstack(chars_logits, axis=1)
      weights_list = tf.unstack(weights, axis=1)
      loss = tf.contrib.legacy_seq2seq.sequence_loss(
        logits_list,
        labels_list,
        weights_list,
        softmax_loss_function=get_softmax_loss_fn(mparams.label_smoothing),
        average_across_timesteps=mparams.average_across_timesteps)
      tf.losses.add_loss(loss)
      return loss

  def create_summaries(self, data, endpoints, charset, is_training):
    """Creates all summaries for the model.

    Args:
      data: InputEndpoints namedtuple.
      endpoints: OutputEndpoints namedtuple.
      charset: A dictionary with mapping between character codes and
        unicode characters. Use the one provided by a dataset.charset.
      is_training: If True will create summary prefixes for training job,
        otherwise - for evaluation.

    Returns:
      A list of evaluation ops
    """

    def sname(label):
      prefix = 'train' if is_training else 'eval'
      return '%s/%s' % (prefix, label)

    max_outputs = 4
    # TODO(gorban): uncomment, when tf.summary.text released.
    # charset_mapper = CharsetMapper(charset)
    # pr_text = charset_mapper.get_text(
    #     endpoints.predicted_chars[:max_outputs,:])
    # tf.summary.text(sname('text/pr'), pr_text)
    # gt_text = charset_mapper.get_text(data.labels[:max_outputs,:])
    # tf.summary.text(sname('text/gt'), gt_text)
    tf.summary.image(sname('image'), data.images, max_outputs=max_outputs)

    if is_training:
      tf.summary.image(
        sname('image/orig'), data.images_orig, max_outputs=max_outputs)
      for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
      return None

    else:
      names_to_values = {}
      names_to_updates = {}

      def use_metric(name, value_update_tuple):
        names_to_values[name] = value_update_tuple[0]
        names_to_updates[name] = value_update_tuple[1]

      use_metric('CharacterAccuracy',
                 metrics.char_accuracy(
                   endpoints.predicted_chars,
                   data.labels,
                   streaming=True,
                   rej_char=self._params.null_code))
      # Sequence accuracy computed by cutting sequence at the first null char
      use_metric('SequenceAccuracy',
                 metrics.sequence_accuracy(
                   endpoints.predicted_chars,
                   data.labels,
                   streaming=True,
                   rej_char=self._params.null_code))

      for name, value in names_to_values.items():
        summary_name = 'eval/' + name
        tf.summary.scalar(summary_name, tf.Print(value, [value], summary_name))
      return list(names_to_updates.values())

  def create_init_fn_to_restore(self, master_checkpoint,
                                caps_checkpoint=None,
                                trainable_base=True):
    """Creates an init operations to restore weights from various checkpoints.

    Args:
     master_checkpoint: path to a checkpointwhich contains all weights for
        the whole model.
      inception_checkpoint: path to a checkpoint which contains weights for the
        inception part only.

    Returns:
      a function to run initialization ops.
    """
    all_assign_ops = []
    all_feed_dict = {}

    def assign_from_checkpoint(variables, checkpoint):
      logging.info('Request to re-store %d weights from %s',
                   len(variables), checkpoint)
      if not variables:
        logging.error('Can\'t find any variables to restore.')
        sys.exit(1)
      assign_op, feed_dict = slim.assign_from_checkpoint(checkpoint, variables)
      all_assign_ops.append(assign_op)
      all_feed_dict.update(feed_dict)

    logging.info('variables_to_restore:\n%s' % utils.variables_to_restore().keys())
    logging.info('moving_average_variables:\n%s' % [v.op.name for v in tf.moving_average_variables()])
    logging.info('trainable_variables:\n%s' % [v.op.name for v in tf.trainable_variables()])
    if master_checkpoint:
      assign_from_checkpoint(utils.variables_to_restore(), master_checkpoint)

    if caps_checkpoint:
      variables = utils.variables_to_restore(
                                        'AttentionOcr_v1/caps_fn/CAPS', 
                                        strip_scope=True, 
                                        trainable=trainable_base)
      assign_from_checkpoint(variables, caps_checkpoint)

    def init_assign_fn(sess):
      logging.info('Restoring checkpoint(s)')
      sess.run(all_assign_ops, all_feed_dict)

    return init_assign_fn
