'''
The Keras-compatible loss function which calculates individual box losses for the SSD model. Currently supports TensorFlow only.

Wrote based on Pierluigi's code here: https://github.com/pierluigiferrari/ssd_keras.git.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''




def compute_individual_loss(self, y_true, y_pred):
'''
Compute the loss PER box of the SSD model prediction against the ground truth.

Arguments:
y_true (array): A Numpy array of shape `(batch_size, #boxes, #classes + 12)`,
    where `#boxes` is the total number of boxes that the model predicts
    per image. Be careful to make sure that the index of each given
    box in `y_true` is the same as the index for the corresponding
    box in `y_pred`. The last axis must have length `#classes + 12` and contain
    `[classes one-hot encoded, 4 ground truth box coordinate offsets, 8 arbitrary entries]`
    in this order, including the background class. The last eight entries of the
    last axis are not used by this function and therefore their contents are
    irrelevant, they only exist so that `y_true` has the same shape as `y_pred`,
    where the last four entries of the last axis contain the anchor box
    coordinates, which are needed during inference. Important: Boxes that
    you want the cost function to ignore need to have a one-hot
    class vector of all zeros.
y_pred (Keras tensor): The model prediction. The shape is identical
    to that of `y_true`, i.e. `(batch_size, #boxes, #classes + 12)`.
    The last axis must contain entries in the format
    `[classes one-hot encoded, 4 predicted box coordinate offsets, 8 arbitrary entries]`.

Returns:
A numpy array containing the resulted loss PER box for classification and localization.
'''

   self.neg_pos_ratio = tf.constant(self.neg_pos_ratio)
   self.n_neg_min = tf.constant(self.n_neg_min)
   self.alpha = tf.constant(self.alpha)

   batch_size = tf.shape(y_pred)[0]
   n_boxes = tf.shape(y_pred)[1]

   classification_loss = tf.cast(self.log_loss(y_true[:,:,:-12], y_pred[:,:,:-12]), float)
   localization_loss = tf.cast(self.smooth_L1_loss(y_true[:,:,-12:-8], y_pred[:,:,-12:-8]), float)

   negatives = y_true[:,:,0]
   positives = tf.cast(tf.reduce_max(y_true[:,:,1:-12], axis=-1), float)

   n_positive = tf.reduce_sum(positives)

   pos_class_loss_all = classification_loss * positives
   neg_class_loss_all = classification_loss * negatives 

   n_neg_losses = tf.count_nonzero(neg_class_loss_all, dtype=tf.int32)
   n_negative_keep = tf.minimum(tf.maximum(self.neg_pos_ratio * tf.cast(n_positive, tf.int32), self.n_neg_min), n_neg_losses)

   def f1():
      return tf.zeros([batch_size])

   def f2():
      neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1]) 
      values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                      k=n_negative_keep,
                                      sorted=False) 
      negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                      updates=tf.ones_like(indices, dtype=tf.int32),
                                      shape=tf.shape(neg_class_loss_all_1D)) 
      negatives_keep = tf.cast(tf.reshape(negatives_keep, [batch_size, n_boxes]), float) 
      neg_class_loss = classification_loss * negatives_keep 
      return neg_class_loss

   neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

   class_loss = pos_class_loss_all + neg_class_loss # Tensor of shape (batch_size, #boxes)

   loc_loss = localization_loss * positives

   total_loss = (class_loss + self.alpha * loc_loss) / tf.maximum(1.0, n_positive)

   total_loss = total_loss * tf.cast(batch_size, float)

   total_loss = total_loss.numpy()

   return total_loss
