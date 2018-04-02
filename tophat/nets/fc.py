import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, dropout, batch_norm


def simple_fc(x: tf.Tensor, reg=None, scope_name='fc'):

    with tf.name_scope(scope_name):

        bn0 = batch_norm(x, decay=0.9)
        drop0 = dropout(bn0, 0.7)

        fc1 = fully_connected(drop0, 30, activation_fn=tf.nn.relu,
                              weights_regularizer=reg)
        bn1 = batch_norm(fc1, decay=0.9)
        drop1 = dropout(bn1, 0.7)
        contrib_f = tf.squeeze(
            fully_connected(drop1, 1, activation_fn=tf.nn.sigmoid),
            name='score')

    return contrib_f
