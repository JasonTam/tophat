import tensorflow as tf
from typing import Callable

PairLossFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


def bpr_loss(pos_score: tf.Tensor, neg_score: tf.Tensor) -> tf.Tensor:
    loss = tf.subtract(1., tf.sigmoid(pos_score - neg_score), name='bpr')
    return tf.reduce_mean(loss, name='bpr_mean')


def hinge_loss(pos_score: tf.Tensor, neg_score: tf.Tensor) -> tf.Tensor:
    loss = tf.clip_by_value(
        neg_score - pos_score + 1.,
        clip_value_min=0.,
        clip_value_max=999.,
        name='hinge')
    return tf.reduce_mean(loss, name='hinge_mean')


def softplus_loss(pos_score: tf.Tensor, neg_score: tf.Tensor) -> tf.Tensor:
    loss = tf.log1p(tf.exp(neg_score - pos_score + 1.), name='softplus')
    return tf.reduce_mean(loss, name='softplus_mean')
