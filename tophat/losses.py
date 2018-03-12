import tensorflow as tf
from typing import Callable, Optional

PairLossFn = Callable[
    [tf.Tensor, tf.Tensor, Optional[tf.Tensor], Optional[int]],
    tf.Tensor]


def bpr_loss(pos_score: tf.Tensor,
             neg_score: tf.Tensor,
             *args) -> tf.Tensor:
    loss = tf.subtract(1., tf.sigmoid(pos_score - neg_score), name='bpr')
    return tf.reduce_mean(loss, name='bpr_mean')


def hinge_loss(pos_score: tf.Tensor,
               neg_score: tf.Tensor,
               *args) -> tf.Tensor:
    loss = tf.maximum(0., neg_score - pos_score + 1., name='hinge')
    return tf.reduce_mean(loss, name='hinge_mean')


def softplus_loss(pos_score: tf.Tensor,
                  neg_score: tf.Tensor,
                  *args) -> tf.Tensor:
    loss = tf.log1p(tf.exp(neg_score - pos_score + 1.), name='softplus')
    return tf.reduce_mean(loss, name='softplus_mean')


def kos_loss(pos_score: tf.Tensor,
             neg_score: tf.Tensor,
             first_violation: tf.Tensor,
             n_items: int,
             ) -> tf.Tensor:
    hinge = tf.maximum(0., neg_score - pos_score + 1., name='hinge')

    apprx_rank = tf.maximum(
        1., tf.cast(
            tf.floor((n_items - 1) / (first_violation + 1))
            , tf.float32), name='apprx_rank')
    loss = tf.log(apprx_rank) * hinge

    return tf.reduce_mean(loss, name='kos_mean')


NAMED_LOSSES = {
    'bpr': bpr_loss,
    'hinge': hinge_loss,
    'softplus': softplus_loss,
    'kos': kos_loss,
}
