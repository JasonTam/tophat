import tensorflow as tf
from typing import Callable, Optional

PairLossFn = Callable[
    [tf.Tensor, tf.Tensor, Optional[tf.Tensor], Optional[int]],
    tf.Tensor]


def bpr_loss(pos_score: tf.Tensor,
             neg_score: tf.Tensor,
             *_) -> tf.Tensor:
    """Bayesian Personalized Ranking loss [1]_

    References:
        .. [1] Rendle, Steffen, et al. "BPR: Bayesian personalized ranking
           from implicit feedback." Proceedings of the twenty-fifth conference
           on uncertainty in artificial intelligence. AUAI Press, 2009.

    """
    loss = tf.subtract(1., tf.sigmoid(pos_score - neg_score), name='bpr')
    return tf.reduce_mean(loss, name='bpr_mean')


def hinge_loss(pos_score: tf.Tensor,
               neg_score: tf.Tensor,
               *_) -> tf.Tensor:
    loss = tf.maximum(0., neg_score - pos_score + 1., name='hinge')
    return tf.reduce_mean(loss, name='hinge_mean')


def softplus_loss(pos_score: tf.Tensor,
                  neg_score: tf.Tensor,
                  *_) -> tf.Tensor:
    loss = tf.log1p(tf.exp(neg_score - pos_score + 1.), name='softplus')
    return tf.reduce_mean(loss, name='softplus_mean')


def kos_loss(pos_score: tf.Tensor,
             neg_score: tf.Tensor,
             first_violation: tf.Tensor,
             n_items: int,
             ) -> tf.Tensor:
    """k-Order Statistic
    Weighted Approximately Ranked Pairwise loss [2]_

    References:
        .. [2] Weston, Jason, Hector Yee, and Ron J. Weiss. "Learning to rank
           recommendations with the k-order statistic loss." Proceedings of the
           7th ACM conference on Recommender systems. ACM, 2013.

    """

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
