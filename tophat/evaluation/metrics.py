import tensorflow as tf
from tensorflow.contrib.metrics import (
    streaming_mean, streaming_sparse_average_precision_at_k)

from typing import Dict


def make_metrics_ops(fwd_op: tf.Tensor,
                     input_fwd_d: Dict[str, tf.Tensor]):
    """Makes operations for calculating metrics

    Args:
        fwd_op: Forward operation of network
        input_fwd_d: Dictionary of feed-forward placeholders
            Optionally, this will also contain target variables

    Returns:
        Tuple of metric operations, reset operations, and feed dictionary
    """

    if (('y_true_ph' in input_fwd_d.keys()) and
            ('y_true_bool_ph' in input_fwd_d.keys())):
        targ_d = {
            'y_true_ph': input_fwd_d['y_true_ph'],
            'y_true_bool_ph': input_fwd_d['y_true_bool_ph'],
        }
    else:
        with tf.name_scope('placeholders_eval'):
            targ_d = {
                'y_true_ph': tf.placeholder('int64'),
                'y_true_bool_ph': tf.placeholder('bool'),
            }

    # Define our metrics: MAP@10 and AUC (hardcoded for now)
    k = 10
    with tf.name_scope('stream_metrics'):
        val_preds = tf.expand_dims(fwd_op(input_fwd_d), 0)
        # NB: THESE STREAMING METRICS DO **MICRO** UPDATES
        # Ex) for AUC, it will effectively concat all predictions and all trues
        #   (over all users)
        # instead of averaging the AUC's over users
        mapk, update_op_mapk = streaming_sparse_average_precision_at_k(
            val_preds, targ_d['y_true_ph'], k=k)
        auc, update_op_auc = tf.metrics.auc(
            targ_d['y_true_bool_ph'], tf.sigmoid(val_preds))
        # Tjur's Pseudo R2 inspired bpr
        pb_t = tf.to_float(targ_d['y_true_bool_ph'],
                           name='true_pos')
        nb_t = tf.to_float(tf.logical_not(targ_d['y_true_bool_ph']),
                           name='true_neg')
        pos_mean = (tf.reduce_sum(tf.multiply(pb_t, val_preds)) /
                    tf.reduce_sum(pb_t))
        neg_mean = (tf.reduce_sum(tf.multiply(nb_t, val_preds)) /
                    tf.reduce_sum(nb_t))
        tjurs_bpr_i = tf.subtract(1., tf.sigmoid(pos_mean - neg_mean),
                                  name='tjurs_bpr_val')
        tjur, update_op_tjur = streaming_mean(tjurs_bpr_i)
        pm, update_op_pm = streaming_mean(pos_mean)
        nm, update_op_nm = streaming_mean(neg_mean)

    stream_vars = [i for i in tf.local_variables()
                   if i.name.split('/')[0] == 'stream_metrics']
    reset_metrics_op = [tf.variables_initializer(stream_vars)]

    metric_ops_d = {
        'mapk': (mapk, update_op_mapk),
        'auc': (auc, update_op_auc),
        'tjurs': (tjur, update_op_tjur),
        'pm': (pm, update_op_pm),
        'nm': (nm, update_op_nm),
    }
    return metric_ops_d, reset_metrics_op, targ_d
