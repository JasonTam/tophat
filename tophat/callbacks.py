"""
Additional Keras-style callbacks

References:
    https://keras.io/callbacks/
"""
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras._impl.keras.callbacks import *
from tophat.utils.log import logger
from tophat.embedding import EmbeddingProjector


class History(Callback):
    def on_train_begin(self, logs=None):
        self.epoch = []
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)


class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))


class TaskLossHistory(Callback):
    def __init__(self, task_name):
        super().__init__()
        self.task_name = task_name

    def on_train_begin(self, logs=None):
        self.epoch_losses = []

    def on_epoch_begin(self, epoch, logs=None):
        self.batch_losses = []

    def on_batch_end(self, batch, logs=None):
        if logs['task'] == self.task_name:
            self.batch_losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_losses.append(np.mean(self.batch_losses))


class Monitor(Callback):
    def __init__(self, loss_hist, freq):
        super().__init__()
        self.loss_hist = loss_hist
        self.freq = freq

    def on_epoch_begin(self, epoch, logs=None):
        self.tic = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if self.freq and (epoch % self.freq) == 0:
            toc = time.time() - self.tic
            logger.info('\t'.join(
                [f'ep={epoch}', f'time={toc:.3f}s'] +
                [f'{l.task_name}_loss={l.epoch_losses[-1]:.3f}'
                 for l in self.loss_hist]
            ))


class Summary(Callback):
    # TODO: consider using TensorBoard callback
    def __init__(self, log_dir, sess=None):
        super().__init__()
        self.sess = sess
        self.summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(
            log_dir, graph=tf.get_default_graph())

    def on_epoch_end(self, epoch, logs=None):
        summary_str = self.sess.run(self.summary_op)
        self.summary_writer.add_summary(summary_str, epoch)
        self.summary_writer.flush()


class Projector(Callback):
    # TODO: consider using TensorBoard callback
    def __init__(self, log_dir, embedding_map, summary_writer, names_d=None):
        super().__init__()
        self.emb_projector = EmbeddingProjector(
            embedding_map, summary_writer,
            log_dir=log_dir,
            names_d=names_d,
        )

    def on_train_end(self, logs=None):
        self.emb_projector.viz()


class ModelSaver(Callback):
    def __init__(self, save_dir, sess=None):
        super().__init__()
        self.sess = sess
        self.saver = tf.train.Saver()
        self.save_dir = save_dir

    def on_train_end(self, logs=None):
        self.saver.save(self.sess, os.path.join(self.save_dir, 'model.ckpt'))


class Scorer(Callback):
    def __init__(self, validator, summary_writer, freq=1, sess=None,
                 macro=True):
        """
        Args:
            freq: frequency of validation (in epochs)
            macro: Macro average across users, else, micro average across
                interactions
        """
        super().__init__()
        self.sess = sess
        self.summary_writer = summary_writer
        self.validator = validator
        self.freq = freq
        self.score_hists = []
        self.macro = macro

    def on_train_begin(self, logs=None):
        self.validator.make_ops()

    def on_epoch_end(self, epoch, logs=None):
        if self.freq and ((epoch + 1) % self.freq) == 0:
            logger.info(f'Scoring ({self.validator.name}):')
            val_scores = self.validator.run_val(
                self.sess, self.summary_writer, step=epoch, macro=self.macro)
            self.score_hists.append(val_scores)

    @property
    def score_df(self):
        return pd.DataFrame(self.score_hists)

