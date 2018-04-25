import tensorflow as tf
import numpy as np
import itertools as it
from tophat.tasks.wrapper import FactorizationTask
import tophat.callbacks as cbks
from typing import Optional, List, Union, Sized


class TophatModel(object):

    def __init__(self,
                 tasks: List[FactorizationTask],
                 sess: Optional[tf.Session] = None,
                 ):

        self.tasks = tasks
        # Assure all tasks are built
        for task in self.tasks:
            if not task.built:
                task.build()

        self.steps_per_epoch = sum((t.steps_per_epoch for t in self.tasks))
        self.task_samp_weights = [t.steps_per_epoch / self.steps_per_epoch
                                  for t in self.tasks]

        self.sess = sess
        self.sess_init()

        # Assume embedding map is shared for all tasks, just grab first
        self.embedding_map = self.tasks[0].embedding_map

        self.global_step = 0

    def sess_init(self):
        self.sess = self.sess or tf.Session()
        # Set session on sampler (in case of adaptive sampling)
        for task in self.tasks:
            task.sampler.sess = self.sess
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def fit(self,
            n_epochs: Optional[int] = 1,
            callbacks: Optional[List[cbks.Callback]] = None,
            verbose: int = 1,
            ):
        """Alternating task fit

        Args:
            n_epochs: number of effective epochs
                This is approximate since we're randomly alternating tasks
            callbacks: list of callbacks to perform during fit loop
            verbose: periodicity (in epochs) of logging

        Returns:

        """

        n_tasks_per_epoch = int(self.steps_per_epoch)
        self.loss_hists = [cbks.TaskLossHistory(t.name) for t in self.tasks]

        _callbacks = (callbacks or []) + self.loss_hists
        if verbose:
            _callbacks.append(cbks.Monitor(self.loss_hists, verbose))

        for c in _callbacks:
            if hasattr(c, 'sess') and c.sess is None:
                c.sess = self.sess

        callbacks = cbks.CallbackList(_callbacks)

        callbacks.on_train_begin()

        for epoch_ind in range(n_epochs):
            callbacks.on_epoch_begin(epoch_ind)

            if len(self.tasks) == 1:
                sampled_tasks = it.repeat(self.tasks[0], n_tasks_per_epoch)
            else:
                sampled_tasks = np.random.choice(
                    self.tasks, n_tasks_per_epoch, p=self.task_samp_weights)

            for step_ind, task in enumerate(sampled_tasks):
                batch_logs = {'batch': step_ind, 'size': task.batch_size,
                              'task': task.name}

                callbacks.on_batch_begin(step_ind, batch_logs)

                # Perform operations
                task_loss, _ = self.sess.run([task.loss, task.train_op])
                batch_logs['loss'] = task_loss

                callbacks.on_batch_end(step_ind, batch_logs)
                self.global_step += 1

            callbacks.on_epoch_end(epoch_ind)
        callbacks.on_train_end()
