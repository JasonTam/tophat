import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from time import time
import pickle

import os
import glob

from tophat.core import FactModel
from tophat.embedding import EmbeddingMap, EmbeddingProjector
from tophat.nets import BilinearNet

from tophat.naive_sampler import PairSampler
from tophat.data import TrainDataLoader
from tophat.evaluation import Validator
from tophat.config_parser import Config
from tophat.export_repr import RepresentationExportJob
from lib_cerebro_py.log import logger
import argparse
import pprint


class FitJob(object):
    def __init__(self, fit_config):
        self.fit_config = fit_config

        self.emb_dim = fit_config.get('emb_dim')
        self.batch_size = fit_config.get('batch_size')
        self.n_steps = fit_config.get('n_steps')
        self.log_every = fit_config.get('log_every')
        self.eval_every = fit_config.get('eval_every')
        self.save_every = fit_config.get('save_every')
        self.seed = fit_config.get('seed')
        tf.set_random_seed(self.seed)

        # Some export stuff
        self.ckpt_upload_s3_uri = fit_config.get('ckpt_upload_s3_uri')
        # TODO: maybe have an `export_config`
        self.repr_export_path = fit_config.get('repr_export_path')

        self.log_dir = fit_config.get('log_dir')
        tf.gfile.MkDir(self.log_dir)

        self.sess: tf.Session = None

        # Allocate fields for Model & related objs
        self.model = None  # TODO: Make base class for this
        self.embedding_map: EmbeddingMap = None
        self.sampler: PairSampler = None
        self.train_data_loader: TrainDataLoader = None
        self.validator: Validator = None

    def model_init(self):
        self.train_data_loader = TrainDataLoader(
            interactions_train=self.fit_config.get('interactions_train'),
            user_features=self.fit_config.get('user_features'),
            item_features=self.fit_config.get('item_features'),
            user_specific_feature=self.fit_config.get('user_specific_feature'),
            item_specific_feature=self.fit_config.get('item_specific_feature'),
            context_cols=self.fit_config.get('context_cols'),
            batch_size=self.fit_config.get('batch_size'),
        )
        if self.fit_config.get('validation_params'):
            self.validator = Validator(
                self.fit_config, self.train_data_loader,
                seed=self.seed,
                **self.fit_config.get('validation_params'))

        # Ops and feature map
        logger.info('Building graph ...')
        if self.validator:
            zero_init_rows = self.validator.zero_init_rows
        else:
            zero_init_rows = None
        self.embedding_map = EmbeddingMap(
            self.train_data_loader,
            embedding_dim=self.emb_dim,
            l2_emb=0.,
            zero_init_rows=zero_init_rows,
            vis_specific_embs=False,
            feature_weights_d=self.fit_config.get('feature_weights_d'),
        )

        self.model = FactModel(net=BilinearNet(
            embedding_map=self.embedding_map)
        )

        # Save the catalog
        pickle.dump(self.model.net.embedding_map.cats_d,
                    open(os.path.join(self.log_dir, 'cats_d.p'), 'wb'))

        if self.validator:
            self.validator.make_ops(self.model)

        # Sample Generator
        logger.info('Setting up local sampler ...')
        self.sampler = PairSampler(
            self.train_data_loader,
            self.model.input_pair_d,
            self.batch_size,
            method='adaptive',
            model=self.model,
            seed=self.seed
        )

    def model_fit(self):
        feed_dict_gen = iter(self.sampler)

        loss = self.model.get_loss()
        train_op = self.model.training(loss)

        # Fit Loop
        n_interactions = len(self.train_data_loader.interactions_df)
        logger.info(f'Approx n_epochs: '
                    f'{(self.n_steps * self.batch_size) / n_interactions}')

        self.sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True))
        # self.sess = tf_debug.LocalCLIDebugWrapperSession(self.sess)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Set session on sampler (in case of adaptive sampling)
        self.sampler.sess = self.sess

        # Write distribution for embeddings
        for col, var in self.embedding_map.embeddings_d.items():
            tf.summary.histogram(f'{col}-emb', var)
        for col, var in self.embedding_map.biases_d.items():
            tf.summary.histogram(f'{col}-bias', var)

        summary = tf.summary.merge_all()
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(
            self.log_dir, graph=tf.get_default_graph())
        embedding_projector = EmbeddingProjector(
            self.embedding_map, summary_writer, self.fit_config)

        tic = time()
        for step in range(self.n_steps):
            feed_pair_dict = next(feed_dict_gen)
            _, loss_val = self.sess.run([train_op, loss],
                                        feed_dict=feed_pair_dict)

            if (step % self.log_every == 0) and step > 0:
                summary_str = self.sess.run(summary, feed_dict=feed_pair_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

                toc = time() - tic
                tic = time()
                logger.info('(%.3f sec) \t Step %d: \t (train)loss = %.8f ' % (
                    toc, step, loss_val))

                if (step % self.save_every == 0) and step > 0:
                    saver.save(self.sess,
                               os.path.join(self.log_dir, 'model.ckpt'), step)
                    logger.info('...Model checkpointed')

                if self.validator and step % self.eval_every == 0:
                    self.validator.run_val(self.sess, summary_writer, step)

        embedding_projector.viz()

    def upload_ckpt(self):
        # Upload LOG_DIR to s3 path if applicable
        if self.ckpt_upload_s3_uri:
            logger.info(f'uploading ckpts to {self.ckpt_upload_s3_uri}...')
            from lib_cerebro_py.aws.aws_s3_object import AwsS3Object
            for p in glob.glob(os.path.join(self.log_dir, '**')):
                if p.startswith('events'):
                    continue
                path_upload = os.path.join(
                    self.ckpt_upload_s3_uri,
                    p.split(self.log_dir)[-1].strip('/'))
                o = AwsS3Object(path_upload)
                o.upload_file(p)
            logger.info(f'...uploaded ckpts to {self.ckpt_upload_s3_uri}')

    def export_reprs(self):
        """ Wrapper for Representation Export Job"""
        if self.repr_export_path:
            # Calculate and Export lightfm-style Representations
            repr_export_job = RepresentationExportJob(
                self.sess,  # TODO: tmp workaround
                self.embedding_map,
                self.train_data_loader,
                dir_export=self.repr_export_path,
            )
            logger.info(f'Running representation export job...')
            repr_export_job.run()
            logger.info(f'...finished representation export job')

    def run(self):
        self.model_init()
        self.model_fit()
        self.upload_ckpt()
        self.export_reprs()

        return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit a tiefRex model from scratch')
    parser.add_argument('environment', help='Run environment',
                        default='local', nargs='?',
                        choices=['local', 'integ', 'prod'])
    parser.add_argument('--log_tag', help='Append this tag to the log dir',
                        default='', nargs='?',)
    parser.add_argument('--log_overwrite',
                        help='Overwrite log dir',
                        action='store_true',)

    args = parser.parse_args()
    logger.info(pprint.pformat(args))
    config = Config(f'config/fit_config-{args.environment}.py')

    config.__dict__['_params']['log_dir'] += args.log_tag

    if args.log_overwrite:
        import shutil
        shutil.rmtree(config.get('log_dir'))

    logger.info(pprint.pformat(config._params))

    job = FitJob(fit_config=config)
    job.run()

