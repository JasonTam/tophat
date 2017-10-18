#!/bin/python3
"""
config file for tiefrex
"""

import os
import glob
import tensorflow as tf
from tiefrex.constants import __file__


local_data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../data/gilt')

BUCKET_IMPORTS = 's3://cerebro-recengine-imports'
BUCKET_FIXTURES = 's3://cerebro-recengine-fixtures'
BUCKET_EXPORTS = 's3://cerebro-recengine-exports'

PREFIX = 'tiefrex'

dir_model = '/tmp/tensorboard-logs/2017-10-16-T171918/'

path_cats = '/tmp/cat_shit.p'
path_meta = glob.glob(os.path.join(dir_model, '*.meta'))[0]
path_ckpt = tf.train.latest_checkpoint(dir_model)

dir_export = '/tmp/tr_export'


