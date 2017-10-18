#!/bin/python3
"""
config file for tiefrex
"""

import os
import glob
import tensorflow as tf
import re
from tiefrex.constants import __file__
from lib_cerebro_py.aws.aws_s3_object import AwsS3Object, AwsS3Uri
from lib_cerebro_py.log import logger


local_data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../data/gilt')

BUCKET_IMPORTS = 's3://cerebro-recengine-imports'
BUCKET_FIXTURES = 's3://cerebro-recengine-fixtures'
BUCKET_EXPORTS = 's3://cerebro-recengine-exports'

PREFIX = 'tiefrex'

dir_model = '/tmp/tr-ckpt-tmp'
if not os.path.exists(dir_model):
    os.mkdir(dir_model)

path_cats = os.path.join(BUCKET_EXPORTS, PREFIX, 'models/test/cat_shit.p')

# Prep for working around tf.train.latest_checkpoint with s3 directory
dir_ckpt = os.path.join(BUCKET_EXPORTS, PREFIX, 'models/test/')
s3_ckpt_uris = AwsS3Object(dir_ckpt).list_objects()
p = r'.*ckpt-(\d+)'
latest_step = max((int(m.group(1))
                   for m in (re.search(p, o.key)
                             for o in s3_ckpt_uris) if m))
s3_uris_to_dl = [o.uri for o in s3_ckpt_uris
                 if f'ckpt-{latest_step}' in o.key
                 or o.key.endswith('checkpoint')]

for uri in s3_uris_to_dl:
     o = AwsS3Object(uri)
     tmp_path = os.path.join('/tmp/tr-ckpt-tmp', os.path.basename(o.s3_uri.key))
     o.download_file(tmp_path)
     logger.info(f'Downloaded: {tmp_path}')

path_meta = glob.glob(os.path.join(dir_model, '*.meta'))[0]
path_ckpt = tf.train.latest_checkpoint(dir_model)


dir_export = os.path.join(BUCKET_EXPORTS, PREFIX, 'factors-sandbox')
