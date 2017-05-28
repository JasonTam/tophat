import os

local_data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')

BUCKET_IMPORTS = 's3://cerebro-recengine-imports'
BUCKET_FIXTURES = 's3://cerebro-recengine-fixtures'
BUCKET_EXPORTS = 's3://cerebro-recengine-exports'
