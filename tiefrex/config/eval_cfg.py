from tiefrex.config.base import *
from tiefrex.constants import *


eval_cfg = {
    'local': {
        'eval_interactions': os.path.join(local_data_dir, 'val/profile/user-product_activity_counts'),
        'filter_activity_set': {b'purch'},
        'user_col': 'ops_user_id',
        'item_col': 'ops_product_id',

        'activity_col': 'activity',
    },
    'seed': SEED
}

