from tiefrex.config.base import *
from tiefrex.constants import *


main_cfg = {
    'local': {
        'paths_input': {
            'target_interactions': os.path.join(local_data_dir, 'train/profile/user-product_activity_counts'),
            'features': {
                'user': {
                },
                'item': {
                    'dim_products': {
                        'dtype': 'categorical',
                        'path': os.path.join(local_data_dir, 'train/dim/dim_products.msg'),
                    },
                },
            },
            'names': {
                'ops_brand_id': os.path.join(local_data_dir, 'train/dim/brand_names.csv'),
                'ops_product_category_id': os.path.join(local_data_dir, 'train/dim/pcat_names.csv'),
            }
        },
        'filter_activity_set': {b'purch'},

        'user_col': 'ops_user_id',
        'item_col': 'ops_product_id',

        'activity_col': 'activity',

        'user_specific_feature': True,
        'item_specific_feature': True,
    },
    'seed': SEED
}

