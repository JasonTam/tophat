import tensorflow as tf
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import tophat.callbacks as cbks
from tophat.data import FeatureSource, InteractionsSource
from tophat.constants import FType, FGroup
from tophat.tasks.wrapper import FactorizationTaskWrapper
from tophat.core import TophatModel
from tophat.evaluation import Validator
from tophat.utils.io import load_vocab

from tophat.datasets.movielens import fetch_movielens  # ref: lightfm

SEED = 322
EMB_DIM = 30

# Get movielens data via lightfm
data = fetch_movielens(
    indicator_features=False,
    genre_features=True,
    min_rating=4.0,  # Pretend 5-star is an implicit 'like'
    download_if_missing=True,
)

# Labels for tensorboard projector
item_lbls_df = pd.DataFrame(data['item_labels']).reset_index()
item_lbls_df.columns = ['item_id', 'item_lbls']
genre_lbls_df = pd.DataFrame([l.split(':')[-1]
                              for l in data['item_feature_labels']]
                             ).reset_index()
genre_lbls_df.columns = ['genre_id', 'genre_lbls']

names_d = {
    'item_id': item_lbls_df,
    'genre_id': genre_lbls_df,
}

# #################### [ INTERACTIONS ] ####################
xn_train = InteractionsSource(
    path=pd.DataFrame(np.vstack(data['train'].nonzero()).T,
                      columns=['user_id', 'item_id']),
    user_col='user_id',
    item_col='item_id',
)

xn_test = InteractionsSource(
    path=pd.DataFrame(np.vstack(data['test'].nonzero()).T,
                      columns=['user_id', 'item_id']),
    user_col='user_id',
    item_col='item_id',
)

# #################### [ FEATURES ] ####################

genre_df = pd.DataFrame(np.vstack(data['item_features'].nonzero()).T,
                        columns=['item_id', 'genre_id'])

# Multiple features within the same group not yet supported in tophat
# keeping only the first genre for each movie :(
genre_df.drop_duplicates('item_id', keep='first', inplace=True)

genre_feats = FeatureSource(
    path=genre_df,
    feature_type=FType.CAT,
    index_col='item_id',
    name='genre',
)

primary_group_features = {
    FGroup.USER: [],
    FGroup.ITEM: [genre_feats],
}


# ##### LOAD PRETRAINED EMBS #####
ckpt_path = 'tests/data/movielens/model.ckpt'

vocab_d = {}
# Columns to load embeddings for
transfer_cols = ['user_id', 'item_id']
for feat_name in transfer_cols:
    for scope in ['embeddings', 'biases']:
        tensor_name = f'{scope}/{feat_name}'
        vocab_file = f'tests/data/movielens/{feat_name}.vocab'
        vocab_d[tensor_name] = vocab_file

existing_cats = load_vocab('tests/data/movielens/')

# #################### [ TASKS ] ####################
opt = tf.train.AdamOptimizer(learning_rate=0.001)

primary_task = FactorizationTaskWrapper(
    loss_fn='bpr',
    sample_method='uniform_verified',
    interactions=xn_train,
    group_features=primary_group_features,
    embedding_map_kwargs={
        'embedding_dim': EMB_DIM,
        'init_emb_via_vocab': vocab_d,
        'path_checkpoint': ckpt_path,
    },
    existing_cats=existing_cats,
    add_new_cats=True,
    batch_size=128,
    optimizer=opt,
    name='primary',
)

primary_validator = Validator(
    xn_test,
    parent_task_wrapper=primary_task,
    limit_items=-1,
    n_users_eval=200,
    include_cold=False,
    cold_only=False,
    name='userXmovie',
)

cold_validator = Validator(
    xn_test,
    parent_task_wrapper=primary_task,
    limit_items=-1,
    n_users_eval=200,
    include_cold=True,
    cold_only=True,
    features_srcs=primary_group_features,
    specific_feature=defaultdict(lambda: True),
    name='userXcoldmovie',
)


if __name__ == '__main__':
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    LOG_DIR = f'/tmp/tensorboard-logs/tophat-movielens/{ts}'
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    summary_cb = cbks.Summary(log_dir=LOG_DIR)

    model = TophatModel(tasks=[primary_task])
    model.sess_init()
    primary_validator.run_val(
        model.sess, summary_cb.summary_writer, step=0, macro=False)
