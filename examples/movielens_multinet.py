import tensorflow as tf
import numpy as np
import pandas as pd
import os

import tophat.callbacks as cbks
from tophat.data import FeatureSource, InteractionsSource
from tophat.constants import FType, FGroup
from tophat.tasks.wrapper import FactorizationTaskWrapper
from tophat.core import TophatModel
from tophat.evaluation import Validator

from lightfm.datasets.movielens import fetch_movielens

SEED = 322

# Get movielens data via lightfm
data = fetch_movielens(
    indicator_features=False,
    genre_features=True,
    min_rating=5.0,  # Pretend 5-star is an implicit 'like'
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

# Synthetic Genre Favorites Interactions
xn_genre_favs = InteractionsSource(
    path=os.path.join(os.path.dirname(os.path.realpath(__file__)),
                      '../data/movielens', 'genre_favs_synthetic.msg'),
    user_col='user_id',
    item_col='genre_id',
    load_fn=pd.read_msgpack,
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

# #################### [ TASKS ] ####################
opt = tf.train.AdamOptimizer(learning_rate=0.001)
EMB_DIM = 30

primary_task = FactorizationTaskWrapper(
    loss_fn='bpr',
    sample_method='uniform',
    interactions=xn_train,
    group_features=primary_group_features,
    embedding_map_kwargs={
        'embedding_dim': EMB_DIM,
    },
    batch_size=128,
    optimizer=opt,
    name='primary',
)

genre_task = FactorizationTaskWrapper(
    loss_fn='bpr',
    sample_method='uniform_verified',
    interactions=xn_genre_favs,
    existing_cats_d=primary_task.data_loader.cats_d,
    embedding_map=primary_task.embedding_map,
    batch_size=128,
    optimizer=opt,
    name='genre',
)

primary_validator = Validator(
    {'interactions_val': xn_test},
    primary_task.data_loader,
    primary_task.task,
    **{
        'limit_items': -1,
        'n_users_eval': 200,
        'include_cold': False,
        'cold_only': False
    },
    name='userXmovie',
)
primary_validator.make_ops(primary_task.task)


if __name__ == '__main__':
    LOG_DIR = '/tmp/tensorboard-logs/tophat-movielens'

    model = TophatModel(tasks=[primary_task, genre_task])

    summary_cb = cbks.Summary(log_dir=LOG_DIR)
    emb_cb = cbks.Projector(log_dir=LOG_DIR,
                            embedding_map=model.embedding_map,
                            summary_writer=summary_cb.summary_writer,
                            names_d=names_d)
    val_cb = cbks.Scorer(primary_validator,
                         summary_writer=summary_cb.summary_writer,
                         freq=5,)
    saver_cb = cbks.ModelSaver(LOG_DIR)
    callbacks = [
        summary_cb,
        emb_cb,
        val_cb,
        saver_cb,
    ]

    model.fit(10, callbacks=callbacks, verbose=3)

