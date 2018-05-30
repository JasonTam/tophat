import tensorflow as tf
import numpy as np
import pandas as pd

import tophat.callbacks as cbks
from tophat.data import FeatureSource, InteractionsSource
from tophat.constants import FType, FGroup
from tophat.tasks.wrapper import FactorizationTaskWrapper
from tophat.core import TophatModel
from tophat.evaluation import Validator

from tophat.datasets.movielens import fetch_movielens  # ref: lightfm

SEED = 322
np.random.seed(SEED)

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


xn_train_df = pd.DataFrame(np.vstack(data['train'].nonzero()).T,
                           columns=['user_id', 'item_id'])
xn_test_df = pd.DataFrame(np.vstack(data['test'].nonzero()).T,
                          columns=['user_id', 'item_id'])
genre_df = pd.DataFrame(np.vstack(data['item_features'].nonzero()).T,
                        columns=['item_id', 'genre_id'])

# #################### [ INTERACTIONS ] ####################

xn_train = InteractionsSource(
    path=xn_train_df,
    user_col='user_id',
    item_col='item_id',
)

xn_test = InteractionsSource(
    path=xn_test_df,
    user_col='user_id',
    item_col='item_id',
)


def synthesize_genre_favs(xn_train_df):
    """
    Making synthetic user-genre favorite interactions

    We're going to just count the genres watched by each user.
    Subsample from a random top percentile of genres and
    consider those the user's favorites. We will then subsample again
    -- simulating the voluntary aspect of favoriting a genre.
    """
    def sample_fav(df, q_thresh=None, frac=None):
        q_thresh = q_thresh or np.random.rand()
        frac = frac or np.random.rand()
        return df.reset_index().genre_id \
            .loc[(df.item_id >= df.item_id.quantile(q_thresh)).values] \
            .sample(frac=frac, replace=False)

    n_users = xn_train_df['user_id'].nunique()

    genre_counts = xn_train_df.groupby(('user_id', 'genre_id')).count()

    xn_genre_favs = genre_counts.groupby(level=0) \
        .apply(sample_fav) \
        .reset_index().drop('level_1', axis=1)

    # say 0.7 users know of the genre favoriting feature
    aware_users = set(np.random.permutation(n_users)[:int(0.7 * n_users)])

    xn_genre_favs_samp = xn_genre_favs.loc[
        xn_genre_favs.user_id.isin(aware_users)]

    return xn_genre_favs_samp


xn_train_df = xn_train_df.merge(genre_df, on='item_id')
xn_genre_favs_df = synthesize_genre_favs(xn_train_df)
# Synthetic Genre Favorites Interactions
xn_genre_favs = InteractionsSource(
    path=xn_genre_favs_df,
    user_col='user_id',
    item_col='genre_id',
)


# #################### [ FEATURES ] ####################

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
    add_new_cats=True,
    build_on_init=False,
    name='primary',
)

genre_task = FactorizationTaskWrapper(
    loss_fn='bpr',
    sample_method='uniform_verified',
    interactions=xn_genre_favs,
    parent_task_wrapper=primary_task,
    batch_size=128,
    optimizer=opt,
    add_new_cats=True,
    build_on_init=False,
    name='genre',
)

primary_validator = Validator(
    xn_test,
    parent_task_wrapper=primary_task,
    **{
        'limit_items': -1,
        'n_users_eval': 200,
        'include_cold': False,
        'cold_only': False
    },
    name='userXmovie',
)


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

