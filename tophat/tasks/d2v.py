# TODO: needs to be standardized with BaseTask

import pandas as pd
import numpy as np
import os
from tophat.constants import SEED
from config.common import *

from gensim.models.doc2vec import TaggedDocument, FAST_VERSION
from gensim.models import Doc2Vec

import fastavro as avro
import itertools as it
from tophat.schemas import factors_avro
from typing import Iterator, Dict, Any, Callable, Tuple

import multiprocessing
n_cores = multiprocessing.cpu_count()

assert FAST_VERSION > -1, "This will be painfully slow otherwise"


def fit_interactions(interactions_df: pd.DataFrame,
                     user_col: str='ops_user_id',
                     item_col: str='ops_product_id',
                     emb_dim: int=16,
                     ):
    # Note: gensim requires the tag(s) and words to be str for silly reasons
    # For now, the only doctag will be the user_id,
    # but you can imagine user features being tags
    interactions_df[item_col+'str'] = interactions_df[item_col].astype(str)

    docs = interactions_df\
        .groupby(user_col)[item_col+'str'].apply(list).reset_index()\
        .apply(lambda row: TaggedDocument(row[item_col+'str'],
                                          tags=[str(row[user_col])]),
               axis=1)\
        .tolist()

    interactions_df.drop(item_col+'str', axis=1, inplace=True)

    model = Doc2Vec(docs,
                    size=emb_dim,
                    dm=1,
                    window=4, min_count=5, negative=5,
                    iter=10,
                    workers=1,  # should be `n_cores`, but see issue gensim#336
                    seed=SEED,
                    )

    return model


def model_to_dfs(d2v_model) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dv = d2v_model.docvecs
    wv = d2v_model.wv

    user_keys = list(dv.doctags.keys())
    user_df = pd.DataFrame(index=pd.Index(user_keys, name='id', dtype=str))
    user_df.reset_index(inplace=True)  # downstream expects `id` column
    user_df['factors'] = dv[user_keys].astype(np.float32).tolist()
    user_df['bias'] = 0.

    item_keys = list(wv.vocab.keys())
    item_df = pd.DataFrame(index=pd.Index(item_keys, name='id', dtype=str))
    item_df.reset_index(inplace=True)
    item_df['factors'] = [wv.word_vec(w) for w in item_keys]
    item_df['bias'] = 0.

    return user_df, item_df


def rec_generator(keys: Iterator,
                  get_vec_fn: Callable,
                  partition: int = 0,
                  n_partitions: int = 1) -> Iterator[Dict[str, Any]]:
    sliced_keys = it.islice(keys, partition, None, n_partitions)
    for k in sliced_keys:
        record = {
            'id': str(k),
            'factors': list(get_vec_fn(k)),
            'bias': 0.,
        }

        yield record


dv_rec_generator = lambda dv, **kwargs: rec_generator(
    keys=dv.doctags.keys(), get_vec_fn=lambda doctag: dv[doctag], **kwargs
)

wv_rec_generator = lambda wv, **kwargs: rec_generator(
    keys=wv.vocab.keys(), get_vec_fn=wv.word_vec, **kwargs
)


def export(d2v_model: Doc2Vec, dir_export: str):
    # Record generators
    user_rec_gen = dv_rec_generator(d2v_model.docvecs)
    item_rec_gen = wv_rec_generator(d2v_model.wv)

    with open(os.path.join(dir_export, 'user_docvecs.avro'), 'wb') as f_out:
        avro.writer(f_out, factors_avro, user_rec_gen, codec='snappy')

    with open(os.path.join(dir_export, 'item_wordvecs.avro'), 'wb') as f_out:
        avro.writer(f_out, factors_avro, item_rec_gen, codec='snappy')
