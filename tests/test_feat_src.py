import pandas as pd
import os
from tophat.constants import FType
from tophat.data import FeatureSource

from tempfile import NamedTemporaryFile


feat_df1 = pd.DataFrame([
    ['i1', 'f0_a', 'f1_x', 'f2_q', 'f3_q2'],
    ['i2', 'f0_b', 'f1_y', 'f2_q', 'f3_q1'],
], columns=['item_id']+[f'feat{i}' for i in range(4)],
)

feat_df2 = pd.DataFrame([
    ['f0_a', 'f2_q__f3_q2'],
    ['f0_b', 'f2_q__f3_q1'],
], columns=['feat0', 'feat2__feat3'],
    index=pd.Index(['i1', 'i2'], name='item_id')
)


def test_feat_src():
    dim = FeatureSource(
        path=feat_df1.copy(),
        feature_type=FType.CAT,
        index_col='item_id',
        use_cols=['feat0', 'feat2', 'feat3'],
        concat_cols=[('feat2', 'feat3')],
        drop_cols=['feat2', 'feat3'],
    )
    dim.load()
    assert dim.data.equals(feat_df2)


def test_feat_fn_src():
    with NamedTemporaryFile(delete=False) as f_tmp:
        feat_df1.to_msgpack(f_tmp.name)

    dim = FeatureSource(
        path=f_tmp.name,
        load_fn=pd.read_msgpack,
        feature_type=FType.CAT,
        index_col='item_id',
        use_cols=['feat0', 'feat2', 'feat3'],
        concat_cols=[('feat2', 'feat3')],
        drop_cols=['feat2', 'feat3'],
    )
    dim.load()
    os.remove(f_tmp.name)
    assert dim.data.equals(feat_df2)







