import pandas as pd
import os

from tophat.data import InteractionsSource

from tempfile import NamedTemporaryFile


xn_df1 = pd.DataFrame([
    ['u1', 'i1', 'a1', 1],
    ['u1', 'i1', 'a1', 2],
    ['u2', 'i2', 'a2', 1],
], columns=['user_id', 'item_id', 'activity', 'count']
)

xn_df2 = pd.DataFrame([
    ['u1', 'i1', 'a1', 1],
    ['u1', 'i1', 'a1', 2],
], columns=['user_id', 'item_id', 'activity', 'count']
)

col_params = {
    'user_col': 'user_id',
    'item_col': 'item_id',
    'activity_col': 'activity',
    'count_col': 'count',
}


def test_xn_src():
    xn = InteractionsSource(
        path=xn_df1.copy(),
        **col_params,
        activity_filter_set={'a1'},
    )
    xn.load()
    assert xn.data.equals(xn_df2)


def test_xn_fn_src():
    with NamedTemporaryFile(delete=False) as f_tmp:
        xn_df1.to_msgpack(f_tmp.name)

    xn = InteractionsSource(
        path=f_tmp.name,
        load_fn=pd.read_msgpack,
        **col_params,
        activity_filter_set={'a1'},
    )
    xn.load()
    os.remove(f_tmp.name)
    assert xn.data.equals(xn_df2)







