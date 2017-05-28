import os
import pandas as pd
from typing import Dict, List, Any


def write_metadata_emb(cats_d: Dict[str, List[Any]],
                       path_names_d: Dict[str, str],
                       log_dir: str) -> Dict[str, str]:
    metas_written_d = {}
    for feat_name, cats in cats_d.items():
        if feat_name in path_names_d:
            path_out = os.path.join(log_dir, f'metadata-{feat_name}.tsv')
            names_df = pd.read_csv(path_names_d[feat_name], index_col=feat_name)
            lbls_embs = names_df.loc[cats].reset_index(drop=True)
            lbls_embs.index.name = 'index'
            lbls_embs.to_csv(path_out, sep='\t')
            metas_written_d[feat_name] = path_out
    return metas_written_d
