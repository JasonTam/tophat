import os
import pandas as pd
from typing import Dict, List, Any


def write_metadata_emb(cats_d: Dict[str, List[Any]],
                       path_names_d: Dict[str, str],
                       log_dir: str) -> Dict[str, str]:
    """Book-keeping and writing of human-readable metadata for embeddings 
    
    Args:
        cats_d: Dictionary of categories
        path_names_d: Dictionary of human-readable labels
        log_dir: Directory to write metadata to
            (should be the same as the log directory of checkpoints etc)

    Returns:
        Dictionary of written metadata paths 

    """
    metas_written_d = {}
    for feat_name, cats in cats_d.items():
        path_out = os.path.join(log_dir, f'metadata-{feat_name}.tsv')
        if feat_name in path_names_d:
            names_df = pd.read_csv(
                path_names_d[feat_name], index_col=feat_name)
            lbls_embs = names_df.reindex(cats).reset_index(drop=True)
            lbls_embs.index.name = 'index'
            lbls_embs.to_csv(path_out, sep='\t')
        else:
            # Write single column
            lbls_embs = pd.Series(cats)
            lbls_embs.to_csv(path_out, index=False)
        metas_written_d[feat_name] = path_out
    return metas_written_d
