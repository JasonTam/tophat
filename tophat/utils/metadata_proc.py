import os
import pandas as pd
from typing import Dict, List, Any, Optional, Union


def write_metadata_emb(cats_d: Dict[str, List[Any]],
                       log_dir: str,
                       names_d: Optional[
                           Dict[str, Union[str, pd.DataFrame]]] = None,
                       ) -> Dict[str, str]:
    """Book-keeping and writing of human-readable metadata for embeddings 
    
    Args:
        cats_d: Dictionary of categories
        log_dir: Directory to write metadata to 
            (should be the same as the log directory of checkpoints etc)
        names_d: Dictionary of human-readable labels per element in vocab. 
            The values can either be a path to a csv file or a dataframe. The 
            index should be in the same units as stored in `cats_d`. The other 
            columns will be used as label names (can have multiple columns 
            for multiple labels).
            If `None`, the embedding projector will just use the raw id of the 
            vocab

    Returns:
        Dictionary of written metadata paths 

    """
    metas_written_d = {}
    for feat_name, cats in cats_d.items():
        path_out = os.path.join(log_dir, f'metadata-{feat_name}.tsv')
        if names_d and (feat_name in names_d):
            name_path_or_df = names_d[feat_name]
            if isinstance(name_path_or_df, str):
                names_df = pd.read_csv(
                    names_d[feat_name], index_col=feat_name)
            elif isinstance(name_path_or_df, pd.DataFrame):
                names_df = name_path_or_df
            else:
                raise ValueError('Name mapping must be path or dataframe')

            lbls_embs = names_df.reindex(cats).reset_index(drop=True)
            lbls_embs.index.name = 'index'
            lbls_embs.to_csv(path_out, sep='\t')

        else:
            # Write single column with just the raw vocab id
            lbls_embs = pd.Series(cats)
            lbls_embs.to_csv(path_out, index=False)
        metas_written_d[feat_name] = path_out
    return metas_written_d
