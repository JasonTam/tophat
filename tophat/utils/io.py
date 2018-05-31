from pathlib import Path
from typing import Dict, Union, List, Any, Optional


def write_vocab(vocab_dir: Union[str, Path],
                cats_d: Dict[str, List[Any]],
                ):
    """Writes a dictionary of categories to vocab files
    Each line of the file will contain 1 word of the vocabulary
    """
    vocab_dir = Path(vocab_dir)
    if not vocab_dir.exists():
        vocab_dir.mkdir()
    for k, v in cats_d.items():
        with open(vocab_dir / f'{k}.vocab', 'w') as f:
            f.write('\n'.join(map(str, v)) + '\n')


def load_vocab(vocab_dir: Union[str, Path],
               pattern: Optional[str] = '*.vocab',
               ) -> Dict[str, List[Any]]:
    """Loads a dictionary of categories from a directory of vocab files
    
    Args:
        vocab_dir: directory containing vocab files
        pattern: glob pattern for finding vocab files. 
            Note: vocab files created by `write_vocab` will have `.vocab` ext

    Returns: dictionary of vocab lists

    """
    vocab_dir = Path(vocab_dir)
    # WARNING: this reads the vocab as str type (could have been Any type)
    cats_d = {}
    for vocab_path in vocab_dir.glob(pattern):
        with open(vocab_path, 'r') as f:
            v = f.read().splitlines()
            cats_d[vocab_path.stem] = v
    return cats_d

