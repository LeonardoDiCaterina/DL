import pandas as pd
from splitting import create_split

def test_create_split_shapes():
    df = pd.DataFrame({
        'file_path': [f'f{i}.jpg' for i in range(20)],
        'phylum': ['a'] * 10 + ['b'] * 10
    })
    folds, test_df = create_split(df['file_path'], df['phylum'], n_folds=5, test_ratio=0.2)
    assert len(test_df) == 4  # 20% of 20
    assert len(folds) == 5
    assert all(isinstance(f, pd.DataFrame) for f in folds)