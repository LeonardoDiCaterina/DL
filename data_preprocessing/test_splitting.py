import pandas as pd
from splitting import create_split



def test_create_split_shapes():
    df = pd.DataFrame({
        'file_path': [f'f{i}.jpg' for i in range(100)],
        'phylum': ['a'] * 50 + ['b'] * 50
    })
    folds, test_df = create_split(df['file_path'], df['phylum'], n_folds=4, test_ratio=0.2)
    assert len(test_df) == 20  # 20% of 20
    assert len(folds) == 5
    assert all(isinstance(f, pd.DataFrame) for f in folds)
    for fold in folds:
        assert len(fold) == 20  # 80% of 100 / 4
        assert len(fold['phylum'].unique()) == 2  # Both classes should be present
    assert all(fold['phylum'].value_counts(normalize=True)['a'] >=  0.4)  
    assert all(fold['phylum'].value_counts(normalize=True)['b'] >=  0.4)