
import pandas as pd
from .data_utils import class_proportion_analyzer, get_n_copies

def test_class_proportion_analyzer_boolean():
    s = pd.Series(['a'] * 8 + ['b'] * 1 + ['c'] * 1)
    result = class_proportion_analyzer(s, alpha=1.5, return_boolean=True)
    assert result['b'] == True
    assert result['c'] == True
    assert result['a'] == False

def test_get_n_copies():
    s = pd.Series(['a'] * 8 + ['b'] * 1 + ['c'] * 1)
    result = get_n_copies(s, alpha=1.5)
    assert result['b'] > result['a']
    assert result['c'] > result['a']