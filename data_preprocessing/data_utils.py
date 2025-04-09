import pandas as pd
import numpy as np
from logger import get_logger

logger = get_logger(__name__)

def class_proportion_analyzer(series: pd.Series, alpha: float = 1.5, return_boolean: bool = False):
    logger.info("Analyzing class proportions")
    counts = series.value_counts()
    theoretical_proportion = 1 / series.nunique()
    actual_proportions = counts / counts.sum()
    threshold = alpha * theoretical_proportion

    if return_boolean:
        result = actual_proportions < threshold
        logger.debug(f"Return boolean under threshold: {result}")
        return result
    result = actual_proportions / threshold
    logger.debug(f"Return normalized proportions: {result}")
    return result

def get_n_copies(labels: pd.Series, alpha=1.5):
    logger.info("Calculating number of oversampling copies per class")
    proportions = class_proportion_analyzer(labels, alpha=alpha, return_boolean=False)
    copies = (1 / proportions).apply(lambda x: int(np.ceil(x)))
    logger.debug(f"Copies per class: {copies}")
    return copies