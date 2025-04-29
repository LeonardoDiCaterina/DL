from functools import reduce
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from .logger import get_logger
from .preprocessing_config import LABEL_COL

logger = get_logger(__name__)

def create_split(paths, targets, n_folds=5, test_ratio=0.2,random_state=42):
    
    logger.info("Splitting dataset into train/test and folds")
    train_paths, test_paths, train_targets, test_targets = train_test_split(
        paths, targets, test_size=test_ratio, stratify=targets, random_state=random_state
    )

    logger.debug(f"Train size: {len(train_paths)}, Test size: {len(test_paths)}")
    
    train_df = pd.DataFrame({'file_path': train_paths, f'{LABEL_COL}': train_targets})
    test_df = pd.DataFrame({'file_path': test_paths, f'{LABEL_COL}': test_targets})

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = [train_df.iloc[test_index] for _, test_index in skf.split(train_df, train_df[f'{LABEL_COL}'])]

    logger.info(f"Created {n_folds} stratified folds")
    return folds, test_df



def split_folds_to_train_val(train_folds: list, val_fold_index: int):
    """Splits the train folds into training and validation datasets based on the specified fold index."""
    
    # Validation fold
    val_ds = train_folds[val_fold_index]
    logger.info(f"Validation dataset length: {len(val_ds)}")
    
    # Training folds (exclude the validation fold)
    train_folds_selected = [train_folds[i] for i in range(len(train_folds)) if i != val_fold_index] 
    
    # Use reduce to concatenate the training folds
    train_ds = reduce(lambda x, y: x.concatenate(y), train_folds_selected)
    logger.info(f"Training dataset shape: {train_ds.element_spec}")
    logger.info(f"Training dataset length: {len(train_ds)}")    
    return train_ds, val_ds