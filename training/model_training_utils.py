# training/train_model.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_preprocessing.splitting import split_folds_to_train_val
from data_preprocessing.logger import get_logger
from functools import reduce
from tabulate import tabulate

#from training.training_config import NUM_CLASSES, INPUT_SHAPE, N_EPOCHS_4FULL_TRAIN
NUM_CLASSES = 202
INPUT_SHAPE = (256, 256, 3)
N_EPOCHS_4FULL_TRAIN = 10 #number of epochs for test set
from tf.keras.optimizers import Adam
from tf.keras.callbacks import EarlyStopping
from training.model_config import build_model

logger = get_logger(__name__)

def full_train(train_folds, test_ds, param_grid, num_classes = NUM_CLASSES, input_shape = INPUT_SHAPE,
                        epochs=N_EPOCHS_4FULL_TRAIN, verbose= False, plot=True):
    """
    Evaluates different model configurations using cross-validation.

    Args:
        train_folds (list): List of training datasets for cross-validation.
        param_grid (list): List of model configurations to evaluate.
        num_classes (int): Number of output classes.
        input_shape (tuple): Shape of the input images.

    """
    modelname, freeze_until, dense_layers, lr = param_grid


    print(f"\nTraining {modelname}: freeze_until={freeze_until}, dense_layers={dense_layers}, lr={lr}")

    train_ds = reduce(lambda x, y: x.concatenate(y),[fold for fold in train_folds])

    try:
        model = build_model(modelname, freeze_until, dense_layers, input_shape, num_classes)
    except ValueError:
        print(f"Invalid configuration for {modelname}: freeze_until={freeze_until}, dense_layers={dense_layers}, lr={lr}")

    model.compile(optimizer=Adam(learning_rate=lr),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
    history = model.fit(train_ds,
                        validation_data=test_ds,
                        epochs=epochs,
                        verbose=0,
                        callbacks=[early_stop])


    if plot:
        plt.figure(figsize=(12, 8))
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'{modelname} - Freeze Until {freeze_until} - Dense Layers {dense_layers}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    if verbose:
        print(f"Training {modelname}: freeze_until={freeze_until}, dense_layers={dense_layers}, lr={lr}")
        print(f"Train Accuracy: {history.history['accuracy'][-1]}")
        print(f"Validation Accuracy: {history.history['val_accuracy'][-1]}")
        print(f"Model Summary:\n{model.summary()}")

    return model, history

