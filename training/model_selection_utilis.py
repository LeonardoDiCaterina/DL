# training/train_model.py
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_preprocessing.splitting import split_folds_to_train_val
from data_preprocessing.logger import get_logger
from functools import reduce
from tabulate import tabulate
import gc # Garbage collector interface

import itertools 
from training.training_config import NUM_CLASSES, INPUT_SHAPER, N_EPOCHS_4CV, LOG_LEVEL

logger = get_logger(__name__)




from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D, RandomFlip, RandomContrast, Lambda)

def random_saturation(x):
        return tf.image.random_saturation(x, lower=0.8, upper=1.2)

def build_model(modelname = 'MobileNetV2', freeze_until = 0, dense_layers = [], input_shape = (256, 256, 3), num_classes = 202, dropout_rate=0.5):
    """
    Builds and returns an image classification model based on MobileNetV2 with optional fine-tuning and a custom dense head

    Args:
        modelname (str): Name of the base model to use. Options are 'MobileNetV2' and 'ResNet50'.
        freeze_until (int): Index of the layer until which to freeze the MobileNetV2 base (exclusive).
        dense_layers (list of int): List specifying the number of units for each dense layer after the base model.
        input_shape (tuple): Shape of the input images, e.g., (height, width, channels).
        num_classes (int): Number of output classes for the final classification layer.
        dropout_rate (float, optional): Dropout rate applied after each dense layer. Default is 0.5.

    Returns:
        keras.Model: A compiled Keras model ready for training.
    """

    inputs = Input(shape=input_shape)

    logger.info(f'Building model with input shape: {input_shape}')
    # Data augmentation block
    x = RandomFlip("horizontal")(inputs)
    x = RandomContrast(0.1)(x)
    x = Lambda(random_saturation, name="random_saturation")(x)

    # Base model
    if modelname == 'MobileNetV2':
      base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=x)
      logger.info(f'Using {modelname} as base model')
    elif modelname == 'ResNet50':
      base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_tensor=x)
      logger.info(f'Using {modelname} as base model')      
    else:
        raise ValueError(f"Unsupported model name: {modelname}. Supported models are 'MobileNetV2' and 'ResNet50'.")
    # Freeze desired layers
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
        
    logger.info(f'Freezing layers up to index {freeze_until} in the base model')
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True
    logger.info(f'Unfreezing layers from index {freeze_until} in the base model')

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Dense Layers 
    if dense_layers:
        logger.info(f'Adding {len(dense_layers)} dense layers with dropout rate {dropout_rate}')
        for units in dense_layers:
            x = Dense(units, activation='relu')(x)
            x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    logger.info(f'Model summary:\n{model.summary()}')
    return model

def build_param_grid(modelnames, freeze_options, dense_options, learning_rates):
    """
    Builds a parameter grid for model configurations.

    Args:
        modelnames (list): List of model names.
        freeze_options (list): List of freeze options.
        dense_options (list): List of dense layer configurations.
        learning_rates (list): List of learning rates.

    Returns:
        pd.DataFrame: A DataFrame containing the parameter grid.
    """
    grid = list(itertools.product(modelnames, freeze_options, dense_options, learning_rates))
    return grid



def evaluated_cross_val(train_folds, input_shape, num_classes,
                       param_grid, epochs=5):
    """
    Evaluates different model configurations using cross-validation.

    Args:
        train_folds (list): List of training datasets for cross-validation.
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of output classes.
        param_grid (list): List of model configurations to evaluate.
    """
    models_list = []
    results = []
    num_folds = len(train_folds)

    for config in param_grid:
        modelname, freeze_until, dense_layers, lr = config
        fold_val_accuracies = []
        fold_train_accuracies = []

        print(f"\nEvaluating {modelname}: freeze_until={freeze_until}, dense_layers={dense_layers}, lr={lr}")

        for k in range(num_folds):
            val_ds = train_folds[k]
            train_ds = reduce(lambda x, y: x.concatenate(y),
                              [fold for i, fold in enumerate(train_folds) if i != k])

            try:
                model = build_model(modelname, freeze_until, dense_layers, input_shape, num_classes)
            except ValueError:
                print(f"Invalid configuration for {modelname}: freeze_until={freeze_until}, dense_layers={dense_layers}, lr={lr}")
                break

            model.compile(optimizer=Adam(learning_rate=lr),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

            early_stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)
            history = model.fit(train_ds,
                                validation_data=val_ds,
                                epochs=epochs,
                                verbose=0,
                                callbacks=[early_stop])

            val_acc = history.history['val_accuracy'][-1]
            train_acc = history.history['accuracy'][-1]

            print(f"  Fold {k+1} - Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            fold_val_accuracies.append(val_acc)
            fold_train_accuracies.append(train_acc)

            # Free memory as I just topped 50 Gigas of RAM (silly me)
            try:
                del model
                del history
                del train_ds
                del val_ds
                gc.collect()
            except Exception as e:
                print(f"Error during memory cleanup: {e}")

        if not fold_val_accuracies:
            print(f"No valid configuration for {modelname}: freeze_until={freeze_until}, dense_layers={dense_layers}, lr={lr}")
            continue
        mean_val_acc = np.mean(fold_val_accuracies)
        mean_train_acc = np.mean(fold_train_accuracies)

        model_entry = {
            'configuration': config,
            'mean_val_accuracy': mean_val_acc,
            'mean_train_accuracy': mean_train_acc,
            'overfit_gap': mean_train_acc - mean_val_acc
        }
        models_list.append(model_entry)

        results.append({
            'Model': modelname,
            'Freeze Until': freeze_until,
            'Dense Layers': dense_layers,
            'Learning Rate': lr,
            'Train Accuracy': mean_train_acc,
            'Val Accuracy': mean_val_acc,
            'Overfit Gap': mean_train_acc - mean_val_acc
        })

    results_df = pd.DataFrame(results)

    # Pretty print results in a fancy fancy way
    print("\nSummary of Configurations:")
    print(tabulate(results_df, headers='keys', tablefmt='fancy_grid', floatfmt=".4f"))

    return models_list, results_df










def train_model(model, train_folds, val_fold_index, loss_fn, metrics, epochs=10, batch_size=32, image_size=(256, 256)):
    """
    Trains the model on the training dataset and evaluates it on the validation dataset.
    
    Parameters:
    - model: Keras model to train.
    - train_folds: List of training folds (from load_data()).
    - val_fold_index: Index of the fold to use for validation.
    - loss_fn: Loss function instance.
    - metrics: List of metric instances.
    - epochs: Number of epochs to train.
    - batch_size: Batch size for training.
    - image_size: Image size to resize the dataset.
    
    Returns:
    - history: History object containing training metrics.
    """
    # Split the folds into train and validation datasets
    logger.info(f"Splitting folds: Using fold {val_fold_index} for validation")
    train_ds, val_ds = split_folds_to_train_val(train_folds, val_fold_index)
    
    # Model compilation
    logger.info(f"Compiling model with loss: {loss_fn}, metrics: {metrics}")
    model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)
    
    #check if train_ds is of dimension (10,)
    

    # Model training
    logger.info(f"Starting training for {epochs} epochs")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Plot the training history
    logger.info("Plotting training history")
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy (if available)
    if 'accuracy' in history.history:
        plt.subplot(2, 1, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

    plt.tight_layout()
    plt.show()

    logger.info("Training completed")
    return history

