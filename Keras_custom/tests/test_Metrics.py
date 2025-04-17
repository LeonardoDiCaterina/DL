import sys
import pathlib
# get the path to the current file
current_file_path = pathlib.Path().resolve()
# get the parent directory
parent_directory = current_file_path.parent
# add the parent directory to the sys.path
sys.path.append(str(parent_directory))
# import the custom modules
from Metrics import DistanceQuantile

import pytest
import numpy as np
import tensorflow as tf

@pytest.mark.parametrize("y_pred, expected_quantile", [
    ([0.0, 0.25, 0.5, 0.75, 1.0], 0.25),  # distances: [0.5, 0.25, 0.0, 0.25, 0.5] → 25th percentile = 0.25
    ([0.5, 0.5, 0.5, 0.5], 0.0),           # all distances are 0
    ([0.0, 1.0], 0.5),                     # distances: [0.5, 0.5]
])
def test_distance_quantile_correctness(y_pred, expected_quantile):
    metric = DistanceQuantile(quantile_level=0.25)
    y_true = tf.zeros_like(y_pred)  # ignored
    metric.update_state(y_true=tf.constant(y_true, dtype=tf.float32),
                        y_pred=tf.constant(y_pred, dtype=tf.float32))
    result = metric.result().numpy()
    assert np.isclose(result, expected_quantile, atol=1e-5)

def test_reset_states():
    metric = DistanceQuantile()
    y_pred = tf.constant([0.0, 1.0], dtype=tf.float32)
    y_true = tf.zeros_like(y_pred)
    metric.update_state(y_true, y_pred)
    assert metric.result().numpy() > 0.0
    metric.reset_states()
    assert metric.result().numpy() == 0.0

def test_tf_integration():
    # Use tf.function to simulate graph mode execution
    metric = DistanceQuantile()

    def run_metric():
        y_pred = tf.constant([0.2, 0.5, 0.8], dtype=tf.float32)
        y_true = tf.constant([0, 1, 0], dtype=tf.float32)
        metric.update_state(y_true, y_pred)
        return metric.result()

    result = run_metric()
    result = tf.convert_to_tensor(metric.result())
    assert isinstance(result, tf.Tensor)
    
def test_metric_per_batch_logging_with_expected():

    # Dummy data (4 samples → 4 batches of 1)
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0, 1, 0, 1], dtype=np.float32)

    # Build simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(1,))
    ])

    # Use your custom metric
    metric = DistanceQuantile(quantile_level=0.25)

    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=[metric])

    # Manual per-batch training
    for i in range(len(X)):
        x_batch = X[i:i+1]
        y_batch = y[i:i+1]

        # Train on single batch
        model.train_on_batch(x_batch, y_batch)

        # Get prediction
        y_pred_batch = model.predict(x_batch, verbose=0).flatten()
        distances = np.abs(y_pred_batch - 0.5)
        expected = np.percentile(distances, 25)

        # Compare with metric
        actual = metric.result().numpy()

        print(f"Batch {i} | y_true: {y_batch} | y_pred: {y_pred_batch}")
        print(f"→ Expected 25th percentile: {expected:.5f} | Metric value: {actual:.5f}")
        print("-" * 50)
        
from Metrics import MulticlassDistanceQuantile


def test_multiclass_reset_states():
    metric = MulticlassDistanceQuantile()
    y_pred = tf.constant([[0.7, 0.2, 0.1]], dtype=tf.float32)
    y_true = tf.one_hot([0], depth=3)
    metric.update_state(y_true, y_pred)
    assert metric.result().numpy() > 0.0
    metric.reset_states()
    assert metric.result().numpy() == 0.0

def test_multiclass_tf_integration():
    metric = MulticlassDistanceQuantile()

    def run_metric():
        y_pred = tf.constant([[0.2, 0.5, 0.3]], dtype=tf.float32)
        y_true = tf.one_hot([1], depth=3)
        metric.update_state(y_true, y_pred)
        return metric.result()

    result = run_metric()
    result = tf.convert_to_tensor(result)
    assert isinstance(result, (tf.Tensor,tf.Variable))

def test_multiclass_per_batch_logging():
    # Dummy multiclass classification
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0, 1, 2, 1], dtype=np.int32)
    y_onehot = tf.one_hot(y, depth=3)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, activation='softmax', input_shape=(1,))
    ])

    metric = MulticlassDistanceQuantile(quantile_level=0.25)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=[metric])

    for i in range(len(X)):
        x_batch = X[i:i+1]
        y_batch = y_onehot[i:i+1]

        model.train_on_batch(x_batch, y_batch)
        y_pred_batch = model.predict(x_batch, verbose=0)

        true_class = tf.argmax(y_batch, axis=1).numpy()[0]
        true_prob = y_pred_batch[0][true_class]
        expected = np.percentile([np.abs(true_prob - 0.5)], 25)

        actual = metric.result().numpy()
        print(f"Batch {i} | y_true: {true_class} | y_pred: {y_pred_batch}")
        print(f"→ Expected 25th percentile: {expected:.5f} | Metric value: {actual:.5f}")
        print("-" * 50)
  
