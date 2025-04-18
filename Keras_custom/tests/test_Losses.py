import numpy as np
import tensorflow as tf
import sys
import pathlib

# get the path to the current file
current_file_path = pathlib.Path().resolve()
# get the parent directory
parent_directory = current_file_path.parent
# add the parent directory to the sys.path
sys.path.append(str(parent_directory))
# import the custom modules
from Losses import OverclassAwareLoss 
def test_lambda_zero():
    """
    Test if the loss behaves like class-level binary crossentropy when lambda_penalty = 0.
    """
    # Create sample target and output
    target = tf.constant([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=tf.float32)  # Example target
    output = tf.constant([[0.9, 0.1, 0, 0], [0.2, 0.8, 0, 0]], dtype=tf.float32)  # Example output

    # Define the overclass_dict and lambda_penalty = 0
    overclass_dict = {0: 2, 1: 0, 2: 1, 3: 0}  # Example mapping
    loss_fn = OverclassAwareLoss(overclass_dict, lambda_penalty=0.0)

    # Compute the loss
    loss = loss_fn(target, output)

    # Calculate the CCE (class-level) separately for comparison
    class_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(target, output))

    print("Loss when lambda_penalty = 0 (Class-level):", loss.numpy())
    print("Class-level binary crossentropy loss:", class_loss.numpy())

    # Test if both losses are approximately equal
    assert np.isclose(loss.numpy(), class_loss.numpy(), atol=1e-6), "Test failed! Losses do not match."

def test_lambda_one():
    """
    Test if the loss behaves like overclass-level binary crossentropy when lambda_penalty = 1.
    """
    # Create sample target and output
    target = tf.constant([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=tf.float32)  # Example target
    output = tf.constant([[0.9, 0.1, 0, 0], [0.2, 0.8, 0, 0]], dtype=tf.float32)  # Example output

    # Define the overclass_dict and lambda_penalty = 1
    overclass_dict = {0: 2, 1: 0, 2: 1, 3: 0}  # Example mapping
    loss_fn = OverclassAwareLoss(overclass_dict, lambda_penalty=1.0)

    # Compute the loss
    loss = loss_fn(target, output)

    # Compute the overclass-level binary crossentropy separately
    target_overclass = tf.matmul(target, loss_fn.class_to_overclass_matrix)
    output_overclass = tf.matmul(output, loss_fn.class_to_overclass_matrix)
    overclass_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(target_overclass, output_overclass))

    print("Loss when lambda_penalty = 1 (Overclass-level):", loss.numpy())
    print("Overclass-level binary crossentropy loss:", overclass_loss.numpy())

    # Test if both losses are approximately equal
    assert np.isclose(loss.numpy(), overclass_loss.numpy(), atol=1e-6), "Test failed! Losses do not match."

def test_edge_case_empty_overclass_dict():
    """
    Test if the loss handles an empty `overclass_dict`.
    """
    target = tf.constant([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=tf.float32)
    output = tf.constant([[0.9, 0.1, 0, 0], [0.2, 0.8, 0, 0]], dtype=tf.float32)
    
    # Empty overclass_dict
    overclass_dict = {}
    try:
        loss_fn = OverclassAwareLoss(overclass_dict, lambda_penalty=0.0)
    except ValueError as e:
        print(f"Error cought at instantiation: {e}")
        return
    # The loss should be calculated like regular class-level binary crossentropy
    try:
        loss = loss_fn(target, output)
    except ValueError as e:
        print(f"Error caught at loss calculation: {e}")
        return
    raise ValueError("The loss function should  raise an error with an empty overclass_dict.")

def test_lambda_intermediate():
    """
    Test if the loss behaves correctly when lambda_penalty is set to an intermediate value (e.g., 0.5).
    """
    target = tf.constant([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=tf.float32)
    output = tf.constant([[0.9, 0.1, 0, 0], [0.2, 0.8, 0, 0]], dtype=tf.float32)

    overclass_dict = {0: 2, 1: 0, 2: 1, 3: 0}
    loss_fn = OverclassAwareLoss(overclass_dict, lambda_penalty=0.5)

    # Compute the loss
    loss = loss_fn(target, output)
    print("Loss when lambda_penalty = 0.5:", loss.numpy())

def test_mismatched_shapes():
    """
    Test if the loss function raises an error when target and output have mismatched shapes.
    """
    target = tf.constant([[1, 0, 0, 0]], dtype=tf.float32)  # Shape (1, 4)
    output = tf.constant([[0.9, 0.1, 0, 0], [0.2, 0.8, 0, 0]], dtype=tf.float32)  # Shape (2, 4)

    overclass_dict = {0: 2, 1: 0, 2: 1, 3: 0}
    try:
        loss_fn = OverclassAwareLoss(overclass_dict, lambda_penalty=0.5)
    
    except ValueError as e:
        print(f"Error caught at instantiation: {e}")
        return

    try:
        loss = loss_fn(target, output)
    except ValueError as e:
        print(f"Error of mismatched shapes caught at loss calculation: {e}")
        return
    raise ValueError("The loss function should raise an error with mismatched shapes.")

def test_random_inputs():
    """
    Test the loss function with random inputs.
    """
    target = tf.random.uniform((5, 4), minval=0, maxval=2, dtype=tf.int32)
    output = tf.random.uniform((5, 4), minval=0, maxval=2, dtype=tf.float32)

    overclass_dict = {0: 2, 1: 0, 2: 1, 3: 0}
    loss_fn = OverclassAwareLoss(overclass_dict, lambda_penalty=0.5)

    loss = loss_fn(target, output)
    print("Random input loss:", loss.numpy())

from Losses import OverclassWeightedLoss 
def test_loss_lambda_zero_behaves_like_class_loss():
    target = tf.constant([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=tf.float32)
    output = tf.constant([[0.9, 0.1, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]], dtype=tf.float32)

    overclass_dict = {0: 2, 1: 0, 2: 1, 3: 0}
    weight_dict = {0: 0.3, 1: 0.3, 2: 0.4}
    
    loss_fn = OverclassWeightedLoss(overclass_dict, weight_dict, lambda_penalty=0.0)
    loss = loss_fn(target, output)

    expected = tf.reduce_mean(tf.keras.losses.binary_crossentropy(target, output))
    assert np.isclose(loss.numpy(), expected.numpy(), atol=1e-6), "Lambda=0 should reduce to class-level loss."


def test_weight_dict_normalization():
    overclass_dict = {0: 0, 1: 1, 2: 2}
    weight_dict = {0: 10.0, 1: 20.0, 2: 30.0}  # Not normalized

    loss_fn = OverclassWeightedLoss(overclass_dict, weight_dict, lambda_penalty=1.0)

    # Check normalization
    total = sum(loss_fn.weight_dict.values())
    assert np.isclose(total, 1.0, atol=1e-6), f"Weights should be normalized, got total {total}"


def test_invalid_weight_dict_raises_error():
    try:
        OverclassWeightedLoss(overclass_dict={0: 0, 1: 1}, weight_dict={0: 1.0})  # Missing weight for class 1
    except ValueError as e:
        print("Caught expected error:", e)
        return
    raise AssertionError("Expected ValueError due to missing weight")


def test_invalid_overclass_dict_type():
    try:
        OverclassWeightedLoss(overclass_dict={"a": 0}, weight_dict={0: 1.0})
    except ValueError as e:
        print("Caught expected error:", e)
        return
    raise AssertionError("Expected ValueError for invalid overclass_dict keys")


def test_mismatched_shapes_raise_error():
    target = tf.constant([[1, 0, 0, 0]], dtype=tf.float32)
    output = tf.constant([[0.9, 0.1, 0.0, 0.0], [0.2, 0.8, 0.0, 0.0]], dtype=tf.float32)

    overclass_dict = {0: 2, 1: 0, 2: 1, 3: 0}
    weight_dict = {0: 0.3, 1: 0.3, 2: 0.4}

    loss_fn = OverclassWeightedLoss(overclass_dict, weight_dict, lambda_penalty=0.5)
    
    try:
        loss = loss_fn(target, output)
    except ValueError as e:
        print("Caught expected shape mismatch error:", e)
        return
    raise AssertionError("Expected ValueError for shape mismatch")


def test_random_input_loss_runs():
    target = tf.cast(tf.random.uniform((10, 4), minval=0, maxval=2, dtype=tf.int32), tf.float32)
    output = tf.random.uniform((10, 4), minval=0, maxval=1, dtype=tf.float32)

    overclass_dict = {0: 0, 1: 1, 2: 2, 3: 1}
    weight_dict = {0: 0.3, 1: 0.3, 2: 0.4}

    loss_fn = OverclassWeightedLoss(overclass_dict, weight_dict, lambda_penalty=0.7)
    loss = loss_fn(target, output)
    print("Random input test passed. Loss:", loss.numpy())
    
    
import tensorflow as tf
import numpy as np

def test_loss_with_dummy_model():
    # Setup
    num_classes = 6
    batch_size = 8
    input_dim = 10

    # Overclass mapping
    overclass_dict = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
    weight_dict = {0: 0.2, 1: 0.3, 2: 0.5}
    loss_fn = OverclassWeightedLoss(overclass_dict, weight_dict, lambda_penalty=0.5)
    loss_fn2 = OverclassAwareLoss(overclass_dict, lambda_penalty=0.5)
    # Dummy data
    X = np.random.rand(batch_size, input_dim).astype(np.float32)
    y = np.random.randint(0, 2, size=(batch_size, num_classes)).astype(np.float32)

    # Simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])

    # Compile and train
    model.compile(optimizer='adam', loss=loss_fn)
    model.fit(X, y, epochs=2, verbose=1)

    model.compile(optimizer='adam', loss=loss_fn2)
    model.fit(X, y, epochs=2, verbose=1)
    print("✅ Dummy model test passed!")

