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

