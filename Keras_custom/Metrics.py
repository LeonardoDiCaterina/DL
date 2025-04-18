import tensorflow as tf
import numpy as np


# DistanceQuantile

class DistanceQuantile(tf.keras.metrics.Metric):
    def __init__(self, quantile_level=0.25, name='distance_quantile', **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantile_level = quantile_level
        self.last_value = self.add_weight(name='last_value', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred)
        y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
        distances = tf.abs(y_pred - 0.5)

        n = tf.shape(distances)[0]
        def compute_quantile():
            sorted_distances = tf.sort(distances)
            q_index = tf.cast(tf.round(self.quantile_level * tf.cast(n - 1, tf.float32)), tf.int32)
            return tf.gather(sorted_distances, q_index)

        quantile_value = tf.cond(n > 0, compute_quantile, lambda: tf.constant(0.0))
        self.last_value.assign(tf.reshape(quantile_value, []))

    def result(self):
        return self.last_value

    def reset_states(self):
        self.last_value.assign(0.0)
        
        
# MulticlassDistanceQuantile

class MulticlassDistanceQuantile(tf.keras.metrics.Metric):
    def __init__(self, quantile_level=0.25, name='25th_quantile', **kwargs):
        
        self.quantile_level = quantile_level
        if quantile_level < 0.0 or quantile_level > 1.0:
            raise ValueError("quantile_level must be between 0.0 and 1.0")
        
        if not isinstance(quantile_level, float):
            raise TypeError("quantile_level must be a float")
        if quantile_level == 0.0:
            raise ValueError("quantile_level cannot be 0.0")
        if quantile_level == 1.0:
            raise ValueError("quantile_level cannot be 1.0")
        if quantile_level != 0.25:
            name = f"{(quantile_level * 100):.2f }th_quantile"
        super().__init__(name=name, **kwargs)
        
        self.last_value = self.add_weight(name='last_value', initializer='zeros')
        
    def compute_quantile(self, distances):
            n = tf.shape(distances)[0]
            sorted_distances = tf.sort(distances)
            q_index = tf.cast(tf.round(self.quantile_level * tf.cast(n - 1, tf.float32)), tf.int32)
            return tf.gather(sorted_distances, q_index)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

        # Handle one-hot or sparse labels
        if y_true.shape.rank == 2:
            y_true = tf.argmax(y_true, axis=-1)
        
        y_true = tf.cast(y_true, tf.int32)  # Make sure types match

        # Get predicted probabilities for the true class
        indices = tf.stack([tf.range(tf.shape(y_true)[0], dtype=tf.int32), y_true], axis=1)
        true_class_probs = tf.gather_nd(y_pred, indices)

        # Clip and compute distances
        true_class_probs = tf.clip_by_value(true_class_probs, 0.0, 1.0)
        distances = tf.abs(true_class_probs - 0.5)


        quantile_value = self.compute_quantile (distances)
        self.last_value.assign(tf.reshape(quantile_value, []))

    def result(self):
        return self.last_value

    def reset_states(self):
        self.last_value.assign(0.0)