import tensorflow as tf
import tensorflow as tf


class OverclassAwareLoss(tf.keras.losses.Loss):
    """
    Combines standard class-level binary crossentropy with an overclass-level binary crossentropy loss.

    Args:
        overclass_dict (dict): Mapping from class index to overclass index.
        lambda_penalty (float): Trade-off between class-level and overclass-level loss.
    """

    def __init__(self, overclass_dict, lambda_penalty=0.0,from_logits=False, name="overclass_aware_loss"):
        super().__init__(name=name)
        self.overclass_dict = overclass_dict
        self._validate_overclass_dict()

        self.num_classes = len(overclass_dict)
        self.n_overclasses = max(overclass_dict.values()) + 1
        self.class_to_overclass_matrix = self._build_projection_matrix()
        self.lambda_penalty = lambda_penalty
        self.from_logits = from_logits
        
    def _validate_overclass_dict(self):
        if not all(isinstance(k, int) and isinstance(v, int) for k, v in self.overclass_dict.items()):
            raise ValueError("overclass_dict must map class indices (int) to overclass indices (int).")

    def _build_projection_matrix(self):
        """
        Builds a binary projection matrix that maps each class to its corresponding overclass.

        Returns:
            tf.Tensor: A 2D tensor of shape (num_classes, n_overclasses) where each row corresponds
                    to a class and each column to an overclass. The entry at (i, j) is 1.0 if class i
                    belongs to overclass j, and 0.0 otherwise.

        How it works:
            - Initializes a zero matrix of shape (num_classes, n_overclasses).
            - Uses self.overclass_dict, which maps class indices to overclass indices, to determine
            which positions in the matrix should be set to 1.0.
            - The positions are updated using tf.tensor_scatter_nd_update.

        Notes:
            You must pass an array because tf.tensor_scatter_nd_update() does not support automatic broadcasting of a scalar across multiple indices.
        
        Example:
            If self.overclass_dict = {0: 2, 1: 0, 2: 1} and there are 3 classes and 3 overclasses,
            the resulting matrix will be:

                [[0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]]
        """
        indices = [[cls, overcls] for cls, overcls in self.overclass_dict.items()]
        updates = tf.ones(len(indices), dtype=tf.float32)
        shape = (self.num_classes, self.n_overclasses)
        return tf.tensor_scatter_nd_update(tf.zeros(shape, dtype=tf.float32), indices, updates)
    
    
    def _compute_overclass_loss(self, target, output):
        target_overclass = tf.matmul(target, self.class_to_overclass_matrix)
        output_overclass = tf.matmul(output, self.class_to_overclass_matrix)
        # numerically stabilize the output
        output_overclass = tf.clip_by_value(output_overclass, tf.keras.backend.epsilon(), 1. - tf.keras.backend.epsilon())
        if self.from_logits:
            output = tf.nn.sigmoid(output)
        
        loss = tf.keras.losses.binary_crossentropy(target_overclass, output_overclass)
        return tf.reduce_mean(loss)
    
    def call(self, target, output):
        """
        Computes the combined loss.
        The combined loss is a weighted sum of the standard binary crossentropy loss
        and the overclass binary crossentropy loss.
        The weights are controlled by the lambda_penalty parameter.

        Args:
            target (_tensor): True labels (one-hot encoded).
            output (_tensor): Predicted probabilities or logits (same shape as target).

        Returns:
            _tensor: Computed loss value.
        """
        target = tf.cast(target, tf.float32)
        output = tf.cast(output, tf.float32)
        
        if self.from_logits:
            output = tf.nn.sigmoid(output)

        class_loss = tf.keras.losses.binary_crossentropy(target, output)
        overclass_loss = self._compute_overclass_loss(target, output)

        return self.lambda_penalty * overclass_loss + (1 - self.lambda_penalty) * class_loss