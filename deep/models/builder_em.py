from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, MaxPooling2D, Conv2D, LeakyReLU
from typing import List, Dict, Any, Tuple, Optional

LAYER_MAPPING = {
    "Conv2D": Conv2D,
    "MaxPooling2D": MaxPooling2D,
    "Flatten": Flatten,
    "Dense": Dense,
    "LeakyReLU": LeakyReLU,
    "Concatenate": Concatenate
}

class SimpleKeras(Model):
    def __init__(self, instructions: Dict[str, Any], base_model: Optional[Model] = None, **kwargs):
        super().__init__(**kwargs)
        self.instructions = instructions
        self.base_model = base_model
        self.trainable = kwargs.get('trainable', False)
        self.layer_dict = {}  # To store layers for Concatenate

    @classmethod
    def build_model(cls, instructions: Dict[str, Any], base_model: Optional[Model] = None) -> Model:
        input_args = instructions["input"]
        layers_instructions = instructions["layers"]
        n_classes = instructions["n_classes"]

        input_layer = Input(**input_args)
        x = base_model(input_layer) if base_model else input_layer

        instance = cls(instructions, base_model=base_model)
        parsed_layers = instance._parse_instructions(layers_instructions)

        for i, layer in enumerate(parsed_layers):
            x = layer(x)
            instance.layer_dict[f"layer_{i}"] = x  # Store outputs for Concatenate

        output_layer = Dense(n_classes, activation="softmax", name="classification_head")(x)
        return Model(inputs=input_layer, outputs=output_layer, name="fine_tuned_cnn")

    def _parse_instructions(self, layers_instructions: List[Tuple[str, Dict[str, Any]]]) -> List[layers.Layer]:
        parsed_layers = []
        for layer_type, params in layers_instructions:
            if layer_type == "Concatenate":
                inputs = [self.layer_dict[input_name] for input_name in params["inputs"]]
                layer_instance = LAYER_MAPPING[layer_type](inputs)
            else:
                layer_instance = LAYER_MAPPING[layer_type](**params)
            parsed_layers.append(layer_instance)
        return parsed_layers

    @staticmethod
    def objective(trial, train_ds, val_ds, base_model, n_classes: int):
        num_layers = trial.suggest_int("num_layers", 1, 5)
        layers_instructions = []
        for i in range(num_layers):
            layer_type = trial.suggest_categorical(f"layer_type_{i}", ["Dense", "Conv2D"])
            if layer_type == "Dense":
                units = trial.suggest_int(f"units_{i}", 64, 512, step=64)
                activation = trial.suggest_categorical(f"activation_{i}", ["relu", "leaky_relu"])
                if activation == "leaky_relu":
                    layers_instructions.append(("LeakyReLU", {}))
                else:
                    layers_instructions.append(("Dense", {"units": units, "activation": activation}))
            elif layer_type == "Conv2D":
                filters = trial.suggest_int(f"filters_{i}", 16, 128, step=16)
                kernel_size = trial.suggest_categorical(f"kernel_size_{i}", [3, 5])
                layers_instructions.append(("Conv2D", {"filters": filters, "kernel_size": kernel_size, "activation": "relu"}))

        if any(layer[0] == "Conv2D" for layer in layers_instructions) and any(layer[0] == "Dense" for layer in layers_instructions):
            layers_instructions.insert(next(i for i, layer in enumerate(layers_instructions) if layer[0] == "Dense"), ("Flatten", {}))

        instructions = {
            "input": {"shape": (224, 224, 3)},
            "layers": layers_instructions,
            "n_classes": n_classes
        }

        model = SimpleKeras.build_model(instructions, base_model=base_model)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        history = model.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)
        return max(history.history["val_accuracy"])
    
import tensorflow as tf
from tensorflow.keras import models, layers
from keras_tuner import HyperModel

class MyTransferHyperModel_1(HyperModel):
    def __init__(self, base_model_callable, input_shape, num_classes):
        """
        Initialize the HyperModel with a flexible base model, input shape, and number of classes.
        
        :param base_model_callable: Callable that returns a base model (e.g., EfficientNetB0, VGG16)
        :param input_shape: Tuple specifying the input shape (e.g., (224, 224, 3))
        :param num_classes: Number of output classes
        """
        self.base_model_callable = base_model_callable
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        # Instantiate the base model with the provided callable
        base_model = self.base_model_callable(include_top=False, input_shape=self.input_shape, weights='imagenet')
        base_model.trainable = False  # Freeze the base model

        # Build the sequential model
        model = models.Sequential()
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())

        # Tune number of dense layers
        num_dense = hp.Int('num_dense', min_value=1, max_value=3)
        for i in range(num_dense):
            units = hp.Int(f'units_{i}', min_value=64, max_value=512, step=64)
            model.add(layers.Dense(units=units, activation='relu'))
            dropout = hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)
            model.add(layers.Dropout(dropout))

        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Tune learning rate
        hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model
    

import tensorflow as tf
from tensorflow.keras import models, layers
from keras_tuner import HyperModel

class MyTransferHyperModel_2(HyperModel):
    def __init__(self, base_model_callable, input_shape, num_classes):
        """
        Initialize the HyperModel with a flexible base model, input shape, and number of classes.
        
        :param base_model_callable: Callable that returns a base model (e.g., EfficientNetB0, VGG16)
        :param input_shape: Tuple specifying the input shape (e.g., (224, 224, 3))
        :param num_classes: Number of output classes
        """
        self.base_model_callable = base_model_callable
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        # Instantiate the base model with the provided callable
        base_model = self.base_model_callable(include_top=False, input_shape=self.input_shape, weights='imagenet')
        base_model.trainable = False  # Freeze the base model

        # Build the sequential model
        model = models.Sequential()
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())  # Reduces spatial dimensions

        # Dense layer units
        hp_units = hp.Int('units', min_value=64, max_value=512, step=64)
        model.add(layers.Dense(units=hp_units, activation='relu'))

        # Dropout rate
        hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
        model.add(layers.Dropout(hp_dropout))

        # Output layer
        model.add(layers.Dense(self.num_classes, activation='softmax'))

        # Tune learning rate
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model