"""TensorFlow-only ANN model following the regression class structure."""
from __future__ import annotations

from typing import Iterable, Optional, List, Union

from data.DataProcessor import DataProcessor
from data.DataTransformer import DataTransformer
from utils.columns_enum import OutputColumn
import tensorflow as tf


class ANNRegressionModel:
    x_train = x_test = y_train = y_test = x_val = y_val = None
    ann = multiple= None

    def __init__(self, data, multiple_outputs: bool = False, col: Union[OutputColumn, str] = OutputColumn.TENSAO):
        """Initialize the ANN regressor.

        Args:
            data: Raw data source (file path or DataFrame) accepted by DataTransformer.
            multiple_outputs: If True, model predicts multiple targets (uses DataProcessor.multi-output path).
            col: Target column. Can be an OutputColumn enum member or a raw string.
        """
        # Resolve enum to its string value if needed
        target_col = col.value if isinstance(col, OutputColumn) else col
        self.col = target_col
        data_processor = DataProcessor(data, DataTransformer.get_transformer('excel'), target_col)
        self.multiple = multiple_outputs
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = data_processor.split_data(multiple_outputs=multiple_outputs)


    @staticmethod
    def _build_tf_ann(input_dim: int, hidden_layers: Iterable[int], activation: str, dropout: float, output_dim: int = 1):
       
        layers = tf.keras.layers
        models = tf.keras.models

        inputs = layers.Input(shape=(input_dim,))
        x = inputs
        for units in hidden_layers:
            x = layers.Dense(units, activation=activation)(x)
            if dropout and dropout > 0:
                x = layers.Dropout(dropout)(x)
        # Regression head: units = output_dim (1 for single target, >1 for multi-output)
        outputs = layers.Dense(output_dim, activation=None)(x)
        model = models.Model(inputs=inputs, outputs=outputs, name='ann_regressor')
        return model

    def train(self):
        input_dim = self.x_train.shape[-1] # quantity of columns(features) used to train
        # Determine output dimension from y_train
        print(len(getattr(self.y_train, 'shape', [])))
        if self.multiple and len(getattr(self.y_train, 'shape', [])) > 1:
            output_dim = int(self.y_train.shape[1])
        else:
            output_dim = 1
        model = self._build_tf_ann(input_dim, hidden_layers=(32, 32, 32, 32), activation='relu', dropout=0.2, output_dim=output_dim)
        # if have much outliers try using 'huber' loss, adam is adaptative moment estimation
        # i need to review this part and understand better optimizers
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
        model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), epochs=50, batch_size=32, verbose=0)
        self.ann = model
        return self.ann

    def predict(self, x_input: Optional[Union[List, tf.Tensor]] = None):
        x = x_input if x_input is not None else self.x_test
        print(self.x_test)
        print(x_input)
        preds = self.ann.predict(x, verbose=0)
        # For multi-output, keep the 2D shape; for single-output, return 1D if possible
        if self.multiple and getattr(preds, 'ndim', 1) == 2 and preds.shape[1] > 1:
            return preds
        return preds.ravel() if hasattr(preds, 'ravel') else preds

    def optimize(self, **kwargs):
        # Use Keras Tuner via utils.optimize, providing an hp-based builder that calls our build_hp_model
        from utils.optimize import (
            keras_tuner_optimize_ann,
        )

        def build_fn_hp(hp):
            return self.build_hp_model(hp)

        xtr, ytr, xva, yva = self.x_train, self.y_train, self.x_val, self.y_val

        params, best_model = keras_tuner_optimize_ann(build_fn_hp, xtr, ytr, xva, yva, **kwargs)

        self.ann = best_model
        return params, best_model

    # --- Keras Tuner compatible builder (hyperparameters) ---
    def build_hp_model(self, hp):
        """Build a compiled Keras model using hyperparameters from Keras Tuner.

        This follows the style shown in the reference image, adapted for
        regression: we select the number of Dense layers, per-layer units,
        activation, dropout, and learning rate. Input/output dims are inferred
        from the current train/label tensors.

        Usage with Keras Tuner:
            import keras_tuner as kt
            tuner = kt.BayesianOptimization(
                lambda hp: ann.build_hp_model(hp),
                objective="val_loss",
                max_trials=20,
                directory="kt_logs",
                project_name="ann_hp",
            )
            tuner.search(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val), epochs=50)
        """
        layers = tf.keras.layers

        input_dim = int(self.x_train.shape[-1])
        if self.multiple and len(getattr(self.y_train, 'shape', [])) > 1 and self.y_train.shape[1] > 1:
            output_dim = int(self.y_train.shape[1])
        else:
            output_dim = 1

        model = tf.keras.Sequential(name="ann_regressor_hp")
        model.add(layers.Input(shape=(input_dim,)))

        # Hyperparameters
        num_layers = hp.Int("num_layers", min_value=1, max_value=5, step=1)
        activation = hp.Choice("activation", values=["relu", "tanh", "elu"])
        dropout_rate = hp.Choice("dropout", values=[0.0, 0.1, 0.2, 0.3])
        learning_rate = hp.Choice("learning_rate", values=[1e-2, 5e-3, 1e-3, 5e-4])

        for i in range(num_layers):
            units = hp.Int(f"units_{i}", min_value=32, max_value=512, step=32)
            model.add(layers.Dense(units, activation=activation))
            if dropout_rate and dropout_rate > 0:
                model.add(layers.Dropout(dropout_rate))

        model.add(layers.Dense(output_dim, activation=None))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )
        return model

    def save(self, filename: Optional[str] = None) -> str:
        """Save the trained ANN under outputs/neural_network.

        If filename ends with '.keras' or '.h5', a single file is written.
        Otherwise, a SavedModel directory is created at that path.
        Returns the full path used.
        """
        import os
        if self.ann is None:
            raise RuntimeError("No trained ANN found to save. Call train() first.")
        base_dir = os.path.join('outputs', 'neural_network')
        os.makedirs(base_dir, exist_ok=True)
        # Default to Keras v3 format file
        name = filename or f"{self.__class__.__name__}-{self.col}.keras"
        path = os.path.join(base_dir, name)
        self.ann.save(path)
        return path

    @staticmethod
    def load(path: str):
        """Load a saved ANN (Keras model file or SavedModel directory)."""
        return tf.keras.models.load_model(path)

