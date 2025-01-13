import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from utils.constants import *
from utils.experiments_tools import *
from utils.regressor_tools import *
from utils.plot_tools import *

class InformerTuner:
    def __init__(
        self,
        output_directory,
        input_shape,
        y_train,
        max_trials=50,
        executions_per_trial=3
    ):
        """
        Initialize Informer Hyperparameter Tuner

        Args:
            output_directory: Directory to save tuning results
            input_shape: Shape of input data
            y_train: Target variable for training
            max_trials: Maximum number of trials for tuning
            executions_per_trial: Number of model executions per trial
        """
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.y_train = y_train
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial

        # Ensure output directory exists
        os.makedirs(output_directory, exist_ok=True)

    def build_model(self, hp):
        """
        Build model with hyperparameters to be tuned

        Args:
            hp: Hyperparameters to be tuned

        Returns:
            Compiled Keras model
        """
        # Tunable hyperparameters
        num_transformer_blocks = hp.Int(
            'num_transformer_blocks',
            min_value=2,
            max_value=8,
            step=2
        )

        head_size = hp.Int(
            'head_size',
            min_value=32,
            max_value=128,
            step=32
        )

        num_heads = hp.Int(
            'num_heads',
            min_value=2,
            max_value=8,
            step=2
        )

        ff_dim = hp.Int(
            'ff_dim',
            min_value=64,
            max_value=256,
            step=64
        )

        learning_rate = hp.Float(
            'learning_rate',
            min_value=1e-4,
            max_value=1e-2,
            sampling='LOG'
        )

        dropout_rate = hp.Float(
            'dropout_rate',
            min_value=0.1,
            max_value=0.5,
            step=0.1
        )

        sparsity_rate = hp.Float(
            'sparsity_rate',
            min_value=0.1,
            max_value=0.5,
            step=0.1
        )

        # Input layer
        inputs = tf.keras.layers.Input(shape=self.input_shape)

        # Positional Encoding
        positions = self._positional_encoding(
            self.input_shape[0],
            self.input_shape[1]
        )
        positions = tf.expand_dims(positions, axis=0)

        # Add positional encoding to input
        x = tf.keras.layers.Add()([inputs, positions])

        # Informer Transformer Blocks
        for _ in range(num_transformer_blocks):
            x = self._informer_encoder(
                x,
                head_size,
                num_heads,
                ff_dim,
                dropout_rate,
                sparsity_rate
            )

        # Sequence Length Reduction
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # MLP Layers
        x = tf.keras.layers.Dense(128, activation="gelu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(64, activation="gelu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

        # Output Layer
        outputs = tf.keras.layers.Dense(1)(x)

        # Create and compile model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='informer_tuned')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

    def _positional_encoding(self, length, d_model):
        """
        Create positional encoding matrix
        """
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(d_model)[np.newaxis, :] / np.float32(d_model)

        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates

        return tf.cast(angle_rads, dtype=tf.float32)

    def _informer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout, sparsity_rate):
        """
        Informer encoder block with probabilistic sparse attention
        """
        # Layer Normalization
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)

        # Calculate key dimension
        key_dim = max(1, head_size // num_heads)

        # Probabilistic Sparse Self Attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )(x, x)

        # Dropout and Residual Connection
        attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
        x = tf.keras.layers.Add()([inputs, attention_output])

        # Feed Forward Network
        ff_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ff_output = tf.keras.layers.Dense(ff_dim, activation="gelu")(ff_output)
        ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
        ff_output = tf.keras.layers.Dense(inputs.shape[-1])(ff_output)
        ff_output = tf.keras.layers.Dropout(dropout)(ff_output)

        return tf.keras.layers.Add()([x, ff_output])

    def tune_hyperparameters(self, X_train, X_val, y_train, y_val):
        """
        Perform hyperparameter tuning

        Args:
            X_train: Training input data
            X_val: Validation input data
            y_train: Training target data
            y_val: Validation target data

        Returns:
            Best hyperparameters and tuning results
        """
        # Create tuner
        tuner = kt.Hyperband(
            self.build_model,
            objective='val_mae',
            max_epochs=500,
            factor=3,
            directory=self.output_directory,
            project_name='informer_tuning',
            executions_per_trial=self.executions_per_trial
        )

        # Early stopping
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

        # Run hyperparameter search
        tuner.search(
            X_train, y_train,
            epochs=500,
            validation_data=(X_val, y_val),
            callbacks=[stop_early]
        )

        # Get best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Retrain best model
        best_model = tuner.hypermodel.build(best_hps)
        history = best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=500,
            callbacks=[stop_early]
        )

        # Save best model
        best_model.save(os.path.join(self.output_directory, 'best_informer_model.h5'))

        # Print best hyperparameters
        print("\nBest Hyperparameters:")
        for param, value in best_hps.values.items():
            print(f"{param}: {value}")

        return best_hps, history

if __name__ == "__main__":
    spline_points = 264  # Example number of spline points

    # Read dataset
    input_dataset = read_csv_dataset(rrl_path, rrls_number)
    ids_dev = input_dataset[source_id].to_numpy().astype(str)
    periods_input = input_dataset[period].to_numpy()

    # Read time series data
    X, _, _ = read_time_series(ids_dev, data_path, spline_points, periods=periods_input, max_phase=1.0)
    y = input_dataset[metallicity].to_numpy()

    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape, )
    # Output directory for results
    output_directory = './informer_tuning_results'

    # Create Informer Tuner
    informer_tuner = InformerTuner(
        output_directory=output_directory,
        input_shape=X_train.shape[1:],
        y_train=y_train,
        max_trials=50,
        executions_per_trial=3
    )

    # Perform hyperparameter tuning
    best_hps, tuning_history = informer_tuner.tune_hyperparameters(
        X_train, X_val, y_train, y_val
    )

    # Optional: Visualize tuning results
    plt.figure(figsize=(10, 5))
    plt.plot(tuning_history.history['val_mae'], label='Validation MAE')
    plt.title('Model Validation MAE During Tuning')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.savefig(os.path.join(output_directory, 'tuning_results.png'))
    plt.close()