import os
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from utils.constants import *
from utils.experiments_tools import *
from utils.regressor_tools import *
from utils.plot_tools import *

class TransformerTuner:
    def __init__(
        self,
        output_directory,
        input_shape,
        y_train,
        max_trials=50,
        executions_per_trial=3
    ):
        """
        Initialize Transformer Hyperparameter Tuner

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
        # Tunable hyperparameters for Transformer architecture
        num_transformer_blocks = hp.Int(
            'num_transformer_blocks',
            min_value=1,
            max_value=6,
            step=1
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

        embedding_dim = hp.Int(
            'embedding_dim',
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

        # Parameters from original Transformer architecture
        num_timesteps = self.input_shape[0]
        num_features = self.input_shape[1]

        # Define Input
        inputs = tf.keras.layers.Input(shape=(num_timesteps, num_features))

        # Embedding
        x = tf.keras.layers.Dense(embedding_dim)(inputs)

        # Positional Encoding
        positions = tf.range(start=0, limit=num_timesteps, delta=1)
        pos_encoding = tf.keras.layers.Embedding(input_dim=num_timesteps, output_dim=embedding_dim)(positions)
        x += pos_encoding

        # Distillation: Reduce sequence length by convolution
        distill_factor = 2
        x = tf.keras.layers.Conv1D(
            filters=embedding_dim,
            kernel_size=3,
            strides=distill_factor,
            padding="same"
        )(x)

        # Transformer Blocks with Sparse Attention
        for _ in range(num_transformer_blocks):
            # ProbSparse Self-Attention
            attention_output = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=embedding_dim,
                dropout=dropout_rate
            )(x, x)
            attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)

            # Feed Forward Network
            ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
            ffn_output = tf.keras.layers.Dense(embedding_dim)(ffn_output)
            ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

        # Pooling Layer
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Output Layer
        outputs = tf.keras.layers.Dense(1)(x)

        # Create and compile model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='transformer_tuned')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mean_squared_error',
            metrics=['mae']
        )

        return model

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
            project_name='transformer_tuning',
            executions_per_trial=self.executions_per_trial
        )

        # Early stopping
        stop_early = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=100,
            restore_best_weights=True
        )

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
        best_model.save(os.path.join(self.output_directory, 'best_transformer_model.h5'))

        # Print best hyperparameters
        print("\nBest Hyperparameters:")
        for param, value in best_hps.values.items():
            print(f"{param}: {value}")

        return best_hps, history

    def plot_tuning_results(self, history, output_path):
        """
        Plot tuning results

        Args:
            history: Model training history
            output_path: Path to save the plot
        """
        plt.figure(figsize=(12, 5))

        # Training and Validation Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Training and Validation MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

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

    # Output directory for results
    output_directory = './tuning/transformer_tuning_results'

    transformer_tuner = TransformerTuner(
    output_directory=output_directory,
    input_shape=X_train.shape[1:],
    y_train=y_train,
    max_trials=50,
    executions_per_trial=3
    )

    # Perform hyperparameter tuning
    best_hps, tuning_history = transformer_tuner.tune_hyperparameters(X_train, X_val, y_train, y_val)

    # Optional: Plot tuning results
    transformer_tuner.plot_tuning_results(tuning_history,
        os.path.join(transformer_tuner.output_directory, 'tuning_results.png'))