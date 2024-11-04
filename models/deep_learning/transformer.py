import tensorflow as tf
import numpy as np
from models.deep_learning.deep_learning_models import DLRegressor
from utils.constants import is_spline

class TransformerRegressor(DLRegressor):
    """
    This is a class implementing the Transformer architecture for time series regression.
    """

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=200,
            batch_size=16,
            loss="mean_squared_error",
            metrics=None,
    ):
        """
        Initialise the Transformer model

        Inputs:
            output_directory: path to store results/models
            input_shape: input shape for the models
            verbose: verbosity for the models
            epochs: number of epochs to train the models
            batch_size: batch size to train the models
            loss: loss function for the models
            metrics: metrics for the models
        """

        self.name = "Transformer"
        self.head_size=64
        self.num_heads=4
        self.ff_dim=128
        self.num_transformer_blocks=6
        self.mlp_units=[128, 64]
        self.dropout=0.1
        self.mlp_dropout=0.1
        
        super().__init__(
            output_directory=output_directory,
            input_shape=input_shape,
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics
        )

    def positional_encoding(self, length, d_model):
        """
        Create positional encoding matrix matching input dimensions
        length: sequence length (264 in our case)
        d_model: number of features (2 in our case)
        """
        positions = np.arange(length)[:, np.newaxis]    # (sequence_length, 1)
        depths = np.arange(d_model)[np.newaxis, :]/np.float32(d_model)  # (1, d_model)
        
        angle_rates = 1 / (10000**depths)         # (1, d_model)
        angle_rads = positions * angle_rates       # (sequence_length, d_model)
        
        # Match the input shape (sequence_length, d_model)
        pos_encoding = angle_rads
        
        return tf.cast(pos_encoding, dtype=tf.float32)

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        """
        Transformer encoder block with fixed key_dim calculation
        """
        # Multi-Head Self Attention
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        
        # Calculate key_dim properly
        key_dim = max(1, head_size // num_heads)  # Ensure key_dim is at least 1
        
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=dropout
        )(x, x)
        
        attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
        x = tf.keras.layers.Add()([inputs, attention_output])

        # Feed Forward
        ff_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ff_output = tf.keras.layers.Dense(ff_dim, activation="gelu")(ff_output)
        ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
        ff_output = tf.keras.layers.Dense(inputs.shape[-1])(ff_output)
        ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
        
        return tf.keras.layers.Add()([x, ff_output])


    def build_model(self, input_shape):
        """
        Build the Transformer model

        Inputs:
            input_shape: input shape for the model
        """
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Create positional encoding with matching dimensions
        positions = self.positional_encoding(input_shape[0], input_shape[1])
        # Add a batch dimension of 1 to positional encoding
        positions = tf.expand_dims(positions, axis=0)  # (1, seq_len, d_model)
    
        # Add positional encoding to input
        x = tf.keras.layers.Add()([inputs, positions])
        
        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = self.transformer_encoder(x, self.head_size, self.num_heads, self.ff_dim, self.dropout)

        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # MLP layers
        for dim in self.mlp_units:
            x = tf.keras.layers.Dense(dim, activation="gelu")(x)
            x = tf.keras.layers.Dropout(self.mlp_dropout)(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(1)(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='transformer')

        return model
