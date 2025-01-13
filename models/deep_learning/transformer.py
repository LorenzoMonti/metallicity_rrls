"""
Explanation of Key Components

    Embedding Layer: Maps each (phase,magnitude)(phase,magnitude) pair to a higher-dimensional space.
    Positional Encoding: Provides positional information to the model.
    Multi-Head Attention: Captures relationships across the sequence.
    Feed-Forward Network: Enhances the capacity of the transformer layers.
    Pooling Layer: Summarizes the entire sequence output into a fixed-length vector, making it suitable for regression.
    Output Layer: A dense layer that provides a single metallicity prediction.

Training and Hyperparameters

    Batch Size and Epochs: You’ll need to experiment with these based on the computational power available.
    Hyperparameter Tuning: Adjust embedding_dim, num_heads, ff_dim, and num_transformer_blocks for optimal results.
    Regularization: If you encounter overfitting, consider adding dropout layers or reducing the model size.

This transformer model should be able to learn patterns in the photometric light curves and potentially generalize to predict the metallicity.

The encoder layer alone can work well for this type of time-series regression task because the focus is on extracting meaningful representations from a fixed input sequence to predict a single output value (metallicity). Here’s why the encoder alone is typically sufficient:
1. Nature of the Problem: Feature Extraction, Not Sequence Generation

    In tasks like photometric light curve analysis, we don’t need to generate new sequences (as we would in, say, language translation or text generation), but rather to extract patterns in the input sequence that correlate with the target (metallicity). The encoder layer is designed precisely for this purpose: it can learn and highlight important relationships within the input sequence and compress this information into a summary representation.

2. Self-Attention Mechanism in Encoders

    The encoder’s self-attention mechanism allows each time step in the sequence to attend to all other time steps. This is particularly useful for your light curves because the model can focus on important interactions between phase and magnitude values at different points in the sequence, which could be linked to metallicity. This is different from a decoder’s cross-attention, which is needed when decoding output tokens in a sequence-to-sequence task.

3. Simplicity and Efficiency

    By using only encoder layers, we reduce the model's complexity, which is computationally more efficient and typically leads to faster training. This is especially helpful given the long sequence length (264 time steps). The model is still fully capable of learning complex patterns with the encoder alone.

4. Pooling Layer for Sequence Compression

    By using global pooling after the encoder layers, we compress the output into a single vector that summarizes all relevant information from the entire input sequence. This makes the architecture ideal for regression tasks, where we need a single, scalar output instead of a full sequence.

    Best Hyperparameters:
    num_transformer_blocks: 1
    head_size: 96
    num_heads: 2
    ff_dim: 128
    embedding_dim: 64
    learning_rate: 0.00023273998841816132
    dropout_rate: 0.1
    tuner/epochs: 500
    tuner/initial_epoch: 167
    tuner/bracket: 3
    tuner/round: 3
    tuner/trial_id: 0659    
"""

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
        self.head_size=96
        self.num_heads=2
        self.ff_dim=128
        self.num_transformer_blocks=1
        self.mlp_units=[128, 64]
        self.dropout=0.1
        self.mlp_dropout=0.1
        
        # Input dimensions
        self.num_timesteps = 265
        self.num_features = 2
        self.embedding_dim = 64  # Dimension to which we embed phase-magnitude pairs

        # Transformer parameters
        self.num_heads = 4
        self.ff_dim = 64  # Feed-forward dimension within transformer
        self.num_transformer_blocks = 2

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
        # Parameters
        num_timesteps = 265
        num_features = 2
        embedding_dim = 64  # Dimension to embed input features
        num_heads = 2
        ff_dim = 128  # Feed-forward network size
        num_transformer_blocks = 1
        distill_factor = 2  # Factor for sequence length reduction
        dropout_rate = 0.1

        # Define Input
        inputs = tf.keras.layers.Input(shape=(num_timesteps, num_features))

        # Embedding
        x = tf.keras.layers.Dense(embedding_dim)(inputs)

        # Positional Encoding
        positions = tf.range(start=0, limit=num_timesteps, delta=1)
        pos_encoding = tf.keras.layers.Embedding(input_dim=num_timesteps, output_dim=embedding_dim)(positions)
        x += pos_encoding

        # Distillation: Reduce sequence length by convolution
        x = tf.keras.layers.Conv1D(filters=embedding_dim, kernel_size=3, strides=distill_factor, padding="same")(x)

        # Transformer Blocks with Sparse Attention
        for _ in range(num_transformer_blocks):

            # ProbSparse Self-Attention (approximation of attention sparsity)
            attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)(x, x)
            attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attention_output)  # Residual connection

            # Feed Forward Network
            ffn_output = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
            ffn_output = tf.keras.layers.Dense(embedding_dim)(ffn_output)
            ffn_output = tf.keras.layers.Dropout(dropout_rate)(ffn_output)
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)  # Residual connection

        # Pooling Layer
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # Output Layer
        outputs = tf.keras.layers.Dense(1)(x)
        
        # Model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='transformer')
        
        return model
