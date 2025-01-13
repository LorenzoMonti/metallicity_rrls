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
- num_transformer_blocks: 8
- head_size: 96
- num_heads: 2
- ff_dim: 256
- learning_rate: 0.002964803176107267
- dropout_rate: 0.1
- sparsity_rate: 0.2
- tuner/epochs: 19
- tuner/initial_epoch: 7
- tuner/bracket: 5
- tuner/round: 2
- tuner/trial_id: 0284
"""

import tensorflow as tf
import numpy as np
from models.deep_learning.deep_learning_models import DLRegressor

import tensorflow as tf
import numpy as np

class ProbSparseAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, key_dim, dropout_rate=0.1, sparsity_rate=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.sparsity_rate = sparsity_rate
        
        self.multi_head_attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim
        )
    
    def call(self, query, key, value, training=None):
        # Standard multi-head attention
        attention_output = self.multi_head_attention(query, key, value)
        
        # Probabilistic sparsity mask during training
        if training:
            mask_shape = tf.shape(attention_output)
            mask = tf.random.uniform(mask_shape) > self.sparsity_rate
            attention_output = attention_output * tf.cast(mask, tf.float32)
        
        return attention_output

class InformerRegressor(DLRegressor):
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
        Initialize the Informer Transformer model for regression
        """
        self.name = "Informer"

        # Hyperparameters
        self.head_size = 96
        self.num_heads = 2
        self.ff_dim = 256
        self.num_transformer_blocks = 8
        self.mlp_units = [128, 64]
        self.dropout = 0.1
        self.mlp_dropout = 0.1
        self.sparsity_rate = 0.2
        
        # Model configuration
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss

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
        """
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(d_model)[np.newaxis, :] / np.float32(d_model)
        
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        
        return tf.cast(angle_rads, dtype=tf.float32)
    
    def informer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0, sparsity_rate=0.2):
        """
        Informer encoder block with probabilistic sparse attention
        """
        # Layer Normalization
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
        
        # Calculate key dimension
        key_dim = max(1, head_size // num_heads)
        
        # Probabilistic Sparse Self Attention
        attention_layer = ProbSparseAttention(
            num_heads=num_heads, 
            key_dim=key_dim, 
            dropout_rate=dropout,
            sparsity_rate=sparsity_rate
        )
        attention_output = attention_layer(x, x, x)
        
        # Dropout and Residual Connection
        attention_output = tf.keras.layers.Dropout(dropout)(attention_output)
        x = tf.keras.layers.Add()([inputs, attention_output])
        
        # Distilling Attention (aggressive sparsity)
        distill_attention = ProbSparseAttention(
            num_heads=num_heads, 
            key_dim=key_dim, 
            dropout_rate=dropout,
            sparsity_rate=0.5  # More aggressive sparsity
        )
        x = distill_attention(x, x, x)
        
        # Feed Forward Network
        ff_output = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        ff_output = tf.keras.layers.Dense(ff_dim, activation="gelu")(ff_output)
        ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
        ff_output = tf.keras.layers.Dense(inputs.shape[-1])(ff_output)
        ff_output = tf.keras.layers.Dropout(dropout)(ff_output)
        
        return tf.keras.layers.Add()([x, ff_output])
    
    def build_model(self, input_shape):
        """
        Build the Informer model
        """
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Positional Encoding
        positions = self.positional_encoding(input_shape[0], input_shape[1])
        positions = tf.expand_dims(positions, axis=0)
        
        # Add positional encoding to input
        x = tf.keras.layers.Add()([inputs, positions])
        
        # Informer Transformer Blocks
        for _ in range(self.num_transformer_blocks):
            x = self.informer_encoder(
                x, 
                self.head_size, 
                self.num_heads, 
                self.ff_dim, 
                self.dropout,
                self.sparsity_rate
            )
        
        # Sequence Length Reduction (Distilling)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # MLP Layers
        for dim in self.mlp_units:
            x = tf.keras.layers.Dense(dim, activation="gelu")(x)
            x = tf.keras.layers.Dropout(self.mlp_dropout)(x)
        
        # Output Layer
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs, name='informer')