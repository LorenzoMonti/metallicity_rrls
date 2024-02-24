import tensorflow as tf

from models.deep_learning.deep_learning_models import DLRegressor
from utils.constants import is_spline

class ConvLSTMRegressor(DLRegressor):
    """
    This is a class implementing the ConvLSTM model for time series regression.
    """

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=2000,
            batch_size=256,
            loss="mean_squared_error",
            metrics=None
    ):
        """
        Initialise the ConvLSTM model

        Inputs:
            output_directory: path to store results/models
            input_shape: input shape for the models
            verbose: verbosity for the models
            epochs: number of epochs to train the models
            batch_size: batch size to train the models
            loss: loss function for the models
            metrics: metrics for the models
        """

        self.name = "ConvLSTM"
        super().__init__(
            output_directory=output_directory,
            input_shape=input_shape,
            verbose=verbose,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
            metrics=metrics
        )

    def build_model(self, input_shape):
        """
        Build the ConvLSTM model

        Inputs:
            input_shape: input shape for the model
        """
        input_layer = tf.keras.layers.Input(input_shape)

        conv1 = tf.keras.layers.Conv1D(filters=128,
                                       kernel_size=8,
                                       padding='same')(input_layer)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Activation(activation='relu')(conv1)

        conv2 = tf.keras.layers.Conv1D(filters=256,
                                       kernel_size=5,
                                       padding='same')(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Activation('relu')(conv2)

        conv3 = tf.keras.layers.Conv1D(128,
                                       kernel_size=3,
                                       padding='same')(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.Activation('relu')(conv3)

        gap_layer = tf.keras.layers.MaxPooling1D()(conv3)

        lstm_layer1 = tf.keras.layers.LSTM(8, return_sequences=True, 
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=2e-6),  
                                           recurrent_regularizer=tf.keras.regularizers.L1L2(l1=2e-6, l2=0),
                                           activation='tanh')(gap_layer)

        lstm_layer1 = tf.keras.layers.Dropout(rate=0.2)(lstm_layer1)
        
        lstm_layer2 = tf.keras.layers.LSTM(4, return_sequences=False, 
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=2e-6),  
                                           recurrent_regularizer=tf.keras.regularizers.L1L2(l1=2e-6, l2=0),
                                           activation='tanh')(lstm_layer1)

        lstm_layer2 = tf.keras.layers.Dropout(rate=0.1)(lstm_layer2)

        output_layer = tf.keras.layers.Dense(1, activation='linear')(lstm_layer2)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer,  name='convlstm')

        return model
