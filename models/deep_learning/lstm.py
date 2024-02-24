import tensorflow as tf

from models.deep_learning.deep_learning_models import DLRegressor
from utils.constants import is_spline

class LSTMRegressor(DLRegressor):
    """
    This is a class implementing the LSTM model for time series regression.
    ! Remember to use regularization techniques such as dropout, weight decay, 
    and batch normalization. !
    """

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=200,
            batch_size=256,
            loss="mean_squared_error",
            metrics=None
    ):
        """
        Initialise the LSTM model

        Inputs:
            output_directory: path to store results/models
            input_shape: input shape for the models
            verbose: verbosity for the models
            epochs: number of epochs to train the models
            batch_size: batch size to train the models
            loss: loss function for the models
            metrics: metrics for the models
        """

        self.name = "LSTM"
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
        Build the LSTM model

        Inputs:
            input_shape: input shape for the model
        """

        if is_spline:
            input_layer = tf.keras.layers.Input(input_shape)
            lstm_layer1 = tf.keras.layers.LSTM(20, return_sequences=True,
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=2e-6),  
                                           recurrent_regularizer=tf.keras.regularizers.L1L2(l1=2e-6, l2=0),
                                           activation='tanh')(input_layer)
        else:
            input_layer = tf.keras.layers.Input(shape=(None, 2))         
            mask = tf.keras.layers.Masking(mask_value=-1, input_shape=input_shape).compute_mask(input_layer)
            lstm_layer1 = tf.keras.layers.LSTM(20, return_sequences=True,
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=2e-6),  
                                           recurrent_regularizer=tf.keras.regularizers.L1L2(l1=2e-6, l2=0),
                                           activation='tanh')(input_layer, mask=mask)

        lstm_layer1 = tf.keras.layers.Dropout(rate=0.2)(lstm_layer1)
        
        lstm_layer2 = tf.keras.layers.LSTM(16, return_sequences=True, 
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=2e-6),  
                                           recurrent_regularizer=tf.keras.regularizers.L1L2(l1=2e-6, l2=0),
                                           activation='tanh')(lstm_layer1)

        lstm_layer2 = tf.keras.layers.Dropout(rate=0.2)(lstm_layer2)

        lstm_layer3 = tf.keras.layers.LSTM(8, return_sequences=False, 
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=2e-6),  
                                           recurrent_regularizer=tf.keras.regularizers.L1L2(l1=2e-6, l2=0),
                                           activation='tanh')(lstm_layer2)

        lstm_layer3 = tf.keras.layers.Dropout(rate=0.1)(lstm_layer3)

        output_layer = tf.keras.layers.Dense(1, activation='linear')(lstm_layer3)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer,  name='lstm')

        return model
