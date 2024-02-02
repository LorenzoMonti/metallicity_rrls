import tensorflow as tf

from models.deep_learning.deep_learning_models import DLRegressor
from utils.constants import is_spline

class GRURegressor(DLRegressor):
    """
    This is a class implementing the GRU model for time series regression.
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
        Initialise the GRU model

        Inputs:
            output_directory: path to store results/models
            input_shape: input shape for the models
            verbose: verbosity for the models
            epochs: number of epochs to train the models
            batch_size: batch size to train the models
            loss: loss function for the models
            metrics: metrics for the models
        """

        self.name = "GRU"
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
        Build the GRU model

        Inputs:
            input_shape: input shape for the model
        """
        if is_spline:
            input_layer = tf.keras.layers.Input(input_shape)
            gru_layer1 = tf.keras.layers.GRU(20, return_sequences=True, 
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=2e-6),  
                                           recurrent_regularizer=tf.keras.regularizers.L1L2(l1=2e-6, l2=0),
                                           activation='tanh')(input_layer)
        else:
            input_layer = tf.keras.layers.Input(shape=(None, 2))         
            mask = tf.keras.layers.Masking(mask_value=-1, input_shape=input_shape).compute_mask(input_layer)
            gru_layer1 = tf.keras.layers.GRU(20, return_sequences=True, 
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=2e-6),  
                                           recurrent_regularizer=tf.keras.regularizers.L1L2(l1=2e-6, l2=0),
                                           activation='tanh')(input_layer, mask=mask)

        gru_layer1 = tf.keras.layers.Dropout(rate=0.2)(gru_layer1)
        
        gru_layer2 = tf.keras.layers.GRU(16, return_sequences=True, 
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=2e-6),  
                                           recurrent_regularizer=tf.keras.regularizers.L1L2(l1=2e-6, l2=0),
                                           activation='tanh')(gru_layer1)

        gru_layer2 = tf.keras.layers.Dropout(rate=0.2)(gru_layer2)

        gru_layer3 = tf.keras.layers.GRU(8, return_sequences=False, 
                                           kernel_regularizer=tf.keras.regularizers.L1L2(l1=0, l2=2e-6),  
                                           recurrent_regularizer=tf.keras.regularizers.L1L2(l1=2e-6, l2=0),
                                           activation='tanh')(gru_layer2)

        gru_layer3 = tf.keras.layers.Dropout(rate=0.1)(gru_layer3)

        output_layer = tf.keras.layers.Dense(1, activation='linear')(gru_layer3)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss=self.loss,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False),
                      metrics=self.metrics,
                      weighted_metrics=[])

        return model
