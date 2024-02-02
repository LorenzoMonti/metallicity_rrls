import tensorflow as tf

from models.deep_learning.deep_learning_models import DLRegressor
from utils.constants import is_spline

class BiGRURegressor(DLRegressor):
    """
    This is a class implementing the BiGRU model for time series regression.
    ! Remember to use regularization techniques such as dropout, weight decay, 
    and batch normalization. !
    """

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=50000,
            batch_size=256,
            loss="mean_squared_error",
            metrics=None
    ):
        """
        Initialise the BiGRU model

        Inputs:
            output_directory: path to store results/models
            input_shape: input shape for the models
            verbose: verbosity for the models
            epochs: number of epochs to train the models
            batch_size: batch size to train the models
            loss: loss function for the models
            metrics: metrics for the models
        """

        self.name = "BiGRU"
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
        Build the BiGRU model

        Inputs:
            input_shape: input shape for the model
        """


        dim1, dim2, rtype, rec_r1, rec_r2, kernel_r1, kernel_r2, dropout1, dropout2 = 16, 16, 'l1', 2e-6, 2e-6, 2e-6, 2e-6, 0.2, 0.2

        assert rtype == 'l1' or rtype == 'l2', "rtype must be either 'l1' or 'l2'"

        if rtype == 'l1':
            kernel_regularizer1 = tf.keras.regularizers.L1L2(l1=float(kernel_r1), l2=0)
            kernel_regularizer2 = tf.keras.regularizers.L1L2(l1=float(kernel_r2), l2=0)
            recurrent_regularizer1 = tf.keras.regularizers.L1L2(l1=float(rec_r1), l2=0)
            recurrent_regularizer2 = tf.keras.regularizers.L1L2(l1=float(rec_r2), l2=0)
        else:
            kernel_regularizer1 = tf.keras.regularizers.L1L2(l1=0, l2=float(kernel_r1))
            kernel_regularizer2 = tf.keras.regularizers.L1L2(l1=0, l2=float(kernel_r2))
            recurrent_regularizer1 = tf.keras.regularizers.L1L2(l1=0, l2=float(rec_r1))
            recurrent_regularizer2 = tf.keras.regularizers.L1L2(l1=0, l2=float(rec_r2))

        if is_spline:
            input_layer = tf.keras.layers.Input(input_shape)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(dim1), return_sequences=True, kernel_regularizer=kernel_regularizer1,
                            recurrent_regularizer=recurrent_regularizer1))(input_layer)
        else:
            input_layer = tf.keras.layers.Input(shape=(None, 2))
            mask = tf.keras.layers.Masking(mask_value=-1, input_shape=input_shape).compute_mask(input_layer)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(dim1), return_sequences=True, kernel_regularizer=kernel_regularizer1,
                            recurrent_regularizer=recurrent_regularizer1))(input_layer, mask=mask)

        if float(dropout1) > 0:
            x = tf.keras.layers.Dropout(rate=float(dropout1))(input_layer)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(int(dim2), return_sequences=False, kernel_regularizer=kernel_regularizer2,
                            recurrent_regularizer=recurrent_regularizer2))(x)

        if float(dropout2) > 0:
            x = tf.keras.layers.Dropout(rate=float(dropout2))(x)

        x = tf.keras.layers.Dense(1, activation=None, name='DenseLayer')(x)

        model = tf.keras.models.Model(inputs=input_layer, outputs=x, name='bigru')

        return model
