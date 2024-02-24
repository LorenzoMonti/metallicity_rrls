import tensorflow as tf

from models.deep_learning.deep_learning_models import DLRegressor
from utils.constants import is_spline

class FCNRegressor(DLRegressor):
    """
    This is a class implementing the FCN model for time series regression.
    The code is adapted from https://github.com/hfawaz/dl-4-tsc designed for time series classification.
    """

    def __init__(
            self,
            output_directory,
            input_shape,
            verbose=False,
            epochs=200,
            batch_size=16,
            loss="mean_squared_error",
            metrics=None
    ):
        """
        Initialise the FCN model

        Inputs:
            output_directory: path to store results/models
            input_shape: input shape for the models
            verbose: verbosity for the models
            epochs: number of epochs to train the models
            batch_size: batch size to train the models
            loss: loss function for the models
            metrics: metrics for the models
        """

        self.name = "FCN"
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
        Build the FCN model

        Inputs:
            input_shape: input shape for the model
        """
        kernel_regularizer = tf.keras.regularizers.L1L2(l2=float(2e-6), l1=0)
        bias_regularizer = tf.keras.regularizers.L1L2(l2=float(2e-6), l1=0)
        
        input_layer = tf.keras.layers.Input(input_shape)

        conv1 = tf.keras.layers.Conv1D(filters=128,
                                       kernel_size=8,
                                       padding='same',
                                       kernel_regularizer=kernel_regularizer,
                                       bias_regularizer=bias_regularizer)(input_layer)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.Activation(activation='relu')(conv1)

        conv2 = tf.keras.layers.Conv1D(filters=256,
                                       kernel_size=5,
                                       padding='same',
                                       kernel_regularizer=kernel_regularizer,
                                       bias_regularizer=bias_regularizer)(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.Activation('relu')(conv2)

        conv3 = tf.keras.layers.Conv1D(128,
                                       kernel_size=3,
                                       padding='same',
                                       kernel_regularizer=kernel_regularizer,
                                       bias_regularizer=bias_regularizer)(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.Activation('relu')(conv3)

        gap_layer = tf.keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = tf.keras.layers.Dense(1, activation='linear')(gap_layer)

        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer,  name='fcn')

        return model
