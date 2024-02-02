import time

import numpy as np
import tensorflow as tf

from models.time_series_models import TimeSeriesRegressor
from utils.tools import save_train_duration, save_test_duration
from utils.regressor_tools import setup_callbacks


def plot_epochs_metric(hist, file_name, model, metric='loss'):
    """
    Plot the train/test metrics of Deep Learning models

    Inputs:
        hist: training history
        file_name: save file name
        model: model name
        metric: metric
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(hist.history[metric], label="train")
    if "val_" + metric in hist.history.keys():
        plt.plot(hist.history['val_' + metric], label="val")

    min_train = np.min(hist.history["loss"])
    idx_train = np.argmin(hist.history["loss"])
    plt.plot(idx_train, min_train, "rx", label="best epoch")
    if "val_" + metric in hist.history.keys():
        plt.plot(idx_train, hist.history['val_' + metric][idx_train], "rx")

    plt.title(model + " " + metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    plt.legend(loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()


class DLRegressor(TimeSeriesRegressor):
    """
    This is a superclass for Deep Learning models for Regression
    """
    name = "DeepLearningTSR"
    model_init_file = "model_init.h5"
    best_model_file = "best_model.h5"

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
        Initialise the DL model

        Inputs:
            output_directory: path to store results/models
            input_shape: input shape for the models
            verbose: verbosity for the models
            epochs: number of epochs to train the models
            batch_size: batch size to train the models
            loss: loss function for the models
            metrics: metrics for the models
        """

        super().__init__(output_directory)
        print('[{}] Creating Regressor'.format(self.name))
        self.X_train = None
        self.y_train = None
        self.weights = None
        self.X_val = None
        self.y_val = None
        self.callbacks = None
        self.hist = None

        self.verbose = verbose
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        if metrics is None:
            metrics = ["mae"]
        self.metrics = metrics

        self.model = self.build_model(input_shape)

        if self.model is not None:
            self.model.summary()
            self.model.save_weights(self.output_directory + self.model_init_file)

    def get_model(self):
        return self.model

    def build_model(self, input_shape):
        """
        Build the DL models

        Inputs:
            input_shape: input shape for the models
        """
        pass

    def fit(self, x_train, y_train, weights, name="", x_val=None, y_val=None, monitor_val=False):
        """
        Fit DL models

        Inputs:
            x_train: training data (num_examples, num_timestep, num_channels)
            y_train: training target
            weights: weights for imbalanced dataset based on Gaussian Kernel Density
            x_val: validation data (num_examples, num_timestep, num_channels)
            y_val: validation target
            monitor_val: boolean indicating if model selection should be done on validation
        """
        print('[{}] Training'.format(self.name))

        start_time = time.perf_counter()

        self.X_train = x_train
        self.y_train = y_train
        self.weights = weights
        self.X_val = x_val
        self.y_val = y_val

        epochs = self.epochs
        batch_size = self.batch_size
        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        file_path = self.output_directory + self.best_model_file
        if (x_val is not None) and monitor_val:
           self.callbacks = setup_callbacks(auto_stop='early', min_delta=1e-5, patience=1000,
                                   optimize_lr=None, min_learning_rate=1e-3,
                                   n_training_epochs=50000,lr_increment_coeff=0.48,
                                   is_checkpoint=False, checkpoint_period=200,
                                   save_model=True, n_zoom=200, n_update=100,
                                   eval_metrics=['root_mean_squared_error'], figname='liveplot')
        else:
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                                              min_delta=1e-5, patience=50, 
                                                              verbose=1, mode='min', 
                                                              baseline=None, 
                                                              restore_best_weights=True)
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                             factor=0.5, patience=50,
                                                             min_lr=0.0001)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                                  monitor='loss',
                                                                  save_best_only=True)
            self.callbacks = [early_stopping, reduce_lr, model_checkpoint]

        # train the model
        if x_val is not None:
            if name == "": # DL without cv
                self.hist = self.model.fit(x_train, y_train,
                                       validation_data=(x_val, y_val, weights[1]),
                                       verbose=self.verbose,
                                       epochs=epochs,
                                       sample_weight=weights[0],
                                       batch_size=mini_batch_size,
                                       shuffle=True,
                                       callbacks=self.callbacks)
            else: # DL with cv (validation weight on validation data)
                self.hist = self.model.fit(x_train, y_train,
                                       validation_data=(x_val, y_val, weights[1]),
                                       verbose=self.verbose,
                                       epochs=epochs,
                                       sample_weight=weights[0],
                                       batch_size=mini_batch_size,
                                       shuffle=True,
                                       callbacks=self.callbacks)

        else: # ML
            self.hist = self.model.fit(x_train, y_train,
                                       verbose=self.verbose,
                                       epochs=epochs,
                                       sample_weight=weights,
                                       batch_size=mini_batch_size,
                                       shuffle=True,
                                       callbacks=self.callbacks)

        self.train_duration = time.perf_counter() - start_time

        save_train_duration(self.output_directory + name + 'train_duration.csv', self.train_duration)

        print('[{}] Training done!, took {}s'.format(self.name, self.train_duration))

        plot_epochs_metric(self.hist,
                           self.output_directory + name + 'epochs_loss.png',
                           metric='loss',
                           model=self.name)
        for m in self.metrics:
            plot_epochs_metric(self.hist,
                               self.output_directory + name + 'epochs_{}.png'.format(m),
                               metric=m,
                               model=self.name)