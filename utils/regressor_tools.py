from time import time
import numpy as np
import tensorflow as tf

name = "RegressorTools"

classical_ml_models = ["xgboost", "svr", "random_forest"]
deep_learning_models = ["fcn", "resnet", "inception", "convlstm", "lstm", "bilstm", "gru", "bigru", "convgru"]
tsc_models = ["rocket"]
linear_models = ["lr", "ridge"]
all_models = classical_ml_models + deep_learning_models + linear_models

############################################
#               Regressor                  #
############################################

def fit_regressor(output_directory, regressor_name, X_train, y_train, density_weight, fold_n = None,
                  X_val=None, y_val=None, itr=1):
    """
    This is a function to fit a regression model given the name and data
    :param output_directory:
    :param regressor_name:
    :param X_train:
    :param y_train:
    :param density_weight:
    :param X_val:
    :param y_val:
    :param itr:
    :return:
    """
    print("[{}] Fitting regressor".format(name))
    start_time = time()

    input_shape = X_train.shape[1:]

    regressor = create_regressor(regressor_name, input_shape, output_directory, itr)
    if (X_val is not None) and (regressor_name in deep_learning_models):
        if fold_n is not None: regressor.fit(X_train, y_train, density_weight, fold_n, X_val, y_val)
        else: regressor.fit(X_train, y_train, density_weight, "", X_val, y_val)
    else:
        regressor.fit(X_train, y_train, density_weight[0])
    elapsed_time = time() - start_time
    print("[{}] Regressor fitted, took {}s".format(name, elapsed_time))
    return regressor

def create_regressor(regressor_name, input_shape, output_directory, verbose=1, itr=1):
    """
    This is a function to create the regression model
    :param regressor_name:
    :param input_shape:
    :param output_directory:
    :param verbose:
    :param itr:
    :return:
    """
    print("[{}] Creating regressor".format(name))
    # SOTA TSC deep learning
    if regressor_name == "resnet":
        from models.deep_learning import resnet
        return resnet.ResNetRegressor(output_directory, input_shape, verbose)
    if regressor_name == "fcn":
        from models.deep_learning import fcn
        return fcn.FCNRegressor(output_directory, input_shape, verbose)
    if regressor_name == "inception":
        from models.deep_learning import inception
        return inception.InceptionTimeRegressor(output_directory, input_shape, verbose)
    if regressor_name == "convlstm":
        from models.deep_learning import convlstm
        return convlstm.ConvLSTMRegressor(output_directory, input_shape, verbose)
    if regressor_name == "lstm":
        from models.deep_learning import lstm
        return lstm.LSTMRegressor(output_directory, input_shape, verbose)
    if regressor_name == "bilstm":
        from models.deep_learning import bilstm
        return bilstm.BiLSTMRegressor(output_directory, input_shape, verbose)
    if regressor_name == "gru":
        from models.deep_learning import gru
        return gru.GRURegressor(output_directory, input_shape, verbose)
    if regressor_name == "bigru":
        from models.deep_learning import bigru
        return bigru.BiGRURegressor(output_directory, input_shape, verbose)
    if regressor_name == "convgru":
        from models.deep_learning import convgru
        return convgru.ConvGRURegressor(output_directory, input_shape, verbose)

    # classical ML models
    if regressor_name == "xgboost":
        from models.classical_models import XGBoostRegressor
        kwargs = {"n_estimators": 100,
                  "n_jobs": 0,
                  "learning_rate": 0.1,
                  "random_state": itr - 1,
                  "verbosity  ": verbose}
        return XGBoostRegressor(output_directory, verbose, kwargs)
    if regressor_name == "random_forest":
        from models.classical_models import RFRegressor
        kwargs = {"n_estimators": 100,
                  "n_jobs": -1,
                  "random_state": itr - 1,
                  "verbose": verbose}
        return RFRegressor(output_directory, verbose, kwargs)
    if regressor_name == "svr":
        from models.classical_models import SVRRegressor
        return SVRRegressor(output_directory, verbose)

    # linear models
    if regressor_name == "lr":
        from models.classical_models import LinearRegressor
        kwargs = {"fit_intercept": True,
                  "n_jobs": -1}
        return LinearRegressor(output_directory, kwargs, type=regressor_name)
    if regressor_name == "ridge":
        from models.classical_models import LinearRegressor
        kwargs = {"fit_intercept": True}
        return LinearRegressor(output_directory, kwargs, type=regressor_name)

def setup_callbacks(auto_stop=None, min_delta=10e-5, patience=200,
                    optimize_lr=False, min_learning_rate=0.0001, n_training_epochs=100, lr_increment_coeff=0.9,
                    is_checkpoint=False, checkpoint_period=100,
                    save_model=False, n_zoom=100, n_update=100, eval_metrics=['accuracy'], figname="liveplot"):
    callbacks = []

    # Stop training automatically:
    if auto_stop is not None:
        if auto_stop == 'late':
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='loss', min_delta=min_delta, patience=patience, verbose=1, mode='min',
                baseline=None, restore_best_weights=False))
        elif auto_stop == 'early':  # i.e., early stopping
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1, mode='min',
                baseline=None, restore_best_weights=True))

    # Change learning rate at each epoch to find optimal value:
    if optimize_lr:
        lr_schedule = tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: min_learning_rate * 10 ** (epoch / (n_training_epochs * lr_increment_coeff)))
        callbacks.append(lr_schedule)

    if is_checkpoint and save_model:
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint('weights{epoch:03d}_{loss:.3f}.h5', monitor='loss',
                                                              save_best_only=False, save_weights_only=True,
                                                              period=checkpoint_period)
        callbacks.append(model_checkpoint)

    return callbacks