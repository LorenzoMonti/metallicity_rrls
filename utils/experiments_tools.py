import math
from time import time
import glob
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KernelDensity
import tensorflow as tf
from scipy.stats import gaussian_kde, binned_statistic as binstat
from tensorflow.keras.utils import pad_sequences
from denseweight import DenseWeight
import matplotlib.pyplot as plt

############################################
#          Run experiments (ML)            #
############################################

def split_dataset(x_array, y_array, rrls_number, weights, weights_var):

    # split dataset
    all_indices = list(range(rrls_number))
    train_indices, test_indices = train_test_split(all_indices, test_size=0.1)

    x_train = x_array[train_indices,:,:]
    x_test = x_array[test_indices,:,:]

    y_train = y_array[train_indices] 
    y_test = y_array[test_indices]

    weight_train, weight_val = weights[train_indices], weights[test_indices] 
    weight_var_train, weight_var_val = weights_var[train_indices], weights_var[test_indices]

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train.reshape(x_train.shape[0], -1)).reshape(x_train.shape)
    x_test = scaler.transform(x_test.reshape(x_test.shape[0], -1)).reshape(x_test.shape)

    return x_train, y_train, x_test, y_test, [weight_train, weight_val], [weight_var_train, weight_var_val]

def calculate_density_kernel(y_train, y_test, alpha, results):
    '''
    The bandwidth here acts as a smoothing parameter, controlling the tradeoff between 
    bias and variance in the result. A large bandwidth leads to a very smooth 
    (i.e. high-bias) density distribution. A small bandwidth leads to an unsmooth 
    (i.e. high-variance) density distribution.
    '''
    kde = KernelDensity(bandwidth=0.2)
    y = np.concatenate((y_train, y_test))
    #print(y)
    kde.fit(np.array(y).reshape(-1, 1))
    tgt_min = y.min()
    tgt_max = y.max()
    ymin_kde = tgt_min - 1
    ymax_kde = tgt_max + 1

    grid = np.linspace(ymin_kde, ymax_kde, 100)
    z = np.exp(kde.score_samples(list(grid.reshape(-1, 1))))
    z_points = np.exp(kde.score_samples(list(y_train.reshape(-1, 1))))
    weights = 1 / z_points

    # Define DenseWeight
    dw = DenseWeight(alpha=alpha)
    weights = dw.fit(y)
    #print(weights)
    #print(weights.shape)
    
    plot_weights(alpha, grid, z, z_points, y, weights, results)

    return weights

def calculate_regression_metrics(y_true, y_pred, eval_weights_val=None, y_true_val=None, y_pred_val=None):
    """
    This is a function to calculate metrics for regression.
    The metrics being calculated are RMSE and MAE.
    :param y_true:
    :param y_pred:
    :param y_true_val:
    :param y_pred_val:
    :return:
    """

    res = pd.DataFrame(data=np.zeros((1, 5), dtype=float), index=[0],
                       columns=['rmse', 'mae', 'wrmse', 'wmae', 'r2'])
    res['rmse'] = math.sqrt(mean_squared_error(y_true, y_pred))
    res['mae'] = mean_absolute_error(y_true, y_pred)
    res['wrmse'] = math.sqrt(mean_squared_error(y_true, y_pred, sample_weight=eval_weights_val))
    res['wmae'] = mean_absolute_error(y_true, y_pred, sample_weight=eval_weights_val)
    res['r2'] = r2_score(y_true, y_pred, sample_weight=eval_weights_val)

    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['rmse_val'] = math.sqrt(mean_squared_error(y_true_val, y_pred_val))
        res['mae_val'] = mean_absolute_error(y_true_val, y_pred_val)
        res['wrmse_val'] = math.sqrt(mean_squared_error(y_true_val, y_pred_val, sample_weight=eval_weights_val))
        res['wmae_val'] = mean_absolute_error(y_true_val, y_pred_val, sample_weight=eval_weights_val)
        res['r2_val'] = r2_score(y_true_val, y_pred_val, sample_weight=eval_weights_val)

    return res

############################################
#          Run experiments (DL)            #
############################################

def find_max_points(data_path):
    max_points = 0
    for file_name in glob.glob(data_path + '*.csv'):
        data = pd.read_csv(file_name, low_memory=False)
        if max_points < data.shape[0]: max_points = data.shape[0]
    return max_points

def write_csv(data_path, title, text_row):
    with open(os.path.join('.', data_path, title), 'a+') as f:
        f.write(text_row)

def read_csv_dataset(data_path, points):
    
    df = pd.read_csv(data_path + "rrls.csv")# read from csv
    #df.insert(2, "totamp_g", [1.3] * points, True) # add a column
    df.drop(["epoch_g"], axis=1) # remove a column
    
    return df

def save_rrls_ids(ids, used_ids):
    np.savetxt(os.path.join('.', 'output', used_ids), ids.T, fmt="%s")

def calculate_sample_weights(y, y_err=None, by_density=None):
    '''
    The bandwidth here acts as a smoothing parameter, controlling the tradeoff between 
    bias and variance in the result. A large bandwidth leads to a very smooth 
    (i.e. high-bias) density distribution. A small bandwidth leads to an unsmooth 
    (i.e. high-variance) density distribution.
    '''
    
    weights_var = 1.0 / y_err ** 2 # inverse squared

    if by_density:
        kde = gaussian_kde(y)
        ykde = kde(y)

        density_weighted_y = y[ykde > by_density]
        y_min_dens = np.min(density_weighted_y)
        y_max_dens = np.max(density_weighted_y)

        weights_dens = 1.0 / ykde
        if y_max_dens is not None:
            weights_dens[y > y_max_dens] = 1.0 / kde(y_max_dens)
        if y_min_dens is not None:
            weights_dens[y < y_min_dens] = 1.0 / kde(y_min_dens)
    else:
        weights_dens = np.ones(y.shape)

    weights = weights_dens * weights_var
    weights = MinMaxScaler(feature_range=(0.01, 1)).fit_transform(weights.reshape(-1, 1)).flatten()
    plot_sample_weights(y, by_density, weights_dens, ykde, weights)

    return weights, weights_var, weights_dens

def plot_sample_weights(y, by_density, weights_dens, ykde, weights):
    sort_indx = np.argsort(y)

    fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
    fig.subplots_adjust(bottom=0.13, top=0.95, hspace=0, left=0.15, right=0.85, wspace=0)
    ax2 = ax1.twinx()
    if by_density:
        ax2.plot(y, weights_dens / weights_dens.max(), 'k.', alpha=1, label="density weights")
        ax1.plot(y[sort_indx], ykde[sort_indx], 'g-', label="KDE")
    ax2.plot(y, weights, 'b,', alpha=1, label="weights")
    ax1.hist(y, facecolor='#3F3FFF', alpha=0.4, bins='sqrt', density=True, label="hist.")
    ax1.set_xlabel("$[Fe/H]$")
    ax1.set_ylabel('norm. density')
    ax2.set_ylabel('weights')
    ax1.tick_params(direction='in')
    ax2.tick_params(direction='in')
    
    plt.savefig('./output/weights.png', format='png', dpi=300)
    plt.close(fig)

def y_indexes(y, n_dev):

    isort = np.argsort(y)  # Indices of sorted Y values
    y_index = np.zeros(n_dev)
    y_index[isort] = np.arange(n_dev)
    y_index = np.floor(y_index / 20).astype(int)
    if np.min(np.bincount(y_index.astype(int))) < 5:
        y_index[y_index == np.max(y_index)] = np.max(y_index) - 1
    return y_index

def read_time_series(name_list, source_dir, time_steps, periods=None, max_phase=1.0):

    print("Reading time series...", file=sys.stderr)
    scaler = StandardScaler(copy=True, with_mean=True, with_std=False)
    extension = '.csv'
    times_dict, mags_dict, phases_dict = dict(), dict(), dict()
    X_list, times, phases, mags = list(), list(), list(), list()       
    X = np.zeros((len(name_list), time_steps, 2))  # Input shape required by an RNN: (batch_size, time_steps, features)
    
    for i, name in enumerate(name_list):
        print('Reading data for {}\r'.format(name), end="", file=sys.stderr)
        series = np.genfromtxt(os.path.join(".", source_dir, name + extension), unpack=True, delimiter=",", skip_header=1)
        phase = series[2]
        magnitude = series[3]
        phasemask = (phase < max_phase)
        phase = phase[phasemask]
        magnitude = magnitude[phasemask]

        if periods is not None: time = phase * periods[i]
        else: time = phase

        magnitude = scaler.fit_transform(magnitude.reshape(-1, 1)).flatten()
        times.append(time)
        mags.append(magnitude)
        phases.append(phase)

    # add padding (-1) to the time series
    X[:, :, 0] = pad_sequences(times, maxlen = time_steps, dtype='float64', padding='post', truncating='post', value=-1)
    X[:, :, 1] = pad_sequences(mags, maxlen = time_steps, dtype='float64', padding='post', truncating='post', value=-1)

    X_list.append(X)
    mags_dict['g'], phases_dict['g'] = mags, phases
    X = np.concatenate(X_list, axis=2)

    return X, mags_dict, phases_dict

def get_model_signature(weights, params, lr, batch_size):
    return "w" + str(weights) + "__" + '_'.join(map(str, params)) + "_lr" + str(lr) + "_Nb" + str(batch_size)

def get_folds(x, yi):
    splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)
    splitter.random_state = 42
    return list(splitter.split(x, yi, groups=None))

def fitting_cv(model, folds: list, x_list: list or tuple, y, compile_kwargs: dict = {},
                   initial_weights: list = None, sample_weight_fit=None, sample_weight_eval=None,
                   n_epochs: int = 1, batch_size: int = None, shuffle=True, verbose: int = 0,
                   callbacks: list = [], metrics: list or tuple = None,log_prefix='', ids=None,
                   rootdir='.', filename_train='train.csv', filename_val='val.csv'):

    
    first_fold = True
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    histories, model_weights, scalers_folds = list(), list(), list()
    Y_train_collected, Y_val_collected = np.array([]), np.array([])
    Y_train_pred_collected, Y_val_pred_collected = np.array([]), np.array([])
    fitting_weights_train_collected, fitting_weights_val_collected = np.array([]), np.array([])
    eval_weights_train_collected, eval_weights_val_collected = np.array([]), np.array([])
    ids_train_collected, ids_val_collected = np.array([]), np.array([])
    numcv_t, numcv_v = np.array([]), np.array([])

    for i_cv, (train_index, val_index) in enumerate(folds):

        tf.keras.backend.clear_session()
        tf.random.set_seed(42)
        model_ = model.get_model()

        if first_fold:
            first_fold = False
            if initial_weights is None:
                initial_weights = model_.get_weights()
        else:
            # Initialize model weights:
            model_.set_weights(initial_weights)

        model_.compile(**compile_kwargs)
        print("fold " + str(i_cv + 1) + "/" + str(len(folds)))
        print("n_train = {}  ;  n_val = {}".format(train_index.shape[0], val_index.shape[0]))
        callbacks_fold = callbacks + [tf.keras.callbacks.CSVLogger(os.path.join(rootdir, log_prefix + f"_fold{i_cv + 1}.log"))]

        x_train_list, x_val_list, scalers = list(), list(), list()
        
        for i, x in enumerate(x_list):
            x_train, x_validation = x[train_index], x[val_index]
            x_train_list.append(x_train)
            x_val_list.append(x_validation)

        y_train, y_val = y[train_index], y[val_index]


        fitting_weights_train, fitting_weights_val = sample_weight_fit[train_index], sample_weight_fit[val_index]
        eval_weights_train, eval_weights_val = sample_weight_eval[train_index], sample_weight_eval[val_index]
        ids_t, ids_v = ids[train_index], ids[val_index]

        # fit the model
        history = model_.fit(x=x_train_list, y=y_train, sample_weight=fitting_weights_train,
                             epochs=n_epochs, initial_epoch=0, batch_size=batch_size, shuffle=shuffle,
                             validation_data=(x_val_list, y_val, fitting_weights_val), verbose=verbose,
                             callbacks=callbacks_fold, validation_freq=1)
        
        Y_train_pred = (model_.predict(x_train_list)).flatten()
        Y_val_pred = (model_.predict(x_val_list)).flatten()
        histories.append(history)
        model_weights.append(model_.get_weights())
        scalers_folds.append(scalers.copy())

        Y_train_collected = np.hstack((Y_train_collected, y_train))
        Y_val_collected = np.hstack((Y_val_collected, y_val))
        Y_train_pred_collected = np.hstack((Y_train_pred_collected, Y_train_pred))
        Y_val_pred_collected = np.hstack((Y_val_pred_collected, Y_val_pred))
        fitting_weights_train_collected = np.hstack((fitting_weights_train_collected, fitting_weights_train))
        fitting_weights_val_collected = np.hstack((fitting_weights_val_collected, fitting_weights_val))
        eval_weights_train_collected = np.hstack((eval_weights_train_collected, eval_weights_train))
        eval_weights_val_collected = np.hstack((eval_weights_val_collected, eval_weights_val))
        ids_train_collected = np.hstack((ids_train_collected, ids_t))
        ids_val_collected = np.hstack((ids_val_collected, ids_v))
        numcv_t = np.hstack((numcv_t, np.ones(Y_train_pred.shape).astype(int) * i_cv))
        numcv_v = np.hstack((numcv_v, np.ones(Y_val_pred.shape).astype(int) * i_cv))

        #sava data 
        val_arr = np.rec.fromarrays((ids_v, y_val, Y_val_pred),names=('id', 'true_val', 'pred_val'))
        train_arr = np.rec.fromarrays((ids_t, y_train, Y_train_pred), names=('id', 'true_train', 'pred_train'))
        save_data_cv(rootdir, [filename_val, filename_train], val_arr, train_arr, i_cv)

        for metric in metrics:
                score_train = metric(y_train, Y_train_pred, sample_weight=eval_weights_train)
                score_val = metric(y_val, Y_val_pred, sample_weight=eval_weights_val)
                print_metrics_cv(metric, score_train, score_val)
    
    val_arr = np.rec.fromarrays((ids_val_collected, numcv_v, Y_val_collected, Y_val_pred_collected), names=('id', 'fold', 'true_val', 'pred_val'))
    train_arr = np.rec.fromarrays((ids_train_collected, numcv_t, Y_train_collected, Y_train_pred_collected), names=('id', 'fold', 'true_train', 'pred_train'))
    save_data_global_cv(rootdir, [filename_val, filename_train], val_arr, train_arr)

    cv_train_output = (Y_train_collected, Y_train_pred_collected, eval_weights_train_collected, ids_train_collected, numcv_t)
    cv_val_output = (Y_val_collected, Y_val_pred_collected, eval_weights_val_collected, ids_val_collected, numcv_v)
    
    return cv_train_output, cv_val_output, model_weights, scalers_folds, histories

def roots_mean_squared_error(y, y_pred, sample_weight=None):
    """
    Compute the root mean squared error metric.
    """
    value = mean_squared_error(y, y_pred, sample_weight=sample_weight)
    return np.sqrt(value)

def save_data_cv(rootdir, filenames, val_arr, train_arr, i_cv):
    np.savetxt(os.path.join(rootdir, filenames[0] + '_cv{}.csv'.format(i_cv + 1)), val_arr, fmt='%s %f %f')
    np.savetxt(os.path.join(rootdir, filenames[1] + '_cv{}.csv'.format(i_cv + 1)), train_arr, fmt='%s %f %f')

def save_data_global_cv(rootdir, filename, val_arr, train_arr):
    np.savetxt(os.path.join(rootdir, filename[0] + '.csv'), val_arr, fmt='%s %d %f %f')
    np.savetxt(os.path.join(rootdir, filename[1] + '.csv'), train_arr, fmt='%s %d %f %f')

def print_metrics_cv(metric, score_train, score_val):
    print(metric.__name__, "  (T) = {0:.3f}".format(score_train))
    print(metric.__name__, "  (V) = {0:.3f}".format(score_val))

def compute_regression_metrics(y, y_pred, metrics: dict = None, sample_weight=None):
    """
    Compute regression metrics and append them to list of metrics in a dictionary.
    :param y: numpy.ndarray
        Array of the true values.
    :param y_pred: numpy.ndarray
        Array of the predicted values.
    :param metrics: dict or None
        A dictionary of metric lists. If None, a new dictionary will be created.
    :param sample_weight: numpy.ndarray or None
        Array of the sample weights.
    :return: metrics: dict
        Dictionary of metric lists.
    """    
    metrics = {'r2': [], 'wrmse': [], 'wmae': [], 'rmse': [], 'mae': []}

    if sample_weight is not None:
        metrics['r2'].append(r2_score(y, y_pred, sample_weight=sample_weight))
        metrics['wrmse'].append(np.sqrt(mean_squared_error(y, y_pred, sample_weight=sample_weight)))
        metrics['wmae'].append(mean_absolute_error(y, y_pred, sample_weight=sample_weight))
    metrics['rmse'].append(np.sqrt(mean_squared_error(y, y_pred)))
    metrics['mae'].append(mean_absolute_error(y, y_pred))

    return metrics

def print_regression_metrics(metrics_1, item, metrics_1_name="training", metrics_2=None, metrics_2_name="\tCV "):
    assert isinstance(metrics_1, dict), "metrics_1 must be of type dict"

    if metrics_2 is not None:

        assert isinstance(metrics_2, dict), "metrics_2 must be of type dict"

        print("  metric  |  {0:10s} |  {1:10s} |".format(metrics_1_name, metrics_2_name))
        print(" --------------------------------------")
        for key in metrics_1.keys():
            print(" {0:8s} | {1:10.4f}  | {2:10.4f}  |".format(key, metrics_1[key][item], metrics_2[key][item]))
        print(" --------------------------------------\n")

    else:

        print("  metric |  {}  |".format(metrics_1_name))
        print(" ----------------------")
        for key in metrics_1.keys():
            print(" {0:8s} | {1:10.4f}  |".format(key, metrics_1[key][item]))
        print(" ----------------------\n")

def print_total_metrics_cv(hparam, metrics_test, metrics_validation):
    print("\n-------------------------------------------")
    print("\nhparams = {}".format(hparam))
    print("-------------------------------------------")
    print("Regression metrics (folds):")
    print_regression_metrics(metrics_test, 0, metrics_1_name="training", 
                             metrics_2=metrics_validation, metrics_2_name="\tCV ")

def save_models(weights_file="model_weights", model=None, model_weights=None, suffix=""):
    
    if model is not None:
        if isinstance(model_weights, (list, tuple, np.ndarray)):
            for ii, ww in enumerate(model_weights):
                model.set_weights(ww)
                # Serialize weights to HDF5:
                model.save_weights(weights_file + suffix + '_' + str(ii) + ".h5")
        else:
            # Serialize weights to HDF5:
            model.save_weights(weights_file + suffix + ".h5")