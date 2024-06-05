import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import glob
import pickle
import matplotlib.pyplot as plt
from utils.constants import *
from utils.experiments_tools import *
from utils.regressor_tools import *
from utils.plot_tools import *

eval_all = True # if the evaluation is from all the stars known (we don't know the metallicity) or not
model_type = "DL" # ML or DL
path = "output/regression_dl/"
model = "bilstm/"
regressor_name = "bilstm"
ml_filename = "rrls/itr_1/finalized_model.sav"
dl_filename = "rrls/itr_13/"
catalog = "data/rrls/valutation_all_c/valutation_catalog/"
rrl_catalog = "data/rrls/valutation_all_c/"

testset_distribution = "rrls/itr_13"
results = "rrls/itr_13/val_predictions.png"
spline_points = et.find_max_points(catalog)
rrl_number = len(os.listdir(catalog))
predictions = list()
labels = np.empty

def find_not_matches(catalog, ids_rrls):
    id_directory = [sub.replace('.csv', '') for sub in os.listdir(catalog)]
    return  list(set(ids_rrls).difference(id_directory))

if eval_all:
    """
    In this case, the test-set is made up of all the known light curves, to verify the validity of the model. 
    Not all sources have either the phi_{31} value or metallicity values. Then, only the metallicity values 
    ​​predicted and source_id by the model will be printed (for each folds), and a file with average and std. 
    In the opposite case, however, the metallicity values ​​are compared with the ground truth.
    """
    input_dataset = read_csv_dataset(rrl_catalog, rrl_number)
    ids_dev = input_dataset[source_id].to_numpy().astype(str)
    print(f"Find rrls that not matched from light_curves folder and rrls.csv file: {find_not_matches(catalog, ids_dev)}")
    periods_input = input_dataset[period].to_numpy()
    rrls, _, _= read_time_series(ids_dev, catalog, spline_points, periods=periods_input, max_phase=1.0)
    #labels = input_dataset[metallicity].to_numpy()
    print(rrls.shape)
    #print(labels.shape)
    regressor = create_regressor(regressor_name, rrls.shape[1:], path + model + dl_filename)

    weights_list = glob.glob(os.path.join(path + model + dl_filename, 'w*.h5'))
    print(weights_list)

    y_pred_list = list()
    for ii, wf in enumerate(weights_list):
        # Recreate the exact same model, including its weights and the optimizer
        load_model = regressor.get_model()
        load_model.compile(**compile_kwargs)
        load_model.load_weights(wf)
        y = load_model.predict(rrls).flatten()
        np.savetxt(os.path.join(path + model + dl_filename + "predictions_test_" + str(ii)), np.rec.fromarrays((ids_dev, y), names=('ids', 'pred')), fmt='%s %f')
        y_pred_list.append(y)

    y_pred = np.mean(np.vstack(y_pred_list), axis=0)
    y_pred_std = np.std(np.vstack(y_pred_list), axis=0)

    outarr = np.rec.fromarrays((ids_dev, y_pred, y_pred_std), names=('ids', 'pred', 'pred_std'))
    np.savetxt(os.path.join(path + model + dl_filename + "predictions_test_mean_std"), outarr, fmt='%s %f %f')

else:
    if model_type == "ML":
        load_model = pickle.load(open(path + model + ml_filename, 'rb'))
    elif model_type == "DL":
        input_dataset = read_csv_dataset(rrl_catalog, rrl_number)
        ids_dev = input_dataset[source_id].to_numpy().astype(str)
        periods_input = input_dataset[period].to_numpy()
        rrls, _, _= read_time_series(ids_dev, catalog, spline_points, periods=periods_input, max_phase=1.0)
        labels = input_dataset[metallicity].to_numpy()
        print(rrls.shape)
        print(labels.shape)
        regressor = create_regressor(regressor_name, rrls.shape[1:], path + model + dl_filename)

        weights_list = glob.glob(os.path.join(path + model + dl_filename, 'w*.h5'))
        print(weights_list)

    if model_type == "ML":
        if len(rrls.shape) == 3:
            rrls = rrls.reshape(rrls.shape[0], rrls.shape[1] * rrls.shape[2])
        prediction = load_model.predict(rrls)
        print('Y prediction: ', prediction)
        print('Y ground truth', labels)
        predictions.append(prediction)
    else:
        
        y_pred_list = list()
        for ii, wf in enumerate(weights_list):
            # Recreate the exact same model, including its weights and the optimizer
            load_model = regressor.get_model()
            load_model.compile(**compile_kwargs)
            load_model.load_weights(wf)

            loss, acc = load_model.evaluate(rrls, labels, verbose=2)

            print('Restored model, loss: {:5.2f}'.format(loss))
            #print('Y prediction: ', predictions)
            #print('Y ground truth', labels)
            y_pred_list.append(load_model.predict(rrls).flatten())

    print(type(labels))
    y_pred = np.mean(np.vstack(y_pred_list), axis=0)
    y_pred_std = np.std(np.vstack(y_pred_list), axis=0)
    #outarr = np.rec.fromarrays((labels, y_pred, y_pred_std), names=('gt', 'pred', 'pred_std'))
    #np.savetxt(os.path.join(path + model + dl_filename + "predictions_test"), outarr, fmt='%f %f %f')

    plt.hist(labels, bins='auto')
    plt.title("Test-set distribution")
    plt.savefig(path + model + testset_distribution)
    plt.close()

    plot_predictions(labels, y_pred, rootdir=path + model + testset_distribution, suffix='testset', figformat='png')