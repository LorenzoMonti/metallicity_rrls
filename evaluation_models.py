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

model_type = "DL" # ML or DL
path = "output/regression_dl/"
model = "gru/"
regressor_name = "gru"
ml_filename = "rrls/itr_1/finalized_model.sav"
dl_filename = "rrls/itr_2/"
catalog = "data/rrls/valutation_dataset/valutation_catalog/"
rrl_catalog = "data/rrls/valutation_dataset/"

testset_distribution = "rrls/itr_2"
results = "rrls/itr_2/val_predictions.png"
spline_points = et.find_max_points(catalog)
rrl_number = len(os.listdir(catalog))
predictions = list()
labels = np.empty

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

    #print("Layers: ")
    #for layer in load_model.layers: print(layer.get_config(), layer.get_weights())

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