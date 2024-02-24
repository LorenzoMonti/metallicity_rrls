import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import glob
import pickle
import matplotlib.pyplot as plt

model_type = "DL" # ML or DL
path = "output/regression_dl/"
model = "bigru/"
ml_filename = "rrls/itr_1/finalized_model.sav"
dl_filename = "rrls/itr_1/weightsw0.5__16_16_l1_5e-06_5e-06_0_0_0.1_0.1_lr0.01_Nb256_0.h5"
catalog = "data/rrls/valutation_dataset/valutation_catalog/"
testset_distribution = "rrls/itr_1/testset-distribution.png"
results = "rrls/itr_1/val_predictions.png"
spline_points = 1000

if model_type == "ML":
    load_model = pickle.load(open(path + model + ml_filename, 'rb'))
elif model_type == "DL":
    # Recreate the exact same model, including its weights and the optimizer
    load_model = tf.keras.models.load_model(path + model + dl_filename)
    # Show the model architecture
    load_model.summary()

dataset_list, predictions = list(), list()
labels = np.empty

for file_name in glob.glob(catalog + '*.csv'):
    data = pd.read_csv(file_name, low_memory=False)
    dataset_list.append(np.array(data[["phase", "magnitude", "FeH"]].values.tolist()))

    if data.shape[0] != spline_points:
        print(file_name + str(data.shape))

    dataset_array = np.stack(dataset_list, axis=0)
    #print(dataset_array.shape)
    rrls = dataset_array[:,:,0:2]
    labels = np.reshape(dataset_array[:,:,2:], -1)[::spline_points]

if model_type == "ML":
        if len(rrls.shape) == 3:
            rrls = rrls.reshape(rrls.shape[0], rrls.shape[1] * rrls.shape[2])
        prediction = load_model.predict(rrls)
        print('Y prediction: ', prediction)
        print('Y ground truth', labels)
        predictions.append(prediction)
else:
        loss, acc = load_model.evaluate(rrls, labels, verbose=2)
        prediction = load_model.predict(rrls)
        print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
        print('Y prediction: ', prediction)
        print('Y ground truth', labels)
        predictions.append(prediction)

plt.hist(labels, bins='auto')
plt.title("Test-set distribution")
plt.savefig(path + model + testset_distribution)
plt.close()
#plt.show()

#print(predictions[0].flatten())
#print(labels)
#print(type(np.array(predictions)), type(labels))
x = np.linspace(0, 1, len(labels))
linex = np.linspace(min(labels), max(labels), len(labels))
liney = np.linspace(min(predictions[0].flatten()), max(predictions[0].flatten()), len(predictions[0].flatten()))
#print(x)
plt.figure(figsize=(19,12))
#plt.plot(x, labels, 'o', label="y_truth")
#plt.plot(x, predictions[0].flatten(), 'o', label="y_pred")
plt.plot(labels, predictions[0].flatten(), 'o', label="x=truth, y=pred")
plt.plot(liney, liney)
plt.xlabel("Ground Truth")
plt.ylabel("Predicted")
plt.savefig(path + model + results)
#plt.show()
