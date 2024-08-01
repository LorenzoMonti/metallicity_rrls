import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

models = ['bigru', 'bilstm', 'convgru', 'convlstm', 'fcn', 'gru', 'inception', 'lstm', 'resnet']
iteration = 'itr_12'
files = ["/train_total.csv", "/val_total.csv"]
list_r2, list_rmse, list_mae, list_wrmse, list_wmae = list(), list(), list(), list(), list()


for model in models:
    print("#########################################################")
    print(model)
    print("#########################################################")
    
    for file in files:
        print(file)
        file_path = "output/regression_dl/" + model + "/rrls/" + iteration + file
        print(file_path)
        data = pd.read_csv(file_path, delimiter=' ')

        unique_values = data.iloc[:, 1].unique()

        for val in unique_values:
            #print(f'fold numero: {val}')
            filtered_data = data[data.iloc[:, 1] == val]
            ground_truth = filtered_data.iloc[:, 2]
            predicted_values = filtered_data.iloc[:, 3]
            
            r2 = r2_score(ground_truth, predicted_values)
            wrmse = np.sqrt(mean_squared_error(ground_truth, predicted_values, sample_weight=None))
            wmae = mean_absolute_error(ground_truth, predicted_values, sample_weight=None)
            rmse = np.sqrt(mean_squared_error(ground_truth, predicted_values))
            mae = mean_absolute_error(ground_truth, predicted_values)

            # Print the results
            #print(f"R^2: {r2}")
            #print(f"RMSE: {rmse}")
            #print(f"MAE: {mae}")

            list_r2.append(r2)
            list_rmse.append(rmse)
            list_mae.append(mae)
            list_wrmse.append(wrmse)
            list_wmae.append(wmae)

        #print(f'mean r2: {np.mean(list_r2)}')
        #print(f'mean mae: {np.mean(list_mae)}')
        #print(f'mean rmse: {np.mean(list_rmse)}')
        print(f'std r2: {np.std(list_r2)}')
        print(f'std wrmse: {np.std(list_wrmse)}')
        print(f'std wmae: {np.std(list_wmae)}')
        print(f'std rmse: {np.std(list_rmse)}')
        print(f'std mae: {np.std(list_mae)}')