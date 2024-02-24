import os
import numpy as np
import pandas as pd
from utils.tools import create_directory 
from utils.regressor_tools import *
from utils.experiments_tools import *
from utils.plot_tools import *
from utils.constants import *

if __name__ == '__main__':
    
    for problem in problems:
        print("#########################################################################")
        print("[{}] Starting Experiments".format(module))
        print("#########################################################################")
        print("[{}] Data path: {}".format(module, data_path))
        print("[{}] Problem: {}".format(module, problem))

        input_dataset = read_csv_dataset(rrl_path, rrls_number)
        ids_dev = input_dataset[source_id].to_numpy().astype(str)
        save_rrls_ids(ids_dev, used_ids) # save id_sources

        # loading the data. X_train and X_test are dataframe of N x n_dim
        print("[{}] Loading data".format(module))
        
        # period, X, magnitudes and phases
        periods_input = input_dataset[period].to_numpy()
        X, mags, phases = read_time_series(ids_dev, data_path, max_rrl_point, periods=periods_input, max_phase=1.0)
        # plot mags and phases
        plot_all_lc(phases['g'], mags['g'], figformat='png', fname="./output/" + "_all_lc")

        # y, y_error and y indexes
        y = input_dataset[metallicity].to_numpy()
        y_error = input_dataset[metallicity_error].to_numpy()
        yi = y_indexes(y, y.shape[0])
        
        # sample weights (imbalanced dataset)
        weights_dev, weights_var_dev, weights_dens = calculate_sample_weights(y, y_err=y_error, by_density=dens_weight)

        for num, regressor_name in enumerate(regressors):
            print("[{}] Regressor: {}".format(module, regressor_name))
            for itr in iterations:
                # create output directory
                output_directory = "output/regression/"

                if dl[num] != False:
                    output_directory = output_directory[:-1] + "_dl/"
                output_directory = output_directory + regressor_name + '/' + problem + '/itr_' + str(itr) + '/'
                create_directory(output_directory)

                print("[{}] Iteration: {}".format(module, itr))
                print("[{}] Output Dir: {}".format(module, output_directory))

                if dl[num]: # DL models            
                    
                    write_csv(output_directory, 'results.csv', "hyper-params,r2,wrmse,wmae,rmse,mae\n")
                    folds = get_folds(X, yi)
                    run_tag = get_model_signature(dens_weight, hparam_grid[num], learning_rate, batch_size)
                    regressor = create_regressor(regressor_name, X.shape[1:], output_directory, itr)
                    
                    cv_train_out, cv_val_out, model_weights, scalers, histories = fitting_cv(
                        regressor, folds, (X,), y, compile_kwargs=compile_kwargs, sample_weight_fit=weights_dev, 
                        sample_weight_eval=weights_var_dev,ids=ids_dev, n_epochs=n_epochs, batch_size=batch_size,
                        callbacks=callbacks, metrics=(r2_score, roots_mean_squared_error), log_prefix=run_tag, 
                        rootdir=output_directory, filename_val="./val" + str(regressor) + str(hparam_grid[num]),
                        filename_train="./train" + str(regressor) + str(hparam_grid[num]))

                    Y_train, Y_train_pred, eval_weights_train, ids_train, numcv_t = cv_train_out
                    Y_val, Y_val_pred, eval_weights_val, ids_val, numcv_v = cv_val_out

                    metrics_t = compute_regression_metrics(Y_train, Y_train_pred, metrics_t, sample_weight=eval_weights_train)
                    metrics_v = compute_regression_metrics(Y_val, Y_val_pred, metrics_v, sample_weight=eval_weights_val)
                    print_total_metrics_cv(hparam_grid[num], metrics_t, metrics_v)

                    progress_plots(histories, os.path.join(output_directory, 'progress_'), start_epoch=50, moving_avg=False, plot_folds=True,
                                    title=str(regressor_name) + ', hpars: ' + str(hparam_grid[num]) + ',\n batch size: ' + str(batch_size) + ', lr: ' + str(learning_rate))

                    plot_predictions(Y_train, Y_train_pred, y_val_true=Y_val, y_val_pred=Y_val_pred, rootdir=output_directory, suffix=run_tag + '', figformat='pdf')
                    save_models(weights_file = output_directory + "weights" + run_tag, model = regressor.get_model(), model_weights = model_weights)
                    
                    cv_metrics = f"{str(hparam_grid[num])},{metrics_v['r2']},{metrics_v['wrmse']},{metrics_v['wmae']},{metrics_v['rmse']},{metrics_v['mae']}\n"
                    write_csv(output_directory, 'results.csv', cv_metrics)

                else: # ML models
                    #split_dataset
                    X_train, y_train, X_test, y_test, weights, weights_var = split_dataset(X, y, rrls_number, weights_dev, weights_var_dev,)
                    # fit the regressor
                    regressor = fit_regressor(output_directory, regressor_name, X_train, y_train, [weights[0], weights[1]], None, X_test, y_test, itr=itr)
                    # start testing
                    y_pred = regressor.predict(X_test)
                    plot_training_predictions(y_test, y_pred, output_directory + results)
                    df_metrics = calculate_regression_metrics(y_test, y_pred, weights_var[1])
                    print(df_metrics)
                    # save the outputs
                    df_metrics.to_csv(output_directory + 'regression_experiment.csv', index=False)