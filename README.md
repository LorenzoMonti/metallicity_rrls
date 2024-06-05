# Metallicity RRLs

Machine learning and deep learning models to estimate metallicity from light curves.

## Data

...

## Dependencies

All Python packages needed are listed in [requirements.txt](requirements.txt) file
and can be installed simply using the pip command.

## Models

The following models are implemented in this repository:

### Classical ML models 

1. Support Vector Regression (SVR) - A wrapper function for sklearn [SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR) 
2. Random Forest Regressor (RF) - A wrapper function for sklearn [RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)
3. XGBoost (XGB) - A wrapper function for [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html) package

### Deep Learning for TSER 

1. Fully Convolutional Network ([FCN](https://github.com/hfawaz/dl-4-tsc))
2. Residual Network ([ResNet](https://github.com/hfawaz/dl-4-tsc))
3. Inception Time ([InceptionTime](https://github.com/hfawaz/InceptionTime))
4. LSTM
5. BiLSTM
6. ConvLSTM
7. GRU
8. BiGRU
9. ConvGRU

## Output

The output folder contains all the training logs, plots, and weights, divided by model.

## Utils

the folder contains all the utility files for creating models, training them, plotting the results, and saving weights and logs.

## Getting started

In order to run all the experiments, the entry point file is `run_experiments.py`, and you can find it in the root folder.

remember to add in the file `constants.py`:
* The list of regressors (variable `regressors`).
* The `dl` list that identifies (Boolean) whether the regressor is of the deep learning type or not.
* the grid of the `param_grid` hyperparameters (model-dependent). Machine learning models have no hyperparameters here (None).
Example. `regressors = ["random_forest", "convgru"]` `dl = [False, True]` e `hparam_grid = [None,  np.array([1, 1, 'l1', 5e-1, 5e-1, 0, 0, 0.1, 0.1])]`

## Support scripts

In the root folder, you can find some support scripts such as: 
* `pre-processing.py` is necessary for the pre-processing of the photonometric light curves.
* `plot_all_lightcurves.py` plot of the photonometric dataset of light curves (phase/magnitude).
* `evaluation_models.py` uses weights from the towed model to evaluate light curves on an unknown test dataset.
* `draw_models.py` draws all the implemented models (found in models/ folder).

## Acknowledgement

Some of the models used are based on [TSER project](https://github.com/ChangWeiTan/TS-Extrinsic-Regression)

## License

This project has an MIT-style license, as found in the [LICENSE](LICENSE) file.
