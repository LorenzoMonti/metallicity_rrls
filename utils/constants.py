"""
This module defines project-level constants.
"""
import os
import numpy as np
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from . import regressor_tools as rt
from . import experiments_tools as et


#Paths
module = "RegressionExperiment"
data_path = "data/rrls/catalog/"
rrl_path = "data/rrls/raw_datasets/raw_dataset_6696/"
problems = ["rrls"]       
regressors = ["bigru"]
iterations = [3]
used_ids = "ids.txt"
results = "train_predictions.png"

# Pre-processing and cross validation
is_spline = True # if spline method has been applied or not
dl = [True] # if deep learning models are used
dens_weight = 0.5 # alpha parameter in density weight
spline_points = et.find_max_points(data_path)
rrls_number = len(os.listdir(data_path))

# rrls dataset
max_rrl_point = et.find_max_points(data_path)
source_id = "source_id"
period = "P_final"
metallicity = "FeH"
metallicity_error = "FeH_error"

# Model parameters
dl_array = np.array([16, 16, 'l1', 5e-6, 5e-6, 0, 0, 0.1, 0.1])
hparam_grid = [dl_array]
learning_rate=0.01
beta_1=0.9
beta_2=0.999
epsilon=1e-07
amsgrad=False
loss = MeanSquaredError()
optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)
metrics = [RootMeanSquaredError()]
n_epochs=20000
batch_size=256

# callbacks parameters
auto_stop="early"
min_delta=1e-5
patience=1000
optimize_lr=False, 
n_training_epochs=50000
is_checkpoint=False
save_model=True, 
n_zoom=200
n_update=100
eval_metrics=['root_mean_squared_error']
model_kwargs = {'n_timesteps': max_rrl_point, 'n_channels': 2, 'n_meta': 0, 'hparams': hparam_grid}
compile_kwargs = {'optimizer': optimizer, 'loss': loss, 'metrics': metrics, 'weighted_metrics': []}

# Metrics
metrics_t = {'r2': [], 'wrmse': [], 'wmae': [], 'rmse': [], 'mae': []}
metrics_v = {'r2': [], 'wrmse': [], 'wmae': [], 'rmse': [], 'mae': []}
callbacks = rt.setup_callbacks(auto_stop="early", min_delta=1e-5, patience=1000, optimize_lr=False,
                            n_training_epochs=50000, is_checkpoint=False, save_model=True,
                            n_zoom=200, n_update=100,eval_metrics=['root_mean_squared_error'])