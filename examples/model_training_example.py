"""
Script for linear model fitting and evaluation
"""
# %% Imports
# File
from pathlib import Path
import logging
import os

# Computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Model based
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
# from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# Metrics
from sklearn.model_selection import train_test_split

# Internal
from codebase import features

# %% Read Data
n_row = None  # number of rows to read from CSV
output_type = 'xco2'  # xco2, xch4

# data_dir = Path('/mnt/c/Users/ktopo/Desktop/ECE537/data')
data_dir = Path('C:/Users/ktopo/Desktop/ECE537/data')

if output_type == 'xco2':
    csv_name = data_dir / Path('co2_data.csv')
else:
    csv_name = data_dir / Path('ch4_data.csv')

df = pd.read_csv(csv_name, nrows=n_row)

lat = df['latitude'].to_numpy()
lon = df['longitude'].to_numpy()
epoch_time = df['time'].to_numpy()
carbon = df[output_type].to_numpy()

del df  # no longer needed, free up memory

# %% Pre-Process Data
X = features.preprocess(lat=lat, lon=lon, epoch_time=epoch_time)
y = carbon
n_feature = X.shape[-1]

# Split test data from the rest. The rest will be used for K-fold x-validation
test_size = 0.20
X_remain, X_test, y_remain, y_test = train_test_split(X, y, test_size=test_size)

del X,y  # remove data we no longer need


# %% Model configuration
n_fold = 3  # for cross-validation. If None, no cross-validation performed

# Define each hyperparameter as a list here. Options:
# 1) Brute force grid search of all parameters
# 2) Random sampling of parameters
# 3) Genetic algorithm given RANGES for parameters

# TODO - Give these each ranges and allow automatic selection
batch_sizes = [32]  # number of examples pased through at once for training
n_epochs = [15]  # number of passes through dataset
learn_rates = [0.00001]  # May be unique to Adam optimizer
loss_functions = ['mse']
optimize_methods = ['adam']
verbosity = 1

# List of length [n_hyperparam], each with dims (n_hyper0, n_hyper1, ...)
hyperparams = np.meshgrid(
    batch_sizes,
    n_epochs,
    learn_rates,
    loss_functions,
    optimize_methods,
    indexing='ij',
)

n_hyper = len(hyperparams)
for i_hyper in range(n_hyper):
    hyperparams[i_hyper] = hyperparams[i_hyper].flatten()

print('Combinations to be tested:')
for params in zip(*hyperparams):
    print(params)

n_model = len(hyperparams[0])

# %% Setup logger
log_file = 'training.log'
fmt = '[%(levelname)-s]: %(asctime)s \t %(message)s'

# Allows us to re-visit results
with open(log_file, 'w'):  # using this allows clearing of file
    logging.basicConfig(
        filename=log_file,  # if None, defaults to stdout
        filemode='w',  # doesnt seem to work
        level=logging.DEBUG,
        format=fmt
    )
    logger = logging.getLogger('LOG_BUDDY')

# %% Load Data and pre-process
# Define the K-fold Cross Validator
kfold = KFold(n_splits=n_fold, shuffle=True)

best_validation_loss = np.inf  # update this as models go along
best_model = None
best_idx = None
BestModel = None

# Loop through sets of hyperparameters
for i_model, params in enumerate(zip(*hyperparams)):
    logger.info(f'\n=======\nModel {i_model}/{n_model}\n=======')

    batch_size, n_epoch, learn_rate, loss_function, optimize_method = params
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'n_epoch: {n_epoch}')
    logger.info(f'learn_rate: {learn_rate}')
    logger.info(f'loss_function: {loss_function}')
    logger.info(f'optimize_method: {optimize_method}')

    # Re-initialize per-fold metrics
    train_loss = np.zeros(n_fold)
    fold_loss = np.zeros(n_fold)

    # Loop through folds and evaluate model for each
    for i_fold, (train_idx, validate_idx) in enumerate(
            kfold.split(X=X_remain, y=y_remain)):
    
        # Extract training and cross-validation data
        X_train = X_remain[train_idx, :]
        y_train = y_remain[train_idx]
    
        X_xvalidate = X_remain[validate_idx, :]
        y_xvalidate = y_remain[validate_idx]
    
        # Define model
        layer_list = [
            keras.Input(shape=(n_feature,), name='Input'),
            layers.Dense(10, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(200, activation='relu'),
            layers.Dense(400, activation='relu'),
            layers.Dense(200, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(1, name='output', activation=None),
        ]
        # layer_list = [
        #     keras.Input(shape=(n_feature,), name='Input'),
        #     layers.Dense(10, activation='relu'),
        #     layers.Dense(50, activation='relu'),
        #     layers.Dense(100, activation='relu'),
        #     layers.Dense(50, activation='relu'),
        #     layers.Dense(10, activation='relu'),
        #     layers.Dense(1, name='output', activation=None),
        # ]
        Model = keras.Sequential(layers=layer_list)
    
        # Compile model
        if optimize_method == 'adam':
            optimizer = optimizers.Adam(learning_rate=learn_rate)
        else:
            raise ValueError('Unknown optimize_method')
    
        if loss_function == 'mse':
            loss = losses.MeanSquaredError()
        else:
            raise ValueError('Unknown loss_function')
    
        Model.compile(
            optimizer=optimizer,  # Create optimizer object and pass in with learning
            loss=loss,
            metrics=None,  # TODO-Store More
            loss_weights=None,
            sample_weight_mode=None,
            weighted_metrics=None,
            target_tensors=None,
            distribute=None
        )

        # Fit data to model
        history = Model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=n_epoch,
            verbose=verbosity
        )
    
        # Generate generalization metrics. TODO-add more metrics
        train_loss[i_fold] = Model.evaluate(
            x=X_train,
            y=y_train,
            verbose=verbosity
        )
        fold_loss[i_fold] = Model.evaluate(
            x=X_xvalidate,
            y=y_xvalidate,
            verbose=verbosity
        )
        
        logger.info(f'\nFold {i_fold}')
        logger.info(f'Training loss: {train_loss[i_fold]}')
        logger.info(f'Cross-validation loss: {fold_loss[i_fold]}')

    mean_train_loss = train_loss.mean()
    mean_fold_loss = fold_loss.mean()
    logger.info(f'Mean Training loss: {mean_train_loss}')
    logger.info(f'Mean Cross-validation loss: {mean_fold_loss}')
 
    # Hang onto best-performing model
    if mean_fold_loss < best_validation_loss:
        best_validation_loss = mean_fold_loss
        best_params = params
        BestModel = Model
        best_idx = i_model

# %% Predict and View if Trend over Time Matches Data
lat, lon = 20.0, 50.0
t_start = datetime.datetime(2003, 1, 1).timestamp()
t_stop = datetime.datetime(2012, 1, 1).timestamp()
sim_time = np.linspace(t_start, t_stop, num=128)

seconds_per_year = 60 * 60 * 24 * 365
sim_years = 1970 + sim_time / seconds_per_year

X_sim = features.preprocess(
    lat=lat * np.ones(sim_time.shape),
    lon=lon * np.ones(sim_time.shape),
    epoch_time=sim_time
)
y_sim = BestModel.predict(X_sim)

plt.figure(1, clear=True)
plt.plot(sim_years, y_sim)
plt.grid()
plt.xlabel('Year')
plt.ylabel('{} Concentration'.format(output_type))
plt.title('Temporal {} Trend for Lat: {:.2f}, Lon: {:.2f}'.format(
    output_type.upper(), lat, lon))

# %% Predict Spatial Map for Single Time Instant
lats = np.linspace(-90.0, 90.0, num=128)
lons = np.linspace(-180.0, 180.0, num=128)

lats, lons = np.meshgrid(lats, lons, indexing='ij')
lats = lats.flatten()
lons = lons.flatten()

date = datetime.datetime(2005, 1, 1)
t = date.timestamp()

X_sim = features.preprocess(
    lat=lats,
    lon=lons,
    epoch_time=t * np.ones(lats.shape)
)
y_sim = BestModel.predict(X_sim)

plt.figure(2, clear=True)
plt.scatter(lons, lats, c=y_sim)
plt.grid()
plt.ylabel('Latitude (Degrees)')
plt.xlabel('Longitude (Degrees)')
plt.title('Spatial Map on {}'.format(date))
plt.colorbar()

# %% Plot Loss vs. Epoch
loss = np.array([15696.4100,
4771.2448,
7.5060,
6.9503,
6.8240,
6.7647,
6.7335,
6.7125,
6.6915,
6.6797,
6.6686,
6.6575,
6.6517,
6.6450,
6.6391,
6.6361,
6.7443,
6.7068,
])
epoch=  np.arange(len(loss))

plt.figure(20, clear=True)
plt.bar(epoch, loss)
plt.grid()
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')

