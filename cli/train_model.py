"""
Script for linear model fitting and evaluation
"""
# %% Imports
# File
from pathlib import Path
import logging
import os
import sys

# Computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import itertools

# Model based
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import KFold

# Metrics
from sklearn.model_selection import train_test_split

# Internal
from codebase import features

# %% Visualization


def _plot_time_prediction(Model, data_type, fignum=None):
    """Plot model prediction at a single location across time

    Parameters
    ----------
    Model : keras.Sequential
        Trained model

    data_type : str
        Type of data; either 'co2' or 'ch4'

    fignum : int/str
        Figure number; if None, opens new figure
    """
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
    y_sim = Model.predict(X_sim)

    plt.figure(fignum, clear=True)
    plt.plot(sim_years, y_sim)
    plt.grid()
    plt.xlabel('Year')
    plt.ylabel('{} Concentration'.format(data_type))
    plt.title('Temporal {} Trend for Lat: {:.2f}, Lon: {:.2f}'.format(
        data_type.upper(), lat, lon))
    plt.draw()
    plt.show(block=False)


def _plot_spatial_map(Model, data_type, fignum='Spatial Map'):
    """Plot model prediction at a single location across time

    Parameters
    ----------
    Model : keras.Sequential
        Trained model

    data_type : str
        Type of data; either 'co2' or 'ch4'

    fignum : int/str
        Figure number; if None, opens new figure
    """
    # Setup lat/lon grid
    lats = np.linspace(-90.0, 90.0, num=128)
    lons = np.linspace(-180.0, 180.0, num=128)

    lats, lons = np.meshgrid(lats, lons, indexing='ij')
    lats = lats.flatten()
    lons = lons.flatten()

    # Arbitrary date
    date = datetime.datetime(2005, 1, 1)
    t = date.timestamp()

    X_sim = features.preprocess(
        lat=lats,
        lon=lons,
        epoch_time=t * np.ones(lats.shape)
    )
    y_sim = Model.predict(X_sim)

    plt.figure(2, clear=True)
    plt.scatter(lons, lats, c=y_sim)
    plt.grid()
    plt.ylabel('Latitude (Degrees)')
    plt.xlabel('Longitude (Degrees)')
    plt.title('Spatial Map on {}'.format(date))
    plt.colorbar()
    plt.draw()
    plt.show(block=False)

# %% Model-Based


def _get_model(
    n_feature,
    hparams,
):
    """Return compiled Keras model for a given set of hyperparameters

    Parameters
    ----------
    n_feature : int
        Number of features per input example

    hparams : {}
        Hyperparameters; must include:
            'loss': str
                Options are 'mse'
            'optimizer': str
                Options are 'adam', 'sgd'
            'learning_rate': float
                Learning rate

    X : (N, F) float
        Dataset with N examples each with F features

    y : (N) float
        Labels

    Returns
    -------
    Model : keras.Sequential
        Keras compiled model
    """
    # TODO-KT: Find automated way to generate these sizes
    hidden_sizes = [10, 50, 100, 200, 100, 50, 10]

    layer_list = [
        keras.Input(shape=(n_feature,), name='Input')
    ]

    for ii, size in enumerate(hidden_sizes):
        layer_list.append(
            layers.Dense(size, activation='relu', name=f'DenseLayer{ii}')
        )

    layer_list.append(
        layers.Dense(1, name='output', activation=None)
    )

    Model = keras.Sequential(layers=layer_list)

    # Compile model
    if hparams['optimizer'] == 'adam':
        optimizer = optimizers.Adam(learning_rate=hparams['learning_rate'])
    elif hparams['optimizer'] == 'sgd':
        optimizer = optimizers.SGD(learning_rate=hparams['learning_rate'])
    else:
        raise ValueError('Unknown optimize_method')

    if hparams['loss'] == 'mse':
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
    return Model


def split_data(X, y, test_percent=0.2, cross_percent=0.2):
    """Split data into training, crossvalidation and testing sets

    Parameters
    ----------
    X : (N, F) float
        Dataset with N examples each with F features

    y : (N) float
        Labels

    test_percent : float
        Percentage of data to use for testing (as a decimal)

    cross_percent : float
        Percentage of data to use for crossvalidation (as a decimal)

    Returns
    -------
    X_train : (n_train, F) float
        Training dataset

    X_cross : (n_cross, F) float
        Crossvalidation dataset

    X_test : (n_test, F) float
        Testing dataset

    y_train : (n_train) float
        Training labels

    y_cross : (n_cross) float
        Crossvalidation labels

    y_test : (n_test) float
        Testing labels
    """
    X_remain, X_test, y_remain, y_test = train_test_split(
        X, y, test_size=test_percent)

    remain_percent = 1.0 - test_percent
    X_train, X_cross, y_train, y_cross = train_test_split(
        X_remain, y_remain, test_size=cross_percent)

    return X_train, X_cross, X_test, y_train, y_cross, y_test


# %% I/O
def _get_logger(log_file):
    """Construct and return a logger

    Parameters
    ----------
    log_file : str
        Path to desired log file output. If None, outputs to STDOUT

    Returns
    -------
    logger : logging.Logger
        Logger
    """
    fmt = '[%(levelname)-s]: %(asctime)s \t %(message)s'
    log_config_args = {'level': logging.DEBUG, 'format': fmt}
    if log_file is not None:
        print('Logging to {}'.format(log_file))

        with open(log_file, 'w'):  # using this allows clearing of file
            logging.basicConfig(
                filename=log_file,  # if None, defaults to stdout
                filemode='w',  # doesnt seem to work
                **log_config_args
            )
    else:
        logging.basicConfig(
            filename=None,  # if None, defaults to stdout
            **log_config_args
        )
    logger = logging.getLogger('LOG_BUDDY')
    return logger


def _load_data(data_type):
    """Load data from CSV for training

    Parameters
    ----------
    data_type : str
        Either 'xco2' or 'xch4' based on what you want to train

    Returns
    -------
    X : (N, F) np.ndarray float
        Input data, N examples with F features each

    y : (N) np.ndarray float
        Output data labels
    """
    n_row = None  # number of rows to read from CSV; None reads all

    data_dir = Path('/mnt/c/Users/ktopo/Desktop/ECE537/data')  # WSL Linux
    # data_dir = Path('C:/Users/ktopo/Desktop/ECE537/data')  # Windows

    csv_file = 'co2_data.csv' if data_type == 'xco2' else 'ch4_data.csv'
    csv_name = data_dir / csv_file
    df = pd.read_csv(csv_name, nrows=n_row)

    lat = df['latitude'].to_numpy()
    lon = df['longitude'].to_numpy()
    epoch_time = df['time'].to_numpy()
    carbon = df[data_type].to_numpy()
    X = features.preprocess(lat=lat, lon=lon, epoch_time=epoch_time)
    y = carbon
    return X, y

# %%


def main():
    """Main training application"""
    # -- Configuration
    output_dir = Path('./output')
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'training.log'
    tb_log_dir = output_dir / 'tb_logs'
    data_type = 'xco2'  # xco2, xch4
    verbosity = 1

    # -- Setup logger
    logger = _get_logger(log_file)

    # -- Read Data
    X, y = _load_data(data_type)
    n_feature = X.shape[-1]

    # Split test data from the rest. The rest will be used for K-fold x-validation
    X_train, X_cross, X_test, y_train, y_cross, y_test = split_data(
        X=X, y=y, test_percent=0.55, cross_percent=0.25)
    del X, y  # remove data we no longer need

    # -- Setup Model
    # Hyperparameters
    HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([64]))
    HP_N_EPOCH = hp.HParam('n_epoch', hp.Discrete([3]))
    HP_LEARNING_RATE = hp.HParam(
        'learning_rate', hp.Discrete([1e-5]))
    HP_LOSS = hp.HParam('loss', hp.Discrete(['mse']))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))
    hyperparameters = [
        HP_BATCH_SIZE.domain.values,
        HP_N_EPOCH.domain.values,
        HP_LEARNING_RATE.domain.values,
        HP_LOSS.domain.values,
        HP_OPTIMIZER.domain.values
    ]
    hyperparameters = list(itertools.product(*hyperparameters))
    n_model = len(hyperparameters)
    logger.info(f'Total Number of models: {n_model}')

    for i_model, params in enumerate(hyperparameters):
        logger.info(f'\n=======\nModel {i_model}/{n_model}\n=======')

        hparams = {
            'batch_size': params[0],
            'n_epoch': params[1],
            'learning_rate': params[2],
            'loss': params[3],
            'optimizer': params[4],
        }

        # Log hyperparameters
        for key, val in hparams.items():
            logger.info(f'{key}: {val}')

        # Extract training and cross-validation data
        # # Define model
        Model = _get_model(n_feature=n_feature, hparams=hparams)

        # Fit data to model
        log_dir = str(tb_log_dir / f'model{i_model}')
        tb_callback = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        )
        hp_callback = hp.KerasCallback(log_dir, hparams)
        history = Model.fit(
            x=X_train,
            y=y_train,
            batch_size=hparams['batch_size'],
            epochs=hparams['n_epoch'],
            verbose=verbosity,
            validation_data=(X_cross, y_cross),
            callbacks=[
                tb_callback,  # log metrics
                hp_callback,  # log hparams
            ],
        )

    # # %% Save a Model
    model_path = output_dir / 'TFModel'
    keras.models.save_model(
        model=Model,
        filepath=model_path
    )

    # logger.info(f'Best model saved to {model_path}')

    # # Predict and View if Trend over Time Matches Data
    # _plot_time_prediction(
    #     Model=BestModel,
    #     data_type=data_type,
    #     fignum='Time Prediction'
    # )

    # # Plot spatial map for a single time instant
    # _plot_spatial_map(
    #     Model=BestModel,
    #     data_type=data_type,
    #     fignum='Spatial Map'
    # )

    logger.info('Application completed')


if __name__ == '__main__':
    main()
