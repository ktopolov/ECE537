"""
Script for linear model fitting and evaluation
"""
# %% Imports
# File
from pathlib import Path

# Computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Model based
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.linear_model import LinearRegression

# Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Internal
from codebase import features
from codebase import model

# %% Read from CSV
n_row = None  # number of rows to read from CSV
output_type = 'xco2'  # xco2, xch4

# data_dir = Path('/mnt/c/Users/ktopo/Desktop/ECE537/data')
data_dir = Path('C:/Users/ktopo/Desktop/ECE537/data')

if output_type == 'xco2':
    csv_name = data_dir / Path('co2_data.csv')
elif output_type == 'xch4':
    csv_name = data_dir / Path('ch4_data.csv')
else:
    raise ValueError('Invalid model type')

df = pd.read_csv(csv_name, nrows=n_row)

lat = df['latitude'].to_numpy()
lon = df['longitude'].to_numpy()
epoch_time = df['time'].to_numpy()

carbon = df[output_type].to_numpy()

# Synthetic Data: This data intentionally has a trend. Using this
# Should show whether the model is able to fit to something
# carbon = 0.2 * lat + 0.5 * lon + 0.01 * days_since_03

# %% Features
X = features.preprocess(
    lat=lat,
    lon=lon,
    epoch_time=epoch_time
)

y = carbon

# TODO-KT: Split training and testing
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# Delete data no longer used
del X, y

# %% Fit Model
n_feature = X_train.shape[1]
modeltype = 'nn'

if modeltype == 'linear':
    Model = LinearRegression()
    model_name = 'linear_model.model'
    fit_kwargs = {}

elif modeltype == 'nn':
    layer_list = [
        keras.Input(shape=(n_feature,), name='Input'),
        layers.Dense(10, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(30, activation='relu'),
        layers.Dense(40, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(40, activation='relu'),
        layers.Dense(30, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(1, name='output', activation=None),
    ]
    Model = keras.Sequential(layers=layer_list)

    # Compile model
    LEARN_RATE = 0.005
    optimizer = optimizers.Adam(learning_rate=LEARN_RATE)
    Model.compile(
        optimizer=optimizer,  # Create optimizer object and pass in with learning
        loss=keras.losses.MSE,
        metrics=['mse'],
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        distribute=None
    )
    fit_kwargs = {
        'epochs': 3,
        'batch_size': 64
    }
    model_name = 'tf_model'

else:
    raise ValueError('Unknown modeltype {}'.format(modeltype))

# %% Wrap model
WrapModel = model.WrapperModel()
WrapModel.init_from_model(Model=Model)
WrapModel.fit(X_train, y_train, **fit_kwargs)

# %% Show Predictions
train_prediction = WrapModel.predict(X_train)

mse = mean_squared_error(y_pred=train_prediction, y_true=y_train)
print('MSE on Training Data: {}'.format(mse))

n_skip = 3000

plt.figure(1, clear=True)
plt.plot(train_prediction[::n_skip], label='Predict')
plt.plot(y_train[::n_skip], label='Truth')
plt.xlabel('Row')
plt.ylabel('Carbon Amount')
plt.grid()
plt.title('Comparing Predictions on Training Data to Truth')
plt.legend()

# %% Show Metrics
y_predict = WrapModel.predict(X_test)
mse = mean_squared_error(y_pred=y_predict, y_true=y_test)
print('MSE on Testing Data: {}'.format(mse))

n_skip = 3000

plt.figure(2, clear=True)
plt.plot(y_predict[::n_skip], label='Predict')
plt.plot(y_test[::n_skip], label='Truth')
plt.xlabel('Row')
plt.ylabel('Carbon Amount')
plt.grid()
plt.title('Comparing Predictions on Testing Data to Truth')
plt.legend()

# %% Save to file
WrapModel.save(model_name)

# %% Predict Mean Carbon over Duration
# -- CONFIGURE
# Timeline
start_date = datetime.datetime(2003, 1, 1)
stop_date = datetime.datetime(2012, 1, 1)
sim_step_size_months = 0.5  # months step size

# Coordinates
min_lat = -70.0
max_lat = 70.0
lat_step = 1.0
min_lon = -150.0
max_lon = 150.0
lon_step = 1.0

# -- DERIVE
# Timeline
start_time = start_date.timestamp()
end_time = stop_date.timestamp()
sim_step_size_sec = sim_step_size_months * 30 * 24 * 3600
sim_times = np.arange(start_time, end_time, sim_step_size_sec)
n_time = sim_times.size

# Coordinates
lat = np.arange(min_lat, max_lat, step=lat_step)
lon = np.arange(min_lon, max_lon, step=lon_step)
n_lat = lat.size
n_lon = lon.size

# %% Predict at Each Timestep
time_grid, lat_grid, lon_grid = np.meshgrid(
    sim_times,
    lat,
    lon,
    indexing='ij'
)

X = features.preprocess(
    lat=lat_grid,
    lon=lon_grid,
    epoch_time=time_grid
)

carbon_pred = np.zeros((n_time, n_lat, n_lon))

predict_grid = WrapModel.predict(X)

# %% Show Data at Each Time
vmin = predict_grid.min()
vmax = predict_grid.max()
levels = np.percentile(predict_grid.flatten(), np.arange(0, 100, 5))

for i_time, ep_time in enumerate(sim_times):
    date_time = datetime.datetime.fromtimestamp(ep_time)  

    plt.figure(10, clear=True)
    plt.imshow(
        predict_grid[i_time, :, :],
        extent=[min_lon, max_lon, max_lat, min_lat],
        cmap='jet',
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
    )
    plt.colorbar()

    contour = plt.contour(
        lon_grid[0, ...],
        lat_grid[0, ...],
        predict_grid[i_time, :, :],
        extent=[min_lon, max_lon, max_lat, min_lat],
        cmap='jet_r',
        vmin=vmin,
        vmax=vmax,
        # aspect='auto',
        levels=levels,
        # interpolation='bilinear',
    )
    # plt.clabel(contour, fmt='%2.2d', colors='k', fontsize=10)

    plt.title('CO2')
    plt.xlabel('Longitude (Degrees)')
    plt.ylabel('Latitude (Degrees)')
    plt.grid()
    plt.title('Carbon Map Prediction on {}'.format(date_time))
    # plt.colorbar()

    plt.pause(0.1)

# %% Show Mean
mean_prediction = np.mean(predict_grid, axis=0)
vmin = mean_prediction.min()
vmax = mean_prediction.max()

plt.figure(11, clear=True)
plt.imshow(
    mean_prediction,
    extent=[min_lon, max_lon, max_lat, min_lat],
    cmap='jet',
    vmin=vmin,
    vmax=vmax,
    aspect='auto',
    interpolation='bilinear',
)
plt.title('CO2')
plt.xlabel('Longitude (Degrees)')
plt.ylabel('Latitude (Degrees)')
plt.grid()
plt.title('Mean Carbon Map Prediction from {} to {}'.format(
    start_date, stop_date))
plt.colorbar()
