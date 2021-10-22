"""
Script for linear model fitting and evaluation
"""
# %% Imports
# File
from pathlib import Path
import pickle

# Computing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
output_type = 'co2'  # co2, ch4

# data_dir = Path('/mnt/c/Users/ktopo/Desktop/ECE537/data')
data_dir = Path('C:/Users/ktopo/Desktop/ECE537/data')

if output_type == 'co2':
    csv_name = data_dir / Path('co2_data.csv')
elif output_type == 'ch4':
    csv_name = data_dir / Path('ch4_data.csv')
else:
    raise ValueError('Invalid model type')

df = pd.read_csv(csv_name, nrows=n_row)

# %% Features
days_since_03 = df['day_since_03'].to_numpy()
months_since_03 = np.floor(days_since_03 / 30)

lat = df['lat'].to_numpy()
lon = df['lon'].to_numpy()
ecf = features.lla_to_ecf(lat=lat, lon=lon)
norm_ecf = ecf / features.EARTH_RADIUS

X = np.concatenate(
    (norm_ecf, months_since_03[:, np.newaxis]),
    axis=-1
)

y = df[output_type].to_numpy()

# TODO-KT: Split training and testing
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# Delete data no longer used
del X, y

# %% Fit Model
n_feature = X_train.shape[1]
modeltype = 'linear'

if modeltype == 'linear':
    Model = LinearRegression()
    fit_kwargs = {}

elif modeltype == 'nn':
    layer_list = [
        keras.Input(shape=(n_feature,), name='Input'),
        layers.Dense(10, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(5, activation='relu'),
        layers.Dense(1, name='output', activation=None),
    ]
    Model = keras.Sequential(layers=layer_list)
    
    # Compile model
    LEARN_RATE = 0.001
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
        'epochs': 10,
        'batch_size': 128
    }

else:
    raise ValueError('Unknown modeltype {}'.format(modeltype))

# %% Wrap model
WrapModel = model.WrapperModel()
WrapModel.init_from_model(Model=Model)
WrapModel.fit(X_train, y_train, **fit_kwargs)

# %% Show Metrics
y_predict = WrapModel.predict(X_test)
mse = mean_squared_error(y_pred=y_predict, y_true=y_test)
print('MSE: {}'.format(mse))

# %% Save to file
model_file = data_dir / 'newmodel.model'
WrapModel.save(model_file)

# %% Predict for large grid
# Configure grid
n_year = 10
months = np.arange(0, 12*n_year)
n_month = months.size

n_lat, n_lon = 100, 90
lats = np.linspace(-70.0, 70.0, num=n_lat)
lons = np.linspace(-180.0, 180.0, num=n_lon)

lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

# Transform/Pre-Process
ecf = features.lla_to_ecf(
    lat=lat_grid.flatten(),
    lon=lon_grid.flatten()
)

ecf = ecf.reshape((n_lat, n_lon, 3))  # back to grid
norm_ecf = ecf / features.EARTH_RADIUS

carbon = np.zeros((n_month, n_lat, n_lon))
for i_month in range(n_month):
    month = months[i_month] * np.ones((n_lat, n_lon, 1))
    x = np.concatenate(
        (norm_ecf, month),
        axis=-1
    )
    carbon[i_month, :, :] = WrapModel.predict(x)

# %% Predict Worldwide CO2
vmin = carbon.min()
vmax = carbon.max()

for i_month, month in enumerate(months):
    plt.figure(2, clear=True)
    # plt.contour(lat_grid, lon_grid, carbon[i_month, :, :], levels=10)
    
    # TODO-KT Is the data transposed? Should carbon increase right to left or
    # other way
    plt.imshow(carbon[i_month, :, :],
              extent=[lons.min(), lons.max(), lats.min(), lats.max()],
              cmap='jet',
              vmin=vmin, vmax=vmax,
              aspect='auto')
    plt.title('CO2')
    plt.xlabel('Longitude (Degrees)')
    plt.ylabel('Latitude (Degrees)')
    plt.xlim([lons.min(), lons.max()])
    plt.ylim([lats.min(), lats.max()])
    plt.grid()
    plt.title('Months Since 2003: {}'.format(month))
    #plt.colorbar()
    plt.pause(0.1)

# %%
mean_concentration = np.mean(carbon, axis=0)
vmax, vmin = mean_concentration.max(), mean_concentration.min()

plt.figure(5, clear=True)
plt.imshow(mean_concentration, vmin=vmin, vmax=vmax,
            extent=[lons.min(), lons.max(), lats.max(), lats.min()])
plt.title('CO2')
plt.xlabel('Longitude (Degrees)')
plt.ylabel('Latitude (Degrees)')
plt.xlim([lons.min(), lons.max()])
plt.ylim([lats.min(), lats.max()])
plt.grid()
plt.title('Mean CO2 by Location over 10 Years')
plt.colorbar()
