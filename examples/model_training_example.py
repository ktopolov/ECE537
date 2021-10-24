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

lat = df['lat'].to_numpy()
lon = df['lon'].to_numpy()
days_since_03 = df['day_since_03'].to_numpy()
carbon = df[output_type].to_numpy()

# Synthetic Data: This data intentionally has a trend. Using this
# Should show whether the model is able to fit to something
# carbon = 0.2 * lat + 0.5 * lon + 0.01 * days_since_03

# %% Features
X = features.preprocess(
    lat=df['lat'].to_numpy(),
    lon=df['lon'].to_numpy(),
    days_since_03=df['day_since_03'].to_numpy()
)

y = carbon

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
    model_name = 'linear_model.model'
    fit_kwargs = {}

elif modeltype == 'nn':
    layer_list = [
        keras.Input(shape=(n_feature,), name='Input'),
        layers.Dense(20, activation='relu'),
        layers.Dense(30, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(30, activation='relu'),
        layers.Dense(20, activation='relu'),
        layers.Dense(1, name='output', activation=None),
    ]
    Model = keras.Sequential(layers=layer_list)
    
    # Compile model
    LEARN_RATE = 0.01
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
        'epochs': 2,
        'batch_size': 64
    }
    model_name = 'tf_model'

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
WrapModel.save(model_name)

# %% Show Mean CO2 from Dataset
monthly_carbon, month_bins, lat_bins, lon_bins = features.calc_monthly_average(
    lat=lat,
    lon=lon,
    days_since_03=days_since_03,
    carbon=carbon,
    lat_lims=(-50.0, 40.0, 2.0),
    lon_lims=(-130.0, 150.0, 2.0),
    month_lims=(0, 120, 6),
)

# %% Predict for Same Grid
n_month = month_bins.size
n_lat = lat_bins.size
n_lon = lon_bins.size

lat_grid, lon_grid = np.meshgrid(lat_bins, lon_bins, indexing='ij')

predict_carbon = np.zeros((n_month, n_lat, n_lon))
for i_month in range(n_month):
    day_since_03 = 30 * month_bins[i_month] * np.ones(n_lat * n_lon)

    x = features.preprocess(
        lat=lat_grid.reshape((n_lat * n_lon)),
        lon=lon_grid.reshape((n_lat * n_lon)),
        days_since_03=day_since_03
    )

    predict_carbon[i_month, :, :] = \
        WrapModel.predict(x).reshape((n_lat, n_lon))

# %% Show Prediction vs. Measurement
vmin, vmax = np.percentile(predict_carbon.flatten(), [5, 95]) 

datas = {
    'From Data': monthly_carbon,
    'Predicted': predict_carbon
}
lon_min, lon_max = lon_bins.min(), lon_bins.max()
lat_min, lat_max = lat_bins.min(), lat_bins.max()

for i_month, month in enumerate(month_bins):
    plt.figure(2, clear=True)
    
    for ii, (label, data) in enumerate(datas.items()):
        plt.subplot(2, 1, ii+1)

        # TODO-KT Is the data transposed? Should carbon increase right to left or
        # other way
        # plt.contour(lat_grid, lon_grid, carbon[i_month, :, :], levels=10)
        plt.imshow(
            data[i_month, :, :],
            extent=[lon_min, lon_max, lat_min, lat_max],
            cmap='jet',
            vmin=vmin, vmax=vmax,
            aspect='auto',
            interpolation='bilinear',
       )
        plt.title('CO2')
        plt.xlabel('Longitude (Degrees)')
        plt.ylabel('Latitude (Degrees)')
        plt.xlim([lon_min, lon_max])
        plt.ylim([lat_min, lat_max])
        plt.grid()
        plt.title('Mean Carbon ({}) - Months Since 2003: {}'.format(
            label, month))
        plt.colorbar()

    plt.pause(0.1)

# %%
predict_mean = predict_carbon.mean(axis=0)
data_mean = monthly_carbon.mean(axis=0)

datas = {
    'From Data': data_mean,
    'Predicted': predict_mean
}

plt.figure(5, clear=True)

for ii, (label, data) in enumerate(datas.items()):
    plt.subplot(1, 2, ii+1)

    plt.imshow(
        data,
        extent=[lon_min, lon_max, lat_min, lat_max],
        aspect='auto'
    )
    plt.title('CO2')
    plt.xlabel('Longitude (Degrees)')
    plt.ylabel('Latitude (Degrees)')
    plt.xlim([lon_min, lon_max])
    plt.ylim([lat_min, lat_max])
    plt.grid()
    plt.title('Mean CO2 {} over duration'.format(label))
    plt.colorbar()
