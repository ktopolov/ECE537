"""
Script for model training and evaluation
"""
# Visit https://catalogue.ceda.ac.uk/uuid/294b4075ddbc4464bb06742816813bdc
# Click "Download"
# Select the tarball
# ESA Greenhouse Gases Climate Change Initiative (GHG_cci): Column-averaged methane from Sentinel-5P, generated with the WFM-DOAS algorithm, version 1.2
# Number of files: 3113

# %% Imports
# Computing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Model based
from tensorflow import keras
from tensorflow.keras import layers, optimizers, regularizers
from sklearn.linear_model import LinearRegression

# Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Internal
import dataio

# %% Read from CSV
n_row = None  # number of rows to read from CSV

output_type = 'co2'  # co2, ch4

if output_type == 'co2':
    csv_name = 'co2_data.csv'
    df = pd.read_csv(csv_name, nrows=n_row)
    y = df['co2'].to_numpy()
elif output_type == 'ch4':
    csv_name = 'ch4_data.csv'
    df = pd.read_csv(csv_name, nrows=n_row)
    y = df['ch4'].to_numpy()
else:
    raise ValueError('Invalid model type')

# %% Features
n_days_since_03 = df['day_since_03'].to_numpy()
lat = df['lat'].to_numpy()
lon = df['lon'].to_numpy()

X = dataio.preprocess_features(
    n_days_since_03=n_days_since_03,
    lat=lat,
    lon=lon,
    )

# TODO-KT: Split training and testing
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# Delete data no longer used
del X, y

# %% Fit Model
n_feat = X_train.shape[1]

model_type = 'linear'  # linear, nn

if model_type == 'linear':
    Model = LinearRegression()
    Model.fit(X=X_train, y=y_train)
elif model_type == 'nn':

    # Define model structure
    layer_list = [
        keras.Input(shape=(n_feat,), name='Input'),
        layers.Dense(10, activation='relu'),
        layers.Dense(50, activation='relu'),
        layers.Dense(10, activation='relu'),
        layers.Dense(5, activation='relu'),
        layers.Dense(1, name='output', activation=None),
    ]
    Model = keras.Sequential(layers=layer_list)
    
    # Compile model
    LEARN_RATE = 0.001
    BATCH_SIZE = 256
    N_EPOCH = 5
    optimizer = optimizers.Adam(learning_rate=LEARN_RATE)
    Model.compile(
        optimizer='adam',  # Create optimizer object and pass in with learning
        loss=keras.losses.MSE,
        metrics=['mse'],
        loss_weights=None,
        sample_weight_mode=None,
        weighted_metrics=None,
        target_tensors=None,
        distribute=None
    )

    Model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCH,
        verbose=1,
    )
else:
    raise ValueError('Invalid model type')

# %% Train Model without need to Load All Data
y_predict = Model.predict(X_test, batch_size=64)
mse = mean_squared_error(y_pred=y_predict, y_true=y_test)

print('MSE: {}'.format(mse))

# %% Predict Worldwide CO2
n_tot_days = 365 * 10  # data exists for about 10 years
days_since_03 = np.arange(start=0, stop=n_tot_days, step=60)

# Data is only well-represented from -55 to +70 deg latitude
n_lon, n_lat = 128, 128
lats = np.linspace(-80.0, 80.0, num=n_lat)
lons = np.linspace(-180.0, 180.0, num=n_lon)

lons, lats = np.meshgrid(lons, lats)
lats = lats.flatten()
lons = lons.flatten()

mean_concentration = np.zeros((n_lat, n_lon))
N = days_since_03.size

for ii, day in enumerate(days_since_03):
    days_since = day * np.ones(lats.size)
    X_extrap = dataio.preprocess_features(
        n_days_since_03=days_since,
        lat=lats,
        lon=lons
    )

    predict_co2 = Model.predict(X_extrap)
    mean_concentration += predict_co2.reshape((n_lat, n_lon)) / N
    predict_co2 = predict_co2.reshape((n_lat, n_lon))

    # Plot
    vmin, vmax = 384, 386
    plt.figure(2, clear=True)
    plt.imshow(predict_co2,
              extent=[lons.min(), lons.max(), lats.min(), lats.max()],
              cmap='jet',
              vmin=vmin, vmax=vmax,
              aspect='auto')
    #plt.scatter(lons, lats, c=predict_co2, vmin=vmin, vmax=vmax)
    plt.title('CO2')
    plt.xlabel('Longitude (Degrees)')
    plt.ylabel('Latitude (Degrees)')
    plt.xlim([lons.min(), lons.max()])
    plt.ylim([lats.min(), lats.max()])
    plt.grid()
    plt.title('Days Since 2003: {}'.format(day))
    plt.colorbar()
    plt.pause(0.1)

# %%
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
