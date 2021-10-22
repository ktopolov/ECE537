"""
Script for model training and evaluation
"""
# Visit https://catalogue.ceda.ac.uk/uuid/294b4075ddbc4464bb06742816813bdc
# Click "Download"
# Select the tarball
# ESA Greenhouse Gases Climate Change Initiative (GHG_cci): Column-averaged methane from Sentinel-5P, generated with the WFM-DOAS algorithm, version 1.2
# Number of files: 3113
# %% Imports
# File
from pathlib import Path

# Computing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Model based
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from sklearn.linear_model import LinearRegression

# Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Internal
from codebase import features

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

    # Define model structure
    
else:
    raise ValueError('Invalid model type')

# %% Save to file

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
