# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from codebase import model, features

# %%
co2_path = Path('/home/ktopolov/ECE537_data/co2_data.csv')
df = pd.read_csv(co2_path, nrows=None)

# %%

lat_step = 2.
lon_step = 2.5

lats = np.arange(-60.0, 70.0, step=lat_step)
lons = np.arange(-180.0, 180.0, step=lon_step)
n_lat = lats.size
n_lon = lons.size

# %%
quant_lat_idx = np.round((df.latitude - lats.min()) / lat_step).astype(int)
quant_lon_idx = np.round((df.longitude - lons.min()) / lon_step).astype(int)

avg_carbon = np.nan * np.zeros((n_lat, n_lon))
for i_lat in range(n_lat):
    print(f'i_lat: {i_lat}/{n_lat}', end='\r')
    for i_lon in range(n_lon):
        idx = (quant_lat_idx == i_lat) & (quant_lon_idx == i_lon)
        avg_carbon[i_lat, i_lon] = np.mean(df.xco2[idx])

# %%
id_nan = np.isnan(avg_carbon)
vmin, vmax = np.percentile(a=avg_carbon[~id_nan], q=[10., 90.])

plt.figure(1, clear=True)
plt.imshow(
    avg_carbon,
    aspect='auto',
    extent=[lons.min(), lons.max(), lats.min(), lats.max()],
    cmap='jet',
    interpolation='bilinear',  # 'none', 'bilinear'
    origin='lower',  # show negative latitude/longitude in bottom left
    vmin=vmin,
    vmax=vmax
)
plt.xlabel('Longitude (Deg)')
plt.ylabel('Latitude (Deg)')
plt.grid()
plt.draw()
plt.colorbar()
plt.title('Average CO2 Concentration (ppm) By Location in Dataset')
plt.show(block=False)

# %% Load Model
Model = model.WrapperModel()

model_path = '/home/ktopolov/ECE537/output/TFModel_complex'
Model.init_from_file(model_type='tf', path=model_path)

n_times = 10
times = np.linspace(df.time.min(), df.time.max(), num=n_times)

lat_grid, lon_grid, t_grid = np.meshgrid(lats, lons, times, indexing='ij')
x = features.preprocess(lat=lat_grid, lon=lon_grid, epoch_time=t_grid)

predict_carbon = Model.predict(x)
predict_carbon[id_nan, :] = np.nan  # Make NaN same as in data for easier comparison

# %% Plot
plt.figure(2, clear=True)
plt.imshow(
    predict_carbon.mean(axis=-1),
    aspect='auto',
    extent=[lons.min(), lons.max(), lats.min(), lats.max()],
    cmap='jet',
    interpolation='bilinear',
    origin='lower',  # show negative latitude/longitude in bottom left
    vmin=vmin,
    vmax=vmax
)
plt.xlabel('Longitude (Deg)')
plt.ylabel('Latitude (Deg)')
plt.grid()
plt.draw()
plt.colorbar()
plt.title('Average CO2 Concentration (ppm) By Location - Model')
plt.show(block=False)
