# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:21:25 2021

@author: ktopo
"""
import numpy as np

from codebase import kml
from codebase import model
from codebase import features

# %% Confugure Sampled Grid
n_lat, n_lon = 150, 100
lat = np.linspace(-20.0, 20.0, num=n_lat)
lon = np.linspace(-20.0, 20.0, num=n_lon)
lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

# %% Predict Carbon Heatmap
is_load = True

if is_load:
    # Load model for carbon prediction
    WrapModel = model.WrapperModel()

    modelpath = 'C:/Users/ktopo/Desktop/ECE537/data/newmodel.model'
    WrapModel.init_from_file(file=modelpath, model_type='sklearn')

    # Transform/Pre-Process
    ecf = features.lla_to_ecf(
        lat=lat_grid.flatten(),
        lon=lon_grid.flatten()
    )
    
    ecf = ecf.reshape((n_lat, n_lon, 3))  # back to grid
    norm_ecf = ecf / features.EARTH_RADIUS

    month = np.zeros((n_lat, n_lon, 1))
    x = np.concatenate(
        (norm_ecf, month),
        axis=-1
    )
    heatmap = WrapModel.predict(x)

else:
    # Make up garbage data
    a, b, c = 3.0, 0.2, 5.0
    heatmap = a*(lat_grid)**2 + b*lat_grid + c \
        + 2*(a*lon_grid**2 + b*lon_grid + c)

# %%
Writer = kml.KmlWriter(kml_file='myfile.kml')
Writer.add_point(
        lat=lat.mean(),
        lon=lon.mean(),
        rgba=(255, 255, 0, 255),
        name='Center of ROI',
        description='Center of the selected region of interest'
    )

Writer.add_contours(
    lat_grid,
    lon_grid,
    values=heatmap,
    levels=30
)

# 1) Figure out ground overlay as kmz
# 2) Figure out contour playback video in kml

Writer.write()


