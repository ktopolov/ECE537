# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:21:25 2021

@author: ktopo
"""
import matplotlib.pyplot as plt
import numpy as np

from codebase import kml

# %% Define Sample CO2 Heatmap (To Be Replaced)
a, b, c = 3.0, 0.2, 5.0

lat_center, lon_center = 0.0, 0.0
lat_extent, lon_extent = 10.0, 10.0

n_lat, n_lon = 100, 50

lat = np.linspace(lat_center - lat_extent/2, lat_center + lat_extent/2, num=n_lat)
lon = np.linspace(lon_center - lon_extent/2, lon_center + lon_extent/2, num=n_lon)

lat_grid, lon_grid = np.meshgrid(lat, lon)

lat_grid_offset = lat_grid - lat_grid.mean()
lon_grid_offset = lon_grid - lon_grid.mean()
heatmap = a*(lat_grid_offset)**2 + b*lat_grid_offset + c \
    + 2*(a*lon_grid_offset**2 + b*lon_grid_offset + c)

# %% Contours
plt.figure(1, clear=True)
contours =  plt.contour(lat_grid, lon_grid, heatmap, levels=10)

# %%
Writer = kml.KmlWriter(kml_file='myfile.kml')

Writer.add_point(
    lat=lat_center,
    lon=lon_center,
    rgba=(255, 255, 0, 255),
    name='Center of ROI',
    description='Center of the selected region of interest'
)

for collection in contours.collections[-1:0:-1]:
    paths = collection.get_paths()

    color = (255 * collection.get_edgecolor()).astype(int).flatten()

    for path in paths:
        lat_points = path.vertices[:, 0]
        lon_points = path.vertices[:, 1]

        Writer.add_path(
            lats=lat_points,
            lons=lon_points,
            rgba=color
        )

Writer.write()

