# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:21:25 2021

@author: ktopo
"""
import simplekml
import matplotlib.pyplot as plt
import numpy as np

# %%
a = 1.0
b = 0.2
c = 5

lat = np.linspace(-30.0, -20.0, num=100)
lon = np.linspace(80.0, 90.0, num=80)
n_lat = lat.size
n_lon = lon.size
heatmap = np.zeros((n_lat, n_lon))

lat_idx = np.arange(n_lat) - n_lat/2
lon_idx = np.arange(n_lon) - n_lon/2
lat_idx, lon_idx = np.meshgrid(lat_idx, lon_idx)

heatmap = (a*lat_idx**2 + b*lat_idx + c) + 2*(a*lon_idx**2 + b*lon_idx + c)

levels = np.arange(5001, step=1000)
plt.figure(1, clear=True)
contours = plt.contourf(lat_idx, lon_idx, heatmap, levels=levels)
# plt.contour(lat_idx, lon_idx, heatmap, levels=levels)
plt.xlabel('Latitude')
plt.ylabel('Longitude')
plt.title('Expected CO2 Concentration')
plt.grid()
plt.colorbar()

# %%
p = contours.collections[0].get_paths()[0]
v = p.vertices
x = v[:,0]
y = v[:,1]

# %%
# All pairs are lon/lat
kml = simplekml.Kml()

polygon = kml.newpolygon(
    extrude=None,
    tessellate=None,
    altitudemode=None,
    gxaltitudemode=None,
    outerboundaryis=[(-83, 42), (-83, 44), (-85, 44), (-85, 42)],
    innerboundaryis=(),
    name='Polygon'
)
polygon.style.polystyle.color = simplekml.Color.green

point = kml.newpoint(name="Point", coords=[(-84, 43)])
point.style.linestyle.width = 3
point.style.linestyle.color = simplekml.Color.red

path = kml.newlinestring(
    name='Path',
    coords=[(-83, 42), (-83, 44), (-85, 43), (-83, 42)]
)
path.style.linestyle.width = 5
path.style.linestyle.color = simplekml.Color.blue

kml.save('myfile.kml')
