# -*- coding: utf-8 -*-
"""
Pre-processing of features
"""
import numpy as np
import pyproj

# %% Constants
EARTH_RADIUS = 6.3781e6

# %% Extract and Pre-Process Features
def lla_to_ecf(lat, lon, alt=None):
    """Convert geodedic to ECF

    Parameters
    ----------
    lat : (N) np.ndarray float
        Latitude (degrees)

    lon : (N) np.ndarray float
        Longitude (degrees)

    alt : (N) np.ndarray float
        Altitude (m). If None, defaults to zero

    Returns
    -------
    ecf : (N, 3) np.ndarray float
        ECF position vectors
    """
    if alt is None:
        alt = np.zeros(lon.shape)

    LlaToEcef = pyproj.Transformer.from_crs(
        {"proj":'latlong', "ellps":'WGS84', "datum":'WGS84'},
        {"proj":'geocent', "ellps":'WGS84', "datum":'WGS84'},
        )

    xyz = LlaToEcef.transform(
        xx=lon,
        yy=lat,
        zz=alt,
        radians=False
    )
    ecf = np.stack(xyz, axis=0).T
    return ecf
