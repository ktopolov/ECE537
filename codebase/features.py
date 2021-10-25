# -*- coding: utf-8 -*-
"""
Pre-processing of features
"""
import numpy as np
import pyproj
import datetime

# %% Constants
EARTH_RADIUS = 6.3781e6

# %% Extract and Pre-Process Features
def month_year_to_days_since_03(month, year):
    """Convert month and year to the number of days since 2003

    Parameters
    ----------
    month : int
        Month

    year : int
        Year

    Returns
    -------
    days_since_03 : int
        Number of days since January 1st, 2003
    """
    months_since_03 = 12 * (year - 2003) + (month - 1)
    days_since_03 = 30 * months_since_03
    return days_since_03

def preprocess(lat, lon, epoch_time):
    """Pre-process specific inputs to a standard form for input to the models

    Parameters
    ----------
    lat : (N) np.ndarray float
        Latitude (degrees)

    lon : (N) np.ndarray float
        Longitude (degrees)

    epoch_time : (N) np.ndarray int
        Epoch time (s) for measurement

    Returns
    -------
    x : (N, 4)
        Pre-processed features
    """
    # ecf = lla_to_ecf(lat=lat, lon=lon)
    # norm_ecf = ecf / EARTH_RADIUS
    min_epoch_time = datetime.datetime(2003, 1, 1).timestamp()
    max_epoch_time = datetime.datetime(2012, 1, 1).timestamp()

    x = np.stack(
        (
            lat/90.0,
            lon/180.0,
            (epoch_time - min_epoch_time) / (max_epoch_time - min_epoch_time),
        ),
        axis=-1
    )
    # x = np.concatenate(
    #     (norm_ecf, months_since_03[:, np.newaxis] / (12*10)),
    #     axis=-1
    # )
    return x

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

def calc_monthly_average(
    lat,
    lon,
    days_since_03,
    carbon,
    lat_lims=(-50.0, 50.0, 5.0),
    lon_lims=(-180.0, 170.0, 10.0),
    month_lims=(0, 120, 1),
):
    """Compute monthly average concentration by binning data per
    location and month

    Parameters
    ----------
    lat : (N) np.ndarray float
        Measurement latitudes (deg)

    lon : (N) np.ndarray float
        Measurement longitudes (deg)

    days_since_03 : (N) np.ndarray int
        Days since 2003

    carbon : (N) np.ndarray float
        Carbon measurements

    lat_lims : (3) np.ndarray float
        Latitude minimum, maximum, and step size

    lon_lims : (3) np.ndarray float
        Longitude minimum, maximum, and step size

    month_lims : (3) np.ndarray float
        Month minimum, maximum, and step size

    Returns
    -------
    mean_carbon : (n_month, n_lat, n_lon) np.ndarray float
        Average carbon

    month_bins : (n_month) np.ndarray float
        Months for grid

    lat_bins : (n_lat) np.ndarray float
        Latitudes for grid

    lon_bins : (n_lon) np.ndarray float
        Longitudes for grid
    """
    months = days_since_03 / 30

    # Define quantization bins
    min_month, max_month, month_step = month_lims
    month_bins = np.arange(min_month, max_month, month_step)
    n_month_bin = month_bins.size

    min_lat, max_lat, lat_step = lat_lims
    lat_bins = np.arange(min_lat, max_lat, lat_step)
    n_lat_bin = lat_bins.size

    min_lon, max_lon, lon_step = lon_lims
    lon_bins = np.arange(min_lon, max_lon, lon_step)
    n_lon_bin = lon_bins.size

    # Extract only data lying within desired quantization interval
    id_valid = (months <= max_month) & (months >= min_month) \
        & (lat <= max_lat) & (lat >= min_lat) \
            & (lon <= max_lon) & (lon >= min_lon)
    valid_lats = lat[id_valid]
    valid_lons = lon[id_valid]
    valid_months = months[id_valid]
    valid_carbons = carbon[id_valid]

    # Quantize remaining valid data
    month_idx, _ = np.divmod(valid_months - min_month, month_step)
    lat_idx, _ = np.divmod(valid_lats - min_lat, lat_step)
    lon_idx, _ = np.divmod(valid_lons - min_lon, lon_step)

    # Store in grid
    mean_carbon = np.zeros((n_month_bin, n_lat_bin, n_lon_bin))

    # Quantize data to nearest bin
    for ii in range(n_month_bin):
        print('Computing for Month {}/{}'.format(ii, n_month_bin))
        id_month = month_idx == ii  # data point belonging to this month
        for jj in range(n_lat_bin):
            id_lat = lat_idx == jj  # data belonging to lat bin
            for kk in range(n_lon_bin):
                id_lon = lon_idx == kk  # data belonging to lon bin

                id_in_bin = np.logical_and(
                    id_month,
                    np.logical_and(id_lat, id_lon)
                )
                
                # If no samples found (over an ocean) ignore
                if np.any(id_in_bin):
                    mean_carbon[ii, jj, kk] += valid_carbons[id_in_bin].mean()

    return mean_carbon, month_bins, lat_bins, lon_bins
