"""Support functions for CLI and UI Application"""
import numpy as np
import datetime
import logging
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

from codebase import kml


def setup_timeline(start_date, stop_date, sim_step, logger=None):
    """Setup simulation timeline

    Parameters
    ----------
    start_date : [] (3) int
        Simulation start date in [YYYY MM DD] format

    stop_date : [] (3)
        Simulation stop date in [YYYY MM DD] format

    sim_step : str
        Simulation step size; options are 'daily', 'monthly', 'quarterly',
        'annually'

    logger : logging.Logger
        Logger

    Returns
    -------
    sim_times : (n_step) np.ndarray float
        Epoch times for simulation
    """
    start_datetime = datetime.datetime(*start_date)
    start_epoch_time = start_datetime.timestamp()
    stop_datetime = datetime.datetime(*stop_date)
    stop_epoch_time = stop_datetime.timestamp()

    if start_epoch_time > stop_epoch_time:
        raise ValueError('Start time must be prior to stop time {}')
        return

    sim_seconds = {
        'daily': 86400.0,
        'weekly': 604800.0,
        'monthly': 2628000.0,
        'quarterly': 7884000.0,
        'annually': 31540000.0,
    }
    if sim_step not in sim_seconds.keys():
        raise ValueError('Unknown sim_step {}'.format(sim_step))
    else:
        step_seconds = sim_seconds[sim_step]

    sim_times = np.arange(start_epoch_time, stop_epoch_time, step=step_seconds)

    if logger is not None:
        logger.info('Start Date: {}'.format(start_datetime))
        logger.info('Stop Date: {}'.format(stop_datetime))
        logger.info('Simulation Duration: {}\n'.format(
            stop_datetime - start_datetime))

    return sim_times


def setup_locs(locs, logger=None):
    """Convert list of lat, lon locations to arrays

    Parameters
    ----------
    locs : []
        List of [lat0, lon0, lat1, lon1, ...] coordinates

    logger : logging.logger
        Logger object

    Exceptions
    ----------
    ValueError

    Returns
    -------
    lats : (N) np.ndarray float
        Latitudes

    lons : (N) np.ndarray float
        Longitudes
    """
    n_loc = len(locs)
    is_odd = (n_loc % 2) != 0
    if is_odd:
        logger.error('--locs needs even # input (lat0 lon0 lat1 lon1 ...)')
        raise ValueError('Odd Number of locs')

    # Parse into lats and lons
    n_pair = n_loc//2
    lats = np.zeros(n_pair)
    lons = np.zeros(n_pair)
    for ii in range(n_pair):
        idx = 2*ii
        lats[ii] = locs[idx]
        lons[ii] = locs[idx+1]

        if logger is not None:
            logging.debug('(lat{}, lon{}): ({}, {})'.format(
                ii, ii, lats[ii], lons[ii]))

    return lats, lons


def setup_spatial_support(
    lat_bounds,
    lon_bounds,
    lat_res,
    lon_res,
    logger=None
):
    """Setup vector of latitudes and longitudes

    Parameters
    ----------
    lat_bounds : (2) float
        Latitude min, max

    lon_bounds : (2) float
        Longitude min, max

    lat_res : float
        Latitude spacing

    lon_res : float
        Longitude spacing

    logger : logging.logger
        Logger

    Returns
    -------
    lats : (N) np.ndarray float
        Latitudes

    lons : (M) np.ndarray float
        Longitudes
    """
    lat_min, lat_max = min(lat_bounds), max(lat_bounds)
    lon_min, lon_max = min(lon_bounds), max(lon_bounds)

    lats = np.arange(lat_min, lat_max, step=lat_res)
    lons = np.arange(lon_min, lon_max, step=lon_res)

    if logger is not None:
        logger.debug('Latitude [Min, Max, Res]: [{}, {}, {}]'.format(
            lat_min, lat_max, lat_res))
        logger.debug('Longitude [Min, Max, Res]: [{}, {}, {}]'.format(
            lon_min, lon_max, lon_res))

    return lats, lons


def write_kml_region(
    lat_grid,
    lon_grid,
    carbon,
    kml_dir='./kml_output',
    logger=None
):
    """Write KML file for region-based simulation other necessary output

    Parameters
    ----------
    lat_grid : (n_lat, n_lon)
        Latitude grid

    lon_grid : (n_lat, n_lon)
        Longitude grid

    carbon : (n_lat, n_lon)
        Carbon concentration at each point

    out_dir : str
        Output directory. If does not exist, will be created

    logger : logging.Logger
        If provided, logs info
    """
    # Setup directories
    kml_dir = Path(kml_dir)
    kml_file = kml_dir / 'region.kml'
    contour_path = kml_dir / 'region_contour.jpeg'

    # Write image with carbon concentration
    # Must flip because imshow flips vertical image
    vmin, vmax = carbon.min(), carbon.max()
    carbon_norm = (carbon - vmin) / (vmax - vmin)
    colormap = cm.get_cmap('jet')
    mean_carbon_rgb = colormap(carbon_norm)
    plt.imsave(
        contour_path.absolute(),
        np.flip(mean_carbon_rgb, axis=0),
    )

    if logger is not None:
        logger.info('{} written'.format(contour_path))

    # Write KML file
    Kml = kml.KmlWriter(kml_file=kml_file)
    Kml.add_ground_overlay(
        max_lat=lat_grid.max(),
        min_lat=lat_grid.min(),
        max_lon=lon_grid.max(),
        min_lon=lon_grid.min(),
        img_path=contour_path.name,
        name='Average Carbon Overlay',
        description='Prediction of mean carbon atmospheric content over simulation'
    )
    Kml.add_contours(
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        values=carbon,
        levels=30
    )
    Kml.write()

    if logger is not None:
        logger.info('{} written'.format(kml_file))


def write_kml_location(
    lats,
    lons,
    carbon,
    kml_dir='./kml_output',
    logger=None
):
    """Write KML for location-based simulation

    Parameters
    ----------
    lats : (N,)
        Latitudes

    lons : (N,)
        Longitudes

    carbon : (N,)
        Carbon amount

    kml_dir : str
        KML output directory

    logger : logging.Logger
        If provided, logs information
    """
    kml_dir = Path(kml_dir)
    kml_file = kml_dir / 'location.kml'

    # Normalize predictions from [0, 1]
    vmin, vmax = carbon.min(), carbon.max()
    mean_carbon_norm = (carbon - vmin) / (vmax - vmin)

    # Map to color values (0, 255)
    colormap = cm.get_cmap('jet')
    norm_colors = colormap(mean_carbon_norm)
    norm_colors *= 255
    norm_colors = norm_colors.astype(int)

    Kml = kml.KmlWriter(kml_file=kml_file)
    for i_loc in range(lats.size):
        lat = lats[i_loc]
        lon = lons[i_loc]

        # KML description per-point should have several metrics
        description = 'Lat: {} - Lon: {} - Mean Predicted Carbon: {}'.format(
            lat, lon, carbon[i_loc])

        Kml.add_point(
            lat=lat,
            lon=lon,
            name='Site {}'.format(i_loc),
            rgba=norm_colors[i_loc, :],
            description=description,
        )
    Kml.write()

    if logger is not None:
        logger.info('{} written'.format(kml_file))
