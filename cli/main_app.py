# -*- coding: utf-8 -*-
"""
Application to leverage trained model for metric reporting
"""
from codebase import model, features, kml
import pandas as pd
import datetime
import logging
from pathlib import Path
import sys
import argparse
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # shut tensorflow up
warnings.filterwarnings('ignore', category=UserWarning)  # ignore sklearn


# %%

def _define_cli_args():
    """Create parser to accept input arguments

    Returns
    -------
    parser : argparse.ArgumentParser
        Instantiated parser object
    """
    # Required
    parser = argparse.ArgumentParser(
        description='Use predictive model for optimal site planning')
    parser.add_argument('--mode', dest='mode', default='region', type=str,
                        help='Run mode', choices=['region', 'location'])
    parser.add_argument('--start-date', dest='start_date', nargs=3, type=int,
                        help='Simulation start date in [DD MM YYYY] format', required=True)
    parser.add_argument('--stop-date', dest='stop_date', nargs=3, type=int,
                        help='Simulation stop date in [DD MM YYYY] format', required=True)
    parser.add_argument(
        '--sim-step', dest='sim_step', type=str, default='monthly',
        help='Step size of simulation',
        choices=['daily', 'weekly', 'monthly', 'quarterly', 'anually']
    )

    # Required if --mode region
    parser.add_argument('--lat-bounds', dest='lat_bounds', nargs=2, type=float,
                        help='Latitude min max in degrees')
    parser.add_argument('--lon-bounds', dest='lon_bounds', nargs=2, type=float,
                        help='Longitude min max in degrees')
    parser.add_argument('--lat-res', dest='lat_res', type=float, default=1.0,
                        help='Latitude resolution in degrees')
    parser.add_argument('--lon-res', dest='lon_res', type=float, default=1.0,
                        help='Longitude resolution in degrees')

    # Required if --mode location
    parser.add_argument('--locs', dest='locs', type=float, nargs='+',
                        help='Lat/lon locations stored as \'lat0 lon0 lat1 lon1 ...\'')

    # Optional
    parser.add_argument('--out-dir', dest='out_dir', type=str,
                        help='Output file directory; defaults to \'./output\'', default='./output')
    return parser


def _setup_timeline(start_date, stop_date, sim_step, logger=None):
    """Setup simulation timeline

    Parameters
    ----------
    start_date : [] (3) int
        Simulation start date in [YYYY MM DD] format

    stop_date : [] (3)
        Simulation stop date in [YYYY MM DD] format

    sim_step : str
        Simulation step size; options are 'daily', 'monthly', 'quarterly', 'annually'

    Returns
    -------
    days_since_03 : (n_month) np.ndarray float
        Sampled days since 2003 for simulation
    """
    start_datetime = datetime.datetime(*start_date)
    start_epoch_time = start_datetime.timestamp()
    stop_datetime = datetime.datetime(*stop_date)
    stop_epoch_time = stop_datetime.timestamp()

    if start_epoch_time > stop_epoch_time:
        raise ValueError('Stop time must be prior to start time {}')
        return

    if sim_step == 'daily':
        step_seconds = 86400.0  # seconds in a day
    elif sim_step == 'weekly':
        step_seconds = 604800.0
    elif sim_step == 'monthly':
        step_seconds = 2628000.0
    elif sim_step == 'quarterly':
        step_seconds = 7884000.0
    elif sim_step == 'annually':
        step_seconds = 31540000.0
    else:
        raise ValueError('Unknown sim_step {}'.format(sim_step))
        return

    sim_times = np.arange(start_epoch_time, stop_epoch_time, step=step_seconds)

    if logger is not None:
        logger.info('Start Date: {}'.format(start_datetime))
        logger.info('Stop Date: {}'.format(stop_datetime))
        logger.info('Simulation Duration: {} Months\n'.format(
            stop_datetime - start_datetime))

    return sim_times


def _setup_locs(locs, logger=None):
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


def _setup_spatial_support(
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

# %%


def main():
    """Main app"""
    # %% Setup
    # TODO-Throw exceptions within subfunctions and figure out how
    # to try/except properly

    # CLI Parsing
    parser = _define_cli_args()
    args = parser.parse_args()

    # File I/O
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # make directory if not exists

    # Logging
    FORMAT = '[%(levelname)-s]: %(asctime)s \t %(message)s'
    log_path = out_dir / 'status.log'
    print('Logging to {}'.format(log_path))
    logging.basicConfig(
        filename=args.log_file,  # if None, defaults to stdout
        stream=sys.stdout,
        level=logging.DEBUG,
        format=FORMAT
    )
    logger = logging.getLogger('LOG_BUDDY')
    logger.info('Output Directory: {}\n'.format(out_dir))

    # CLI call
    sep = ' '
    argv = sep.join(['python'] + sys.argv)
    logger.info('=== CLI ===')
    logger.info('{}\n'.format(argv))

    # Mode
    if args.mode not in ['location', 'region']:
        logger.error('Unknown --mode {}'.format(args.mode))
        return
    logger.info('Mode: {}\n'.format(args.mode))

    # Timeline - sample points throughout the duration of the simulation
    logger.info('=== TIMELINE ===')
    try:
        sim_times = _setup_timeline(
            start_date=args.start_date,
            stop_date=args.stop_date,
            sim_step=args.sim_step,
            logger=logger
        )
    except ValueError as err:
        logger.error(err)
        return

    n_time = sim_times.size

    # %% Setup Predictive Model
    logger.info('=== MODEL ===')

    is_windows = os.name == 'nt'
    basepath = Path('C:/Users/ktopo/Desktop/ECE537') if is_windows \
        else Path('/mnt/c/Users/ktopo/Desktop/ECE537')  # Linux (WSL)

    model_type = 'tf'  # 'tf' or 'sklearn'
    if model_type == 'tf':
        model_path = basepath / 'tf_model'
    elif model_type == 'sklearn':
        model_path = basepath / 'linear_model.model'
    else:
        logger.error('Unsupported model_type {}'.format(model_type))
        return

    Model = model.WrapperModel()

    Model.init_from_file(
        model_type=model_type,
        path=model_path
    )
    logger.info('Model loaded from {}\n'.format(model_path))

    # %% Setup KML Writer
    kml_file = out_dir / 'predictions.kml'
    Kml = kml.KmlWriter(kml_file=kml_file)

    # %% Enter Mode-Specific Execution
    logger.info('=== SPATIAL SETUP ===')

    # Compute grid of (day, lat, lon) pairs for prediction
    if args.mode == 'location':
        # Store locations as lat/lon
        try:
            lats, lons = _setup_locs(locs=args.locs, logger=logger)
        except ValueError as err:
            logger.error(err)
            return

        # (n_day, n_location)
        n_location = lats.size
        lat_grid = np.repeat(lats[np.newaxis, :], axis=0, repeats=n_time)
        lon_grid = np.repeat(lons[np.newaxis, :], axis=0, repeats=n_time)
        time_grid = np.repeat(
            sim_times[:, np.newaxis], axis=-1, repeats=n_location)

    else:  # region mode
        lats, lons = _setup_spatial_support(
            lat_bounds=args.lat_bounds,
            lon_bounds=args.lon_bounds,
            lat_res=args.lat_res,
            lon_res=args.lon_res,
            logger=logger
        )

        # Grid all inputs
        time_grid, lat_grid, lon_grid = np.meshgrid(
            sim_times,
            lats,
            lons,
            indexing='ij'
        )

    grid_shape = time_grid.shape
    logger.info(
        'Prediction Size: (n_time, n_lat, n_lon) = {}'.format(grid_shape))
    logger.info('Total Predictions: {}\n'.format(np.prod(grid_shape)))

    # %% Prediction
    logger.info('=== PREDICTION ===')

    x = features.preprocess(
        lat=lat_grid,
        lon=lon_grid,
        epoch_time=time_grid,
    )

    logger.info('Predicting Carbon\n')
    predict_carbon_grid = Model.predict(x)
    mean_carbon = predict_carbon_grid.mean(axis=0)  # mean across the days

    # %% Write outputs
    logger.info('=== OUTPUTS ===')

    # CSV - Write out predictions for each location and day
    data_dict = {
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten(),
        'time': time_grid.flatten(),
        'predict_carbon': predict_carbon_grid.flatten(),
    }
    csv_path = out_dir / 'predictions.csv'
    pd.DataFrame(data_dict).to_csv(csv_path, index=None)
    logger.info('{} written'.format(csv_path))

    # KML
    if args.mode == 'location':
        lats = lat_grid[0, :]  # lats are repeated for every day
        lons = lon_grid[0, :]  # lons are repeated for every day

        # Normalize predictions from [0, 1]
        vmin = mean_carbon.min()
        vmax = mean_carbon.max()
        mean_carbon_norm = (mean_carbon - vmin) / (vmax - vmin)

        # Map to color values (0, 255)
        colormap = cm.get_cmap('jet')
        norm_colors = colormap(mean_carbon_norm)
        print(norm_colors.shape)
        norm_colors *= 255
        norm_colors = norm_colors.astype(int)

        for i_loc in range(lats.size):
            lat = lats[i_loc]
            lon = lons[i_loc]
            description = 'Lat: {} - Lon: {} - Mean Predicted Carbon: {}'.format(
                lat, lon, mean_carbon[i_loc])

            Kml.add_point(
                lat=lat,
                lon=lon,
                name='Site {}'.format(i_loc),
                rgba=norm_colors[i_loc, :],
                description=description,
            )

    else:
        # Lats/lons repeat themselves for each timestamp
        lat_grid = lat_grid[0, :, :]
        lon_grid = lon_grid[0, :, :]

        contour_path = out_dir / 'contour.jpeg'
        Kml.add_ground_overlay(
            max_lat=lat_grid.max(),
            min_lat=lat_grid.min(),
            max_lon=lon_grid.max(),
            min_lon=lon_grid.min(),
            img_path=contour_path.name,
            name='Average Carbon Overlay',
            description='Prediction of mean carbon atmospheric content over simulation'
        )

        # Must flip because imshow flips vertical image
        vmin = mean_carbon.min()
        vmax = mean_carbon.max()
        colormap = cm.get_cmap('jet')
        mean_carbon_norm = (mean_carbon - vmin) / (vmax - vmin)
        mean_carbon_rgb = colormap(mean_carbon_norm)
        plt.imsave(
            contour_path.absolute(),
            np.flip(mean_carbon_rgb, axis=0),
        )
        logger.info('{} written'.format(contour_path))

        Kml.add_contours(
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            values=mean_carbon,
            levels=30
        )

    Kml.write()
    logger.info('{} written'.format(kml_file))


# %%
if __name__ == '__main__':
    main()
