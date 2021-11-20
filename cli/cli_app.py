# -*- coding: utf-8 -*-
"""
Application to leverage trained model for metric reporting
"""
import pandas as pd
import datetime
import logging
from pathlib import Path
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # shut tensorflow up
warnings.filterwarnings('ignore', category=UserWarning)  # ignore sklearn

import app_support as app
from codebase import model, features, kml

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
    kml_dir = out_dir / 'kml_output'
    kml_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'predictions.csv'

    # Logging
    FORMAT = '[%(levelname)-s]: %(asctime)s \t %(message)s'
    log_path = out_dir / 'status.log'
    logging.basicConfig(
        filename=log_path,  # if None, defaults to stdout
        level=logging.DEBUG,
        format=FORMAT
    )
    logger = logging.getLogger('LOG_BUDDY')

    # Log info
    print('Log Path: {}'.format(log_path))  # this needs to go to stdout
    logger.info('Output Directory: {}'.format(out_dir))
    logger.info('CSV Predictions Path: {}'.format(csv_path))
    logger.info('KML Directory: {}'.format(kml_dir))
    argv = ' '.join(['python'] + sys.argv)
    logger.info('Command: {}'.format(argv))

    # Load model
    Model = model.WrapperModel()
    model_dir = '/home/ktopolov/ECE537/output/TFModel'
    Model.init_from_file(model_type='tf', path=model_dir)
    logger.info('Model loaded from {}'.format(model_dir))

    # Mode
    if args.mode not in ['location', 'region']:
        logger.error('Unknown --mode {}'.format(args.mode))
        return
    logger.info('Mode: {}\n'.format(args.mode))

    # Timeline - sample points throughout the duration of the simulation
    try:
        sim_times = app.setup_timeline(
            start_date=args.start_date,
            stop_date=args.stop_date,
            sim_step=args.sim_step,
            logger=logger
        )
    except ValueError as err:
        logger.error(err)
        return

    n_time = sim_times.size

    # %% Mode-specific execution
    # Compute grid of (day, lat, lon) pairs for prediction
    if args.mode == 'location':
        # Store locations as lat/lon
        try:
            lats, lons = app.setup_locs(locs=args.locs, logger=logger)
        except ValueError as err:
            logger.error(err)
            return

        # (n_day, n_loc)
        n_loc = lats.size
        lat_grid = np.repeat(lats[np.newaxis, :], axis=0, repeats=n_time)
        lon_grid = np.repeat(lons[np.newaxis, :], axis=0, repeats=n_time)
        time_grid = np.repeat(sim_times[:, np.newaxis], axis=-1, repeats=n_loc)

    else:  # region mode
        lats, lons = app.setup_spatial_support(
            lat_bounds=args.lat_bounds,
            lon_bounds=args.lon_bounds,
            lat_res=args.lat_res,
            lon_res=args.lon_res,
            logger=logger
        )

        # Grid all inputs
        time_grid, lat_grid, lon_grid = np.meshgrid(
            sim_times, lats, lons, indexing='ij')

    grid_shape = time_grid.shape
    logger.info('Predicting... (n_time, n_lat, n_lon) = {}'.format(grid_shape))
    logger.info('Total Predictions: {}'.format(np.prod(grid_shape)))
    x = features.preprocess(lat=lat_grid, lon=lon_grid, epoch_time=time_grid)
    predict_carbon_grid = Model.predict(x)
    mean_carbon = predict_carbon_grid.mean(axis=0)  # mean across the days

    # %% Write outputs
    logger.info('Writing CSV output')
    data_dict = {
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten(),
        'time': time_grid.flatten(),
        'predict_carbon': predict_carbon_grid.flatten(),
    }
    pd.DataFrame(data_dict).to_csv(csv_path, index=None)
    logger.info('{} written'.format(csv_path))

    # KML
    if args.mode == 'location':
        app.write_kml_location(
            lats=lat_grid[0, :],  # lats are repeated for every day
            lons=lon_grid[0, :],  # lons are repeated for every day
            carbon=mean_carbon,
            kml_dir=kml_dir,
            logger=logger,
        )
    else:
        app.write_kml_region(
            lat_grid=lat_grid[0, :, :],
            lon_grid=lon_grid[0, :, :],
            carbon=mean_carbon,
            out_dir=kml_dir,
            logger=logger
        )

    logger.info('Completed')


# %%
if __name__ == '__main__':
    main()

def simulate_location(sim_times, lats, lons, Model, logger=None):
    """Wrapper for simulating location-based mode

    Parameters
    ----------
    sim_times : (N)
        Simulation epoch times (s)

    lats : (M)
        Latitudes of locations (deg)

    lons : (M)
        Longitudes of locations (deg)

    Model : model.WrapperModel
        Model for carbon prediction

    logger : logging.Logger
        Logger

    Returns
    -------
    carbon : (N, M)
        Carbon predictions per-time, per location
    """
    # Timeline - sample points throughout the duration of the simulation
    try:
        sim_times = app.setup_timeline(
            start_date=args.start_date,
            stop_date=args.stop_date,
            sim_step=args.sim_step,
            logger=logger
        )
    except ValueError as err:
        logger.error(err)
        return

    n_time = sim_times.size

    # %% Mode-specific execution
    # Compute grid of (day, lat, lon) pairs for prediction
    if args.mode == 'location':
        # Store locations as lat/lon
        try:
            lats, lons = app.setup_locs(locs=args.locs, logger=logger)
        except ValueError as err:
            logger.error(err)
            return

        # (n_day, n_loc)
        n_loc = lats.size
        lat_grid = np.repeat(lats[np.newaxis, :], axis=0, repeats=n_time)
        lon_grid = np.repeat(lons[np.newaxis, :], axis=0, repeats=n_time)
        time_grid = np.repeat(sim_times[:, np.newaxis], axis=-1, repeats=n_loc)

    else:  # region mode
        lats, lons = app.setup_spatial_support(
            lat_bounds=args.lat_bounds,
            lon_bounds=args.lon_bounds,
            lat_res=args.lat_res,
            lon_res=args.lon_res,
            logger=logger
        )

        # Grid all inputs
        time_grid, lat_grid, lon_grid = np.meshgrid(
            sim_times, lats, lons, indexing='ij')

    grid_shape = time_grid.shape
    logger.info('Predicting... (n_time, n_lat, n_lon) = {}'.format(grid_shape))
    logger.info('Total Predictions: {}'.format(np.prod(grid_shape)))
    x = features.preprocess(lat=lat_grid, lon=lon_grid, epoch_time=time_grid)
    predict_carbon_grid = Model.predict(x)
    mean_carbon = predict_carbon_grid.mean(axis=0)  # mean across the days

    # %% Write outputs
    logger.info('Writing CSV output')
    data_dict = {
        'lat': lat_grid.flatten(),
        'lon': lon_grid.flatten(),
        'time': time_grid.flatten(),
        'predict_carbon': predict_carbon_grid.flatten(),
    }
    pd.DataFrame(data_dict).to_csv(csv_path, index=None)
    logger.info('{} written'.format(csv_path))

    # KML
    if args.mode == 'location':
        app.write_kml_location(
            lats=lat_grid[0, :],  # lats are repeated for every day
            lons=lon_grid[0, :],  # lons are repeated for every day
            carbon=mean_carbon,
            kml_dir=kml_dir,
            logger=logger,
        )
    else:
        app.write_kml_region(
            lat_grid=lat_grid[0, :, :],
            lon_grid=lon_grid[0, :, :],
            carbon=mean_carbon,
            out_dir=kml_dir,
            logger=logger
        )

    logger.info('Completed')
