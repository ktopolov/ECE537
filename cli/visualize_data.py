"""
Script for model training and evaluation
"""
# Visit https://catalogue.ceda.ac.uk/uuid/294b4075ddbc4464bb06742816813bdc
# Click "Download"
# Select the tarball
# ESA Greenhouse Gases Climate Change Initiative (GHG_cci): Column-averaged methane from Sentinel-5P, generated with the WFM-DOAS algorithm, version 1.2
# Number of files: 3113

# %%
# Path related
import argparse

# Computing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from scipy import signal

# %% Main
def main():
    """Visualization of dataset"""
    parser = argparse.ArgumentParser(description='Write CO2 and CH4 NETCDF files to .csv')
    parser.add_argument('--co2-csv', dest='co2_path', default=None, type=str, help='CO2 .csv file from write_data_to_csv.py')
    parser.add_argument('--ch4-csv', dest='ch4_path', default=None, type=str, help='CH4 .csv file from write_data_to_csv.py')
    parser.add_argument('--rows', dest='rows', default=None, type=int, help='Number of rows of data to load')
    args = parser.parse_args()

    # %% Read from CSV
    if args.co2_path is not None:
        print('Reading CO2 Data...')
        df = pd.read_csv(args.co2_path, nrows=args.rows)

        lat = df['latitude'].to_numpy()
        lon = df['longitude'].to_numpy()
        
        # %% Show Where Data Samples Are
        print('Plotting CO2 Data...')
        plt.figure(0, clear=True)
        plt.hist2d(
            x=lon,
            y=lat,
            bins=256,
            density=False,
            cmin=1,
            cmax=60000
        )
        plt.xlabel('Longitude (Deg)')
        plt.ylabel('Latitude (Deg)')
        plt.title('# CO2 Records by Location\n{} Samples Total'.format(
            len(df)))
        plt.colorbar()
        plt.grid()
        
        # %% Show Number of Samples per Year
        epoch_time = df['time'].to_numpy()
        ref_epoch_time = datetime.datetime(2003, 1, 1).timestamp()
        epoch_time_03 = epoch_time - ref_epoch_time
        days_since_03 = epoch_time_03 / (60*60*24)
        years_since_03 = days_since_03 / 365.25
        
        plt.figure(1, clear=True)
        plt.hist(years_since_03, bins=10, edgecolor='k')
        plt.xlabel('Years Since 2003')
        plt.ylabel('Number of CO2 Data Samples By Year')
        plt.title('CO2 Samples Per Year')
        plt.grid()
        
        # %% Show Average Levels by Month
        n_moving_avg = 100
        mov_avg_kernel = np.ones(n_moving_avg) / n_moving_avg

        # Sort by timestamp to get global time average
        df = df.sort_values(by=['time'])
        co2_mov_avg = signal.lfilter(b=mov_avg_kernel, a=1, x=df['xco2'])
        n_skip = 5000  # skip every few measurements so plot doesn't lock up

        epoch_time = df['time'].to_numpy()
        ref_epoch_time = datetime.datetime(2003, 1, 1).timestamp()
        epoch_time_03 = epoch_time - ref_epoch_time
        days_since_03 = epoch_time_03 / (60*60*24)
        years_since_03 = days_since_03 / 365.25

        plt.figure(2, clear=True)
        plt.plot(
            years_since_03[n_moving_avg::n_skip],
            co2_mov_avg[n_moving_avg::n_skip]
        )
        plt.xlabel('Years Since 2003')
        plt.ylabel('ppm')
        plt.grid()
        plt.title('{} Sample Global Moving Average CO2 Concentration in ppm'.format(n_moving_avg))
    
    
    # %% Read from CSV
    if args.ch4_path is not None:
        print('Reading CH4 Data...')
        df = pd.read_csv(args.ch4_path, nrows=args.rows)

        lat = df['latitude'].to_numpy()
        lon = df['longitude'].to_numpy()
        
        # -- Show Where Data Samples Are
        print('Plotting CH4 Data...')
        plt.figure(3, clear=True)
        plt.hist2d(
            x=lon,
            y=lat,
            bins=256,
            density=False,
            cmin=1,
            cmax=60000
        )
        plt.xlabel('Longitude (Deg)')
        plt.ylabel('Latitude (Deg)')
        plt.title('# CH4 Records by Location\n{} Samples Total'.format(
            len(df)))
        plt.colorbar()
        plt.grid()
        
        # -- Show Number of Samples per Year
        epoch_time = df['time'].to_numpy()
        ref_epoch_time = datetime.datetime(2003, 1, 1).timestamp()
        epoch_time_03 = epoch_time - ref_epoch_time
        days_since_03 = epoch_time_03 / (60*60*24)
        years_since_03 = days_since_03 / 365.25
        
        plt.figure(4, clear=True)
        plt.hist(years_since_03, bins=10, edgecolor='k')
        plt.xlabel('Years Since 2003')
        plt.ylabel('Number of CH4 Data Samples By Year')
        plt.title('CH4 Samples Per Year')
        plt.grid()

        # -- Show Average Levels by Month
        n_moving_avg = 100
        mov_avg_kernel = np.ones(n_moving_avg) / n_moving_avg

        # Sort by timestamp to get global time average
        df = df.sort_values(by=['time'])
        ch4_mov_avg = signal.lfilter(b=mov_avg_kernel, a=1, x=df['xch4'])
        n_skip = 5000  # skip every few measurements so plot doesn't lock up

        epoch_time = df['time'].to_numpy()
        ref_epoch_time = datetime.datetime(2003, 1, 1).timestamp()
        epoch_time_03 = epoch_time - ref_epoch_time
        days_since_03 = epoch_time_03 / (60*60*24)
        years_since_03 = days_since_03 / 365.25

        plt.figure(5, clear=True)
        plt.plot(
            years_since_03[n_moving_avg::n_skip],
            ch4_mov_avg[n_moving_avg::n_skip]
        )
        plt.xlabel('Years Since 2003')
        plt.ylabel('ppm')
        plt.grid()
        plt.title('{} Sample Global Moving Average CH4 Concentration in ppb'.format(n_moving_avg))

    plt.show()
    print('Completed')

if __name__ == '__main__':
    main()
