# -*- coding: utf-8 -*-
"""
Support for loading and writing dataset
"""
# %% Imports
import numpy as np
import pandas as pd
import netCDF4 as nc
import datetime

# %% Write to CSV
def co2_data_to_csv(files, csv_name, is_debug=False):
    """
    Write dataset from a series of .nc files to a single .csv

    Parameters
    ----------
    files : [] str
        Full file paths to .nc files

    csv_name : str
        Full filepath of desired CSV file

    is_debug : bool
        Print information giving status of data loading

    Returns
    -------
    """
    n_file = len(files)

    for ii, filepath in enumerate(files):
        if is_debug and (np.mod(ii, 100) == 0):
            print('File {}/{}'.format(ii+1, n_file))

        dataset = nc.Dataset(filepath)
        n_points = dataset['time'].size
    
        # Store day month year
        day = np.zeros(n_points, dtype=int)
        month = np.zeros(n_points, dtype=int)
        year = np.zeros(n_points, dtype=int)
        
        # Store days since Jan. 1 2003 when data was first collected
        ref_date_time = datetime.datetime(2003, 1, 1)
        day_since_03 = np.zeros(n_points, dtype=int)
    
        epoch_time = np.array(dataset['time'])
        for jj, time in enumerate(epoch_time):
            date_time = datetime.datetime.fromtimestamp(time) 
    
            day[jj] = date_time.day
            month[jj] = date_time.month
            year[jj] = date_time.year
    
            rel_date_time = date_time - ref_date_time
            day_since_03[jj] = rel_date_time.days
    
        # Add dictionary to dataframe
        data = {
            'lon': np.array(dataset['longitude']),
            'lat': np.array(dataset['latitude']),
            'day': day,
            'month': month,
            'year': year,
            'day_since_03': day_since_03,  # why does 2003 folder have epoch time for 2004?
            'co2': np.array(dataset['xco2']),
            }
        temp_df = pd.DataFrame(data)

        if ii == 0:  # create .csv
            temp_df.to_csv(csv_name, mode='w', index=False, header=True)
        else:  # append to existing .csv
            temp_df.to_csv(csv_name, mode='a', index=False, header=False)  # don't keep rewriting header


def ch4_data_to_csv(files, csv_name, is_debug=False):
    """
    Write CH4 data from a series of .nc files to a single .csv

    Parameters
    ----------
    files : [] str
        Full file paths to .nc files

    csv_name : str
        Full filepath of desired CSV file

    is_debug : bool
        Print information giving status of data loading

    Returns
    -------
    """
    n_file = len(files)

    for ii, filepath in enumerate(files):
        if is_debug and (np.mod(ii, 100) == 0):
            print('File {}/{}'.format(ii+1, n_file))

        # Print variable names
        # print(dataset.variables.keys())
        dataset = nc.Dataset(filepath)
        n_points = dataset['time'].size
    
        # Store day month year
        day = np.zeros(n_points, dtype=int)
        month = np.zeros(n_points, dtype=int)
        year = np.zeros(n_points, dtype=int)
        
        # Store days since Jan. 1 2003 when data was first collected
        ref_date_time = datetime.datetime(2003, 1, 1)
        day_since_03 = np.zeros(n_points, dtype=int)
    
        epoch_time = np.array(dataset['time'])
        for jj, time in enumerate(epoch_time):
            date_time = datetime.datetime.fromtimestamp(time) 
    
            day[jj] = date_time.day
            month[jj] = date_time.month
            year[jj] = date_time.year
    
            rel_date_time = date_time - ref_date_time
            day_since_03[jj] = rel_date_time.days
    
        # Add dictionary to dataframe
        data = {
            'lon': np.array(dataset['longitude']),
            'lat': np.array(dataset['latitude']),
            'day': day,
            'month': month,
            'year': year,
            'day_since_03': day_since_03,  # why does 2003 folder have epoch time for 2004?
            'ch4': np.array(dataset['xch4']),
            }
        temp_df = pd.DataFrame(data)

        if (ii == 0):
            temp_df.to_csv(csv_name, mode='w', index=False, header=True)
        else:  # append to existing .csv
            temp_df.to_csv(csv_name, mode='a', index=False, header=False)  # don't keep rewriting header
