"""
Script for model training and evaluation
"""
# %% Imports
# Path related
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import netCDF4 as nc

# %% Support
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
            print('\rFile {}/{}'.format(ii+1, n_file))

        # filepath = 'C:/Users/ktopo/Desktop/ece537_data/C02_SCIA_IMAP/2003/ESACCI-GHG-L2-CO2-SCIAMACHY-BESD-20030108-fv1.nc'
        dataset = nc.Dataset(filepath)
        # print(dataset.variables.keys())

        # Add dictionary to dataframe
        data = {
            'longitude': np.array(dataset['longitude']),
            'latitude': np.array(dataset['latitude']),
            'time': np.array(dataset['time']),
            'solar_zenith_angle': np.array(dataset['solar_zenith_angle']),
            'sensor_zenith_angle': np.array(dataset['sensor_zenith_angle']),
            'xco2_quality_flag': np.array(dataset['xco2_quality_flag']),
            'xco2': np.array(dataset['xco2']),
            'xco2_uncertainty': np.array(dataset['xco2_uncertainty']),
            #'co2_profile_apriori': np.array(dataset['co2_profile_apriori']),
            # 'pressure_levels': np.array(dataset['pressure_levels']),
            # 'pressure_weight': np.array(dataset['pressure_weight']),
            #'xco2_averaging_kernel': np.array(dataset['xco2_averaging_kernel']),,
        }
        # for key, val in data.items():
        #     print('{}: {}'.format(key, val.shape))
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
            print('\rFile {}/{}'.format(ii+1, n_file))

        # filepath = 'C:/Users/ktopo/Desktop/ece537_data/CH4_SCIA_IMAP_v72/2003/ESACCI-GHG-L2-CH4-SCIAMACHY-IMAP-20030108-fv1.nc'
        dataset = nc.Dataset(filepath)
        # print(dataset.variables.keys())

        # Add dictionary to dataframe
        data = {
            'solar_zenith_angle': np.array(dataset['solar_zenith_angle']),
            'sensor_zenith_angle': np.array(dataset['sensor_zenith_angle']),
            'time': np.array(dataset['time']),
            'longitude': np.array(dataset['longitude']),
            'latitude': np.array(dataset['latitude']),
            # 'pressure_levels': np.array(dataset['pressure_levels']),
            # 'pressure_weight': np.array(dataset['pressure_weight']),
            'xch4_raw': np.array(dataset['xch4_raw']),
            'xch4': np.array(dataset['xch4']),
            'h2o_ecmwf': np.array(dataset['h2o_ecmwf']),
            'xch4_prior': np.array(dataset['xch4_prior']),
            'xco2_prior': np.array(dataset['xco2_prior']),
            'xco2_retrieved': np.array(dataset['xco2_retrieved']),
            'xch4_uncertainty': np.array(dataset['xch4_uncertainty']),
            # 'xch4_averaging_kernel': np.array(dataset['xch4_averaging_kernel']),
            # 'ch4_profile_apriori': np.array(dataset['ch4_profile_apriori']),
            'xch4_quality_flag': np.array(dataset['xch4_quality_flag']),
            # 'dry_airmass_layer': np.array(dataset['dry_airmass_layer']),
            'surface_elevation': np.array(dataset['surface_elevation']),
            'surface_temperature': np.array(dataset['surface_temperature']),
            'chi2_ch4': np.array(dataset['chi2_ch4']),
            'chi2_co2': np.array(dataset['chi2_co2']),
            'xco2_macc': np.array(dataset['xco2_macc']),
            'xco2_CT2015': np.array(dataset['xco2_CT2015']),
            'xch4_v71': np.array(dataset['xch4_v71']),
        }
        # for key, val in data.items():
        #     print('{}: {}'.format(key, val.shape))

        temp_df = pd.DataFrame(data)

        if (ii == 0):
            temp_df.to_csv(csv_name, mode='w', index=False, header=True)
        else:  # append to existing .csv
            temp_df.to_csv(csv_name, mode='a', index=False, header=False)  # don't keep rewriting header


# %% Write CO2 Dataset to single CSV
def main():
    """
    Write data in NETCDF files to .csv
    """
    parser = argparse.ArgumentParser(description='Write CO2 and CH4 NETCDF files to .csv')
    parser.add_argument('--co2', dest='co2_dir', default=None, type=str, help='Path to CO2 netcdf ')
    parser.add_argument('--ch4', dest='ch4_dir', default=None, type=str, help='Path to CH4 netcdf ')
    parser.add_argument('--out-dir', default='.', type=str, help='Path to output .csv file(s) ')
    args = parser.parse_args()

    assert args.out_dir is not None, 'Require output directory'
    out_dir = Path(args.out_dir)

    is_write_co2 = args.co2_dir is not None
    if is_write_co2:
        co2_csv_name = out_dir / Path('co2_data.csv')
        co2_dir = Path(args.co2_dir)
        filepaths = [path for path in co2_dir.rglob('*.nc')]
        co2_data_to_csv(
            files=filepaths,
            csv_name=co2_csv_name,
            is_debug=True
        )
        print('{} written'.format(co2_csv_name))

    is_write_ch4 = args.ch4_dir is not None
    if is_write_ch4:
        ch4_csv_name = out_dir / Path('ch4_data.csv')
        ch4_dir = Path(args.ch4_dir)
        filepaths = [path for path in ch4_dir.rglob('*.nc')]
        ch4_data_to_csv(
            files=filepaths,
            csv_name=ch4_csv_name,
            is_debug=True
        )
        print('{} written'.format(ch4_csv_name))

if __name__ == '__main__':
    main()
