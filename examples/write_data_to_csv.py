"""
Script for model training and evaluation
"""
# %% Imports
# Path related
import os
from pathlib import Path
import argparse

# Internal
import codebase.dataio

# %% Write CO2 Dataset to single CSV
# Only do this if you have not already written to .csv
is_write = True
csv_name = 'co2_data.csv'

if is_write:
    # only use one year for now
    data_path = os.path.normpath('./data/C02_SCIA_IMAP')
    filepaths = [path for path in Path(data_path).rglob('*.nc')]

    dataio.co2_data_to_csv(
        files=filepaths,
        csv_name=csv_name,
        is_debug=True
    )

# %% Write CH4 Dataset to single CSV
# Only do this if you have not already written to .csv
is_write = True
csv_name = 'ch4_data.csv'

if is_write:
    # only use one year for now
    data_path = os.path.normpath('./data/CH4_SCIA_IMAP_v72')
    filepaths = [path for path in Path(data_path).rglob('*.nc')]

    dataio.ch4_data_to_csv(
        files=filepaths,
        csv_name=csv_name,
        is_debug=True
    )

def main():
    """
    Write data in NETCDF files to .csv
    """
    parser = argparse.ArgumentParser(description='Write CO2 and CH4 NETCDF files to .csv')
    parser.add_argument('--co2', type=str, help='Path to CO2 netcdf ')
    parser.add_argument('--ch4', type=str, help='Path to CH4 netcdf ')
    parser.add_argument('--ch4', type=str, help='Path to CH4 netcdf ')

    args = parser.parse_args()
    print(args.accumulate(args.integers))

if __name__ == '__main__':
    main()
