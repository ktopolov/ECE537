"""
Script for model training and evaluation
"""
# %% Imports
# Path related
from pathlib import Path
import argparse

# Internal
from codebase import dataio

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
        dataio.co2_data_to_csv(
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
        dataio.ch4_data_to_csv(
            files=filepaths,
            csv_name=ch4_csv_name,
            is_debug=True
        )
        print('{} written'.format(ch4_csv_name))

if __name__ == '__main__':
    main()
