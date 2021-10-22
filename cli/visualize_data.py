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
        df = pd.read_csv(args.co2_path, nrows=args.rows)

        lat = df['lat'].to_numpy()
        lon = df['lon'].to_numpy()
        
        # %% Show Where Data Samples Are
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
        year = df['year'].to_numpy()
        
        plt.figure(1, clear=True)
        plt.hist(year, bins=10, edgecolor='k')
        plt.xlabel('Year')
        plt.ylabel('Number of CO2 Data Samples By Year')
        plt.title('CO2 Samples Per Year')
        plt.grid()
        
        # %% Show Average Levels by Month
        years = np.arange(2003, 2013)
        months = np.arange(1, 13)
        
        months, years = np.meshgrid(months, years)
        years = years.flatten()
        months = months.flatten()
        
        mean_co2 = np.zeros(years.shape)
        
        for ii, (month, year) in enumerate(zip(months, years)):
            samples = df.loc[(df.month == month) & (df.year == year)]
            mean_co2[ii] = samples['co2'].mean()
        
        plt.figure(2, clear=True)
        plt.plot(mean_co2)
        plt.xlabel('Months Since 2003')
        plt.ylabel('ppm')
        plt.grid()
        plt.title('Mean Global CO2 Concentration in ppm')
    
    
    # %% Read from CSV
    if args.ch4_path is not None:
        df = pd.read_csv(args.ch4_path, nrows=args.rows)

        lat = df['lat'].to_numpy()
        lon = df['lon'].to_numpy()
        
        # -- Show Where Data Samples Are
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
        year = df['year'].to_numpy()
        
        plt.figure(4, clear=True)
        plt.hist(year, bins=10, edgecolor='k')
        plt.xlabel('Year')
        plt.ylabel('Number of CH4 Data Samples By Year')
        plt.title('CH4 Samples Per Year')
        plt.grid()
        
        # -- Show Average Levels by Month
        years = np.arange(2003, 2013)
        months = np.arange(1, 13)
        
        months, years = np.meshgrid(months, years)
        years = years.flatten()
        months = months.flatten()
        
        mean_ch4 = np.zeros(years.shape)
        
        for ii, (month, year) in enumerate(zip(months, years)):
            samples = df.loc[(df.month == month) & (df.year == year)]
            mean_ch4[ii] = samples['ch4'].mean()
        
        plt.figure(5, clear=True)
        plt.plot(mean_ch4)
        plt.xlabel('Months Since 2003')
        plt.ylabel('ppm')
        plt.grid()
        plt.title('Mean Global CH4 Concentration in ppm')

    plt.show()

if __name__ == '__main__':
    main()
