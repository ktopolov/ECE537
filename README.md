# ECE537
This code is used for a project predicting optimal locations for carbon removal site locations using collected CO2 and CH4 datasets.  
  
See "Midterm_Proposal.docx" for a description on the goal for the project.  
  
# Repository Structure  
The repository is a Python-based repository. The structure splits files into three main categories:  
*  `codebase`: This is a pip-installable package which contains utility functions and command line interface (CLI) scripts for support  
*  `examples`: This contains example scripts demonstrating how to use the tools inside of `codebase`  
*  `cli`: This contains all command-line executable python scripts including the main application itself  
  
## codebase  
To install `codebase`, first you must install the following external packages:  

```
pyproj     # See "https://anaconda.org/conda-forge/pyproj"    - Command is "conda install -c conda-forge pyproj"
netcdf4    # See "https://anaconda.org/anaconda/netcdf4"      - Command is "conda install -c anaconda netcdf4"
pathlib    # See "https://anaconda.org/anaconda/pathlib"      - Command is "conda install -c anaconda pathlib"  
mpi4py     # See "https://anaconda.org/conda-forge/mpi4py/"   - Command is "conda install -c conda-forge mpi4py"
simplekml  # See "https://anaconda.org/conda-forge/simplekml" - Command is "conda install -c conda-forge simplekml" (May need to install with pip instead)
```

navigate inside the `ECE537` repository and use:  
```
pip install .
```  
  
## cli  
This section describes CLI scripts' purpose and provides example CLI calls.  
  
#### write_data_to_csv.py  
This script allows us to extract data from folders containing the `.nc` files containing atmospheric data, extract the desired data and store in a single `.csv` file. Example call is:  
```
python /mnt/c/Users/ktopo/Desktop/ECE537/cli/write_data_to_csv.py --co2 /mnt/c/Users/ktopo/Desktop/ece537_data/C02_SCIA_IMAP/ --ch4 /mnt/c/Users/ktopo/Desktop/ece537_data/CH4_SCIA_IMAP_v72/ --out-dir /mnt/c/Users/ktopo/Desktop/ECE537/data  
```
#### visualize_data.py  
This script opens many plots to give a high-level overview of the data stored in the input CSV files. Example is:  
```
python /mnt/c/Users/ktopo/Desktop/ece537_data/visualize_data.py --co2-csv /mnt/c/Users/ktopo/Desktop/ECE537/data/co2_data.csv --ch4-csv /mnt/c/Users/ktopo/Desktop/ECE537/data/ch4_data.csv 
```  
  
  #### main_app.py  
  This is the main application to use a trained model for prediction. There are two modes:  
  *  Location-Based: Given a set of lat/lon locations to test, rank them in terms of which is optimal location for carbon removal site and provide some metrics. Output to KML.  
  *  Region-Based: Given latitude/longitude boundaries and some spacing, measure metrics over the region and provide contours over the region indicating where metrics are maximized  
  
Input arguments are:  
*  `--mode`: Can be `region` or `location`.  
*  `--start-time`: Starting month and year of simulation. Stored as `--start-time month year`
*  `--months`: Number of months to simulate  
*  `--lat-bounds`: Minimum and maximum latitude boundaries (degrees, -90.0 to 90.0) entered as `min max`. Only needed for `--mode region`
*  `--lon_bounds`: Minimum and maximum longitude boundaries (degrees, -180.0 to 180.0) entered as `min max`. Only needed for `--mode region`  
*  `--lat-res`: Resolution or spacing of latitude grid samples (deg); only for `--mode region`
*  `--lon-res`: Resolution or spacing of longitude grid samples (deg); only for `--mode region`  
*  `--locs`: Locations of points to test for `--mode location`. Stored as `--locs lat1 lon1 lat2 lon2 ...`
Example command line calls are:  
```
--mode region --lat-bounds -20.0 20.0 --lat-res 1.0 --lon-bounds -30.0 30.0 --lon-res 2.0 --start-time 10 2022 --months 24
```  
  
```  
--mode location --locs 20.0 30.0 15.5 22.3 --start-time 10 2022 --months 36
```
