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
python /mnt/c/Users/ktopo/Desktop/ECE537/examples/python write_data_to_csv.py --co2 /mnt/c/Users/ktopo/Desktop/ece537_data/C02_SCIA_IMAP/ --ch4 /mnt/c/Users/ktopo/Desktop/ece537_data/CH4_SCIA_IMAP_v72/ --out-dir /mnt/c/Users/ktopo/Desktop/ECE537/data  
```
#### visualize_data.py  
This script opens many plots to give a high-level overview of the data stored in the input CSV files. Example is:  
```
python /mnt/c/Users/ktopo/Desktop/ece537_data/visualize_data.py --co2-csv /mnt/c/Users/ktopo/Desktop/ECE537/data/co2_data.csv --ch4-csv /mnt/c/Users/ktopo/Desktop/ECE537/data/ch4_data.csv 
```
