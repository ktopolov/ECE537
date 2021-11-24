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
python /mnt/c/Users/ktopo/Desktop/ECE537/cli/visualize_data.py --co2-csv /mnt/c/Users/ktopo/Desktop/ECE537/data/co2_data.csv --ch4-csv /mnt/c/Users/ktopo/Desktop/ECE537/data/ch4_data.csv 
```  
  
  #### main_app.py  
  This is the main application to use a trained model for prediction. There are two modes:  
  *  Location-Based: Given a set of lat/lon locations to test, rank them in terms of which is optimal location for carbon removal site and provide some metrics. Output to KML.  
  *  Region-Based: Given latitude/longitude boundaries and some spacing, measure metrics over the region and provide contours over the region indicating where metrics are maximized  
  
Input arguments are:  
*  `--mode`: Can be `region` or `location`.  
*  `--start-date`: Start date and year of simulation. Stored as `--start-time DD MM YYYY`
*  `--stop-date`: Stop date of simulation. Stored as `--stop-date DD MM YYYY`  
*  `--sim-step`: Tiem resolution of simulation. Options are any one of the following: `daily, weekly, monthly, quarterly, anually`
*  `--lat-bounds`: Minimum and maximum latitude boundaries (degrees, -90.0 to 90.0) entered as `min max`. Only needed for `--mode region`
*  `--lon_bounds`: Minimum and maximum longitude boundaries (degrees, -180.0 to 180.0) entered as `min max`. Only needed for `--mode region`  
*  `--lat-res`: Resolution or spacing of latitude grid samples (deg); only for `--mode region`
*  `--lon-res`: Resolution or spacing of longitude grid samples (deg); only for `--mode region`  
*  `--locs`: Locations of points to test for `--mode location`. Stored as `--locs lat1 lon1 lat2 lon2 ...`  
*  `--out-dir`: Output file directory to store KML, PNG or other files from simulation  
Example command line calls are:  
```
python /home/ktopolov/ECE537/cli/cli_app.py --mode region --start-date 2003 10 01 --stop-date 2012 10 01 --lat-bounds -50.0 50.0 --lat-res 1.0 --lon-bounds -150.0 150.0 --lon-res 1.0 --sim-step weekly --out-dir /home/ktopolov/ECE537/output/runs
```  
  
```  
python /home/ktopolov/ECE537/cli/cli_app.py --mode location --start-date 2003 10 01 --stop-date 2012 10 01 --locs 10.0 20.0 15.5 36.2 18.2 25.5 17.7 32.9 30.5 18.5 --sim-step weekly --out-dir /home/ktopolov/ECE537/output/runs
```  
  
# External Tools  
## TensorBoard  
TensorBoard is a tool used to visualize the machine learning results. This will be installed by default with TensorFlow itself. It can track model performance in a graphical way real-time and also hyperparameter effects, parameter distributions, etc.  
  
To use TensorBoard, the code should create a **callback** with a user-specified output directory where TensorBoard logs will sit:  
```
tb_callback = keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
)
```
This clalback is then passed into the fitting for a Keras model:  
```
history = Model.fit(
  x=X_train,
  y=y_train,
  batch_size=hparams['batch_size'],
  epochs=hparams['n_epoch'],
  verbose=verbosity,
  validation_data=(X_cross, y_cross),
  callbacks=[
      tb_callback,  # log metrics
      hp_callback,  # log hparams
  ],
)
```  
Any metrics that are asked to be tracked during `Model.compile()` will appear in the logs. Cross-validation and training data also.  
  
To use TensorBoard, begin training your model. While training, or after it finishes, run `tensorboard --logdir <path_to_logs>` on the terminal, where th epath to the logs is a path either **directly** containing the Tensorboard logs (called `log_dir` in the example above), or it could contain many subfolders, each of which contain a single model's log directory.  
*  the first option can be used to look at a single model  
*  the second option allows us to make a different TensorBoard log directory for each of N models (for example, in hyperparameter training), and then compare their results.

After running that on the terminal, we see:
```
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.7.0 at http://localhost:6006/ (Press CTRL+C to quit)
```  
Copythe `http` address and paste it into an internet browser. Provided that you passed a valid TensorBoard log directory and there is already some file output there, TensorBoard will start showing you metrics.
