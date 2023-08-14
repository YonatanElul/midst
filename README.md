# midst
This is an official implementation of the code used in the paper "Data-Driven Modelling of Interrelated Dynamical Systems"

## Setup
In order to setup the package, simply create a new virtual 
environment using (for example):

`conda create --name midst_venv python==3.9`


Then activate the virtual environment using:

`conda activate midst_venv`

Finally, install the midst package using:

`pip install -e .`

## Running experiments
Once the midst package is installed, experiments can easily
be run from the `run` module.

In order to run the Lorenz attractors with single dynamics experiments:
* Generate the data using the script at: `midst\run\data_generation\generate_strange_attractors_similar_parameters.py`
* Run the appropriate experiment from the scripts in: `midst\run\lorenz\similar_parameters`

In order to run the Lorenz attractors with different dynamics experiments:
* Generate the data using the script at: `midst\run\data_generation\generate_strange_attractors_different_parameters.py`
* Run the appropriate experiment from the scripts in: `midst\run\lorenz\different_parameters`

In order to run the multiple attractors with different dynamics experiments:
* Generate the data using the script at: `midst\run\data_generation\generate_strange_attractors_different_attractors.py`
* Run the appropriate experiment from the scripts in: `midst\run\lorenz\different_attractors`

In order to run the Long QT Syndrome (LQTS) experiments:
* Generate the data using the scripts at: `midst\run\data_generation\generate_physionet_ecgrdvq_data.py`, followed by the one at: `midst\run\data_generation\generate_ecgrdvq_dataset.py`
* Run the appropriate experiment from the scripts in: `midst\run\leave_one_out_lqts`

In order to run the SST experiments:
* Download the raw sst data from `https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/` to the data dir at: `data\SSTV2`, specifically, we need the following files: `lsmask.nc, sst.mnmean.nc, sst.wkmean.1990-present.nc`
* Run the appropriate experiment from the scripts in: `midst\run\sst`

Each experiment will create its own logs directory in the logs directory of the project.
In it, it will automatically log the results for the train/validation/test phases of each experiment