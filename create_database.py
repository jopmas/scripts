import os
import sys
import xarray as xr
import numpy as np

path = os.getcwd().split('/')
machine_path = f'/{path[1]}/{path[2]}' #cat the /home/user/ or /Users/user from system using path

path_to_functions = f"{machine_path}/opt/scripts"
sys.path.append(os.path.abspath(path_to_functions))

if '' in sys.path:
    sys.path.remove('')
from functions.mandyocIO import read_mandyoc_output, read_datasets

model_path = os.getcwd()
model_name = os.path.split(model_path)[1]

# if(__name__ == '__main__'):
#     print('Running as script')
# # print(sys.path)
#     print(read_mandyoc_output.__module__)
#     print(read_mandyoc_output.__code__.co_filename)
#     print('\n')

datasets = (#Properties from mandyoc. Comment/uncomment to select properties of the dataset
            'density',
            'depletion_factor',
            'incremental_melt',
            'melt',
            'radiogenic_heat',
            'pressure',
            'strain',
            'strain_rate', ### Read ascii outputs and save them as xarray.Datasets,
            'temperature',
            'viscosity',
            'surface',
            )

ds_data = read_mandyoc_output(
        model_path,
        datasets=datasets,
        parameters_file="param.txt"
    )

dataset = read_datasets(model_path, datasets, save_big_dataset = False)

