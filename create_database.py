import os
import gc
import xarray as xr
import numpy as np

from functions.mandyocIO import read_mandyoc_output, read_datasets, read_particle_path, plot_data


model_path = os.getcwd()
model_name = os.path.split(model_path)[1]

datasets = (#Properties from mandyoc. Comment/uncomment to select properties of the dataset
            'density',
            'radiogenic_heat',
            'pressure',
            'strain',
            'strain_rate',### Read ascii outputs and save them as xarray.Datasets,
            'surface',
            'temperature',
            'viscosity'
            )

ds_data = read_mandyoc_output(
        model_path,
        datasets=datasets,
        parameters_file="param.txt"
    )

dataset = read_datasets(model_path, datasets, save_big_dataset = True)

