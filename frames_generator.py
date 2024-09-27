#!/usr/bin/env python
# coding: utf-8

import os
import gc
import sys
import pymp
import shutil
import subprocess
import numpy as np
import xarray as xr


import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import Bbox
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d


matplotlib.use('agg')

path = os.getcwd().split('/')
machine_path = f'/{path[1]}/{path[2]}' #cat the /home/user/ or /Users/user from system using path

path_to_functions = f"{machine_path}/opt/scripts"
sys.path.append(os.path.abspath(path_to_functions))

if '' in sys.path:
    sys.path.remove('')
from functions.mandyocIO import read_datasets, change_dataset, single_plot

####################################################################################################################################
model_path = os.getcwd() # Get local file
model_name = model_path.split('/')[-1]
output_path = '_output'
print(f"Model name: {model_name}\n")
print(f"Model path: {model_path}\n")
print(f"Output path: {output_path}\n")
####################################################################################################################################

#Setting extra details to plot
plot_isotherms = True
# plot_isotherms = False

plot_melt = True
# plot_melt = False

melt_method = 'dry'
# melt_method = 'wet'

if(plot_isotherms or plot_melt):
    clean_plot=False
else:
    clean_plot = True

if not os.path.isdir(output_path):
    os.makedirs(output_path)

make_videos = True 
make_gifs = True
zip_files = True

# make_videos = False
# make_gifs = False
# zip_files = False

##########################################################################################################################################################################

datasets = [#Properties from mandyoc. Comment/uncomment to select properties of the dataset
            'density',
            'radiogenic_heat',
            'pressure',
            'strain',
            'strain_rate',### Read ascii outputs and save them as xarray.Datasets,
            'surface',
            'temperature',
            'viscosity'
            ]# Read data and convert them to xarray.Dataset

properties = [#Properties from mandyoc. Comment/uncomment to select which ones you would like to plot
#              'density',
#              'radiogenic_heat',
             'lithology',
#              'pressure',
            #  'strain',
            #  'strain_rate',
            #  'temperature',
             'temperature_anomaly',
             'surface',
            #  'viscosity'
             ]

print("Reading datasets...")

new_datasets = change_dataset(properties, datasets)
to_remove = []

remove_density=False
if ('density' not in properties): #used to plot air/curst interface
        properties.append('density')
        new_datasets = change_dataset(properties, datasets)
        to_remove.append('density')
        # remove_density=True
if ('surface' not in properties): #used to plot air/curst interface
        properties.append('surface')
        new_datasets = change_dataset(properties, datasets)
        to_remove.append('surface')

if (plot_isotherms): #add datasets needed to plot isotherms
    if ('temperature' not in properties):
        properties.append('temperature')
        new_datasets = change_dataset(properties, datasets)
        to_remove.append('temperature')
        
# if (plot_melt): #add datasets needed to plot melt fraction
#     if ('pressure' not in properties):
#         properties.append('pressure')
#     if ('temperature' not in properties):
#         properties.append('temperature')
#     new_datasets = change_dataset(properties, datasets)

#     #removing the auxiliary datasets to not plot
#     to_remove.append('pressure')
#     to_remove.append('temperature')

if (plot_melt): #add datasets needed to plot melt fraction
    if ('melt' not in properties):
        properties.append('melt')
    if ('incremental_melt' not in properties):
        properties.append('incremental_melt')
    new_datasets = change_dataset(properties, datasets)

    #removing the auxiliary datasets to not plot
    to_remove.append('melt')
    to_remove.append('incremental_melt')

if(clean_plot): #a clean plot
    new_datasets = change_dataset(properties, datasets)


for item in to_remove:
    properties.remove(item)


dataset = read_datasets(model_path, new_datasets)

# Normalize velocity values
if ("velocity_x" and "velocity_z") in dataset.data_vars:
    v_max = np.max((dataset.velocity_x**2 + dataset.velocity_z**2)**(0.5))    
    dataset.velocity_x[:] = dataset.velocity_x[:] / v_max
    dataset.velocity_z[:] = dataset.velocity_z[:] / v_max

print("Datasets read!")
##########################################################################################################################################################################

# plot_particles = True 
plot_particles = False
unzip_steps = False

if(plot_particles):
    # unzip_steps = True
    unzip_steps = False
    if(unzip_steps):
        print("Unzipping step_*.txt files...")
        comand = f"unzip -o {model_path}/{model_name}.zip step*.txt -d {model_path}"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)


t0 = dataset.time[0]
t1 = dataset.time[1]
dt = int(t1 - t0)

start = int(t0)
end = int(dataset.time.size - 1)
step = 1

# start = 100
# end = 102
# step = 1

# start = 30
# end = 31
# step = 1

print("Generating frames...")
with pymp.Parallel() as p:
    for i in p.range(start, end+step, step):
        # data = dataset.isel(time=i)
        per = np.round(100*(i+1-start)/(end-start), 2)
        text = f"Time: {np.round(float(dataset.isel(time=i).time), 2)} Myr; Step: {int(dataset.isel(time=i).step)}/{int(dataset.step.max())}, ({per:.2f}%)."
        
        # print(text, end='\r')
        data = dataset.isel(time=i)
        for prop in properties:
    #         print(f"Handeling {prop}.", end='\n')
            if(prop != 'surface'): # you can customize
                if(prop == 'strain_rate'):
                    Lcraton = 1200.0 #km
                    xlims = [float(dataset.isel(time=i).lx)/2.0e3 - Lcraton/2 - 50, float(dataset.isel(time=i).lx)/2.0e3 + Lcraton/2 + 50]
                    ylims = [-210, 40]
                else:
                    # xlims = [0, float(dataset.isel(time=i).lx) / 1.0e3]
                    ylims = [-float(dataset.isel(time=i).lz) / 1.0e3 + 40, 40]
                    xlims = [0, float(dataset.isel(time=i).lx) / 1.0e3]
                    # ylims = [-400, 40]

            else:
                xmin = 0 #+ 200
                xmax = float(dataset.isel(time=i).lx) / 1.0E3 #- 200
                xlims = [xmin, xmax]
                ylims = [-1, 1]

            if(prop == 'viscosity'):
                single_plot(data, prop, xlims, ylims, model_path, output_path,
                        plot_isotherms = plot_isotherms,
                        plot_particles = False,
                        particle_size = 0.02,
                        particle_marker = ".",
                        ncores = 20,
                        # step_plot = 3,
                        isotherms = [500, 1300],
                        plot_melt = plot_melt,
                        melt_method = melt_method)
            else:
                single_plot(data, prop, xlims, ylims, model_path, output_path,
                            plot_isotherms = plot_isotherms,
                            plot_particles = plot_particles,
                            particle_size = 0.02,
                            # particle_size = 0.2,
                            particle_marker = ".",
                            ncores = 20,
                            # step_plot = 3,
                            isotherms = [500, 1300],
                            plot_melt = plot_melt,
                            melt_method = melt_method)
            
            
        del data
        gc.collect()


print("\tDone!")

##########################################################################################################################################################################
if(make_videos):
    print("Generating videos...")

    fps = 30    
    for prop in properties:
        videoname = f'{model_path}/_output/{model_name}_{prop}'

        if(plot_melt):
            videoname = f'{videoname}_MeltFrac'

        if(plot_particles):
            if(prop == 'viscosity'):
                videoname = f'{videoname}'
            else:
                videoname = f'{videoname}_particles'
                # videoname = f'{videoname}_particles_onlymb'
            
        try:
            comand = f"rm {videoname}.mp4"
            result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
            print(f"\tRemoving previous {prop} video.")
        except:
            print(f"\tNo {prop} video to remove.")

        comand = f"ffmpeg -r {fps} -f image2 -s 1920x1080 -pattern_type glob -i \"{videoname}_*.png\" -vcodec libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 24 -y -an -crf 25 -pix_fmt yuv420p {videoname}.mp4"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
    print("\tDone!")


##########################################################################################################################################################################

# # Converting videos to gifs
# 
# ss: skip seconds
# 
# t: duration time of the output
# 
# i: inputs format
# 
# vf: filtergraph (video filters)
# 
#     - fps: frames per second
# 
#     - scale: resize accordint to given pixels (e.g. 1080 = 1080p wide)
#     
#     - lanczos: scaling algorithm
#     
#     - palettegen and palette use: filters that generate a custom palette
#     
#     - split: filter that allows everything to be done in one command
# 
# loop: number of loops
# 
#     - 0: infinite
# 
#     - -1: no looping
# 
#     - for numbers n >= 0, create n+1 loops


if(make_gifs):
    print("Converting videos to gifs...")
    for prop in properties:
        gifname = f'{model_path}/_output/{model_name}_{prop}'

        if(plot_melt):
            gifname = f'{gifname}_MeltFrac'

        if(plot_particles):
            if(prop == 'viscosity'):
                gifname = f'{gifname}'
            else:
                gifname = f'{gifname}_particles'
                # gifname = f'{gifname}_particles_onlymb'
            

        try:
            comand = f"rm {gifname}.gif"
            result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
            print(f"\tRemoving previous {prop} gif.")
        except:
            print(f"\tNo {prop} gif to remove.")
        
        comand = f"ffmpeg -ss 0 -t 15 -i '{gifname}.mp4' -vf \"fps=30,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" -loop 0 {gifname}.gif"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True) 
    print("\tDone!")

##########################################################################################################################################################################

if(zip_files):
    #zip plots, videos and gifs
    print('Zipping figures, videos and gifs...')
    outputs_path = f'{model_path}/_output/'
    os.chdir(outputs_path)
    subprocess.run(f"zip {model_name}_imgs.zip *.png", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"zip {model_name}_videos.zip *.mp4", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"zip {model_name}_gifs.zip *.gif", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"rm *.png", shell=True, check=True, capture_output=True, text=True)
    print('Zipping complete!')


if(unzip_steps):
    # clean_steps = True
    clean_steps = False
    if(clean_steps):
        print("Cleaning step_*.txt files...")
        comand = f"rm {model_path}/step*.txt"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)

