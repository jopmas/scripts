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

# path = os.getcwd().split('/')
# machine_path = '/'+path[1]+'/'+path[2] #cat the /home/user/ or /Users/user from system using path

machine_path = '/home/joao_macedo' #cat the /home/user/ or /Users/user from system using path

sys.path.insert(0, f"{machine_path}/opt/mandyoc-scripts/functions")
from mandyocIO import calc_mean_temperaure_region, calc_and_plot_YSE

####################################################################################################################################
#Get model informations
    
model_path = os.getcwd() # Get local file
model_name = model_path.split('/')[-1]
model_features = model_name.split('_')
output_path = '_output'
print(f"Model name: {model_name}\n")
print(f"Model path: {model_path}\n")
print(f"Output path: {output_path}\n")



if not os.path.isdir(output_path):
    os.makedirs(output_path)

#Read dataset
dataset = xr.open_dataset(f'{model_path}/_output_temperature.nc')

#Domain infos
Nx = int(dataset.nx)
Nz = int(dataset.nz)
Lx = float(dataset.lx)
Lz = float(dataset.lz)

instant = dataset.time[-1]

x = np.linspace(0, Lx/1000.0, Nx)
z = np.linspace(Lz/1000.0, 0, Nz)
xx, zz  = np.meshgrid(x, z)

if('lc' in model_features):
    Lcraton = 2000.0 #km
else:
    Lcraton = 1200.0 #km
print(f"Keel Length: {Lcraton} km\n")

if('sc' in model_features):
    shift_craton = 700.0e3
else:
    shift_craton = 0.0e3
print(f"Shift on cratonic keel: {shift_craton} km\n")
    
t0 = dataset.time[0]
t1 = dataset.time[1]
dt = int(t1 - t0)

start = int(t0)
end = int(dataset.time.size - 1)
step = 1

# start = 0
# end = 1
# step = 1#5

print(f"Plottting YSE frames...")
with pymp.Parallel() as p:
    for i in p.range(start, end+step, step):

        data = dataset.temperature[i].values.T #Nz vs Nx
        time = np.round(dataset.time[i].values, 2)

        ############################################################################################

        #Calculating mean temperature profile for different regions  

        xcenter = (Lx/2)/1.0e3 + shift_craton/1.0e3

        # keel_mean_l = calc_mean_temperaure_region(data, Nz, xx, xcenter - 600, xcenter - 200)
        # keel_mean_c = calc_mean_temperaure_region(data, Nz, xx, xcenter - 100, xcenter + 100)
        # keel_mean_r = calc_mean_temperaure_region(data, Nz, xx, xcenter + 200, xcenter + 600)

        keel_mean_l = calc_mean_temperaure_region(data, Nz, xx, xcenter - 350, xcenter - 250)
        keel_mean_c = calc_mean_temperaure_region(data, Nz, xx, xcenter - 50, xcenter + 50)
        keel_mean_r = calc_mean_temperaure_region(data, Nz, xx, xcenter + 250, xcenter + 350)

        out_mean_l = calc_mean_temperaure_region(data, Nz, xx, 200, xcenter - Lcraton/2 - 200)
        out_mean_r = calc_mean_temperaure_region(data, Nz, xx, xcenter + Lcraton/2 + 200, Lx/1.03 - 200)

        thickness_sa = 40.0e3
        z_aux = z - thickness_sa/1.0e3
        cond1 = z_aux >= 0

        temperature_profiles = [out_mean_l, keel_mean_l, keel_mean_c, keel_mean_r, out_mean_r]
        Hlits = [80.0e3, 200.0e3, 200.0e3, 200.0e3, 80.0e3]

        plt.close()
        fig, axs = plt.subplots(1, 5, figsize=(16,4), sharex=True, sharey=True, constrained_layout=True)
        axs.flatten()

        for ax, temp_profile, thickness_litho in zip(axs, temperature_profiles, Hlits):
            calc_and_plot_YSE(ax, temp_profile[cond1], z_aux[cond1], thickness_litho=thickness_litho)


        #Set plot details
        # locations = [f'outside keel left side:\n $200 \leq x \leq {int(xcenter - 800)}$ km',
        #             f'keel left side:\n ${int(xcenter - 600)} \leq x \leq {int( xcenter - 200)}$ km',
        #             f'keel center:\n ${int(xcenter - 100)} \leq x \leq {int( xcenter + 100)}$ km',
        #             f'keel right side:\n ${int(xcenter + 200)} \leq x \leq {int( xcenter + 600)}$ km',
        #             f'outside keel right side:\n ${int(xcenter + 800)} \leq x \leq {int(Lx/1.0e3 - 200)}$ km'
        #             ]
            
        locations = [f'outside keel left side:\n $200 \leq x \leq {int(xcenter - Lcraton/2 - 200)}$ km',
                    f'keel left side:\n ${int(xcenter - 350)} \leq x \leq {int(xcenter - 250)}$ km',
                    f'keel center:\n ${int(xcenter - 50)} \leq x \leq {int(xcenter + 50)}$ km',
                    f'keel right side:\n ${int(xcenter + 250)} \leq x \leq {int(xcenter + 350)}$ km',
                    f'outside keel right side:\n ${int(xcenter + Lcraton/2 + 200)} \leq x \leq {int(Lx/1.0e3 - 200)}$ km'
                    ]

        for ax, location in zip(axs, locations):
            ax.grid('-k', alpha=0.7)
            ax.set_xlabel(r'$\Delta\sigma$ [GPa]')
            ax.text(0.015, 0.02, location, fontsize=12, transform=ax.transAxes)
            
        axs[-1].text(0.65, 0.95, f"{time} Myr", fontsize=14, transform=axs[-1].transAxes)
        axs[0].set_ylabel('Depth [km]')

        figname = f'YSE_{model_name}_{str(int(dataset.step[i].values)).zfill(8)}'
        # print(figname)
        fig.savefig(f'{output_path}/{figname}.png', dpi=400)

        plt.close('all')
        del data
        del fig
        del axs
        gc.collect()
print("Done!\n")

##########################################################################################################################################################################
#Create video and gif
        
make_videos = True 
# make_videos = False

if(make_videos):
    print("Generating videos...")

    fps = 30    
    videoname = f"{model_path}/{output_path}/YSE_{model_name}"

    try:
        comand = f"rm {videoname}.mp4"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
        print(f"\tRemoving previous YSE video.")
    except:
        print(f"\tNo YSE video to remove.")

    comand = f"ffmpeg -r {fps} -f image2 -s 1920x1080 -pattern_type glob -i \"{videoname}_*.png\" -vcodec libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 24 -y -an -crf 25 -pix_fmt yuv420p {videoname}.mp4"
    
    result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
    print("\tDone!\n")

make_gifs = True
# make_gifs = False

if(make_gifs):
    print("Converting videos to gifs...")
    gifname = f'{model_path}/{output_path}/YSE_{model_name}'
            
    try:
        comand = f"rm {gifname}.gif"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
        print(f"\tRemoving previous YSE gif.")
    except:
        print(f"\tNo YSE gif to remove.")
    
    comand = f"ffmpeg -ss 0 -t 15 -i '{gifname}.mp4' -vf \"fps=60,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" -loop 0 {gifname}.gif"
    result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True) 
    print("\tDone!\n")

zip_files = True
# zip_files = False
if(zip_files):
    #zip plots, videos and gifs
    print('Zipping YSE figures, videos and gifs...')
    outputs_path = f'{model_path}/{output_path}/'
    os.chdir(outputs_path)
    subprocess.run(f"zip {model_name}_imgs.zip YSE_{model_name}_*.png", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"zip {model_name}_videos.zip YSE_{model_name}.mp4", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"zip {model_name}_gifs.zip YSE_{model_name}.gif", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"rm YSE_{model_name}_*.png", shell=True, check=True, capture_output=True, text=True)
    print('Zipping complete!')