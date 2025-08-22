#!/usr/bin/env python
# coding: utf-8

import os
import gc
import sys
import multiprocessing #needed to run pymp in mac
multiprocessing.set_start_method('fork') #needed to run pymp in mac
import pymp
import subprocess
import numpy as np
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker


matplotlib.use('agg')

path = os.getcwd().split('/')
machine_path = f'/{path[1]}/{path[2]}' #cat the /home/user/ or /Users/user from system using path

path_to_functions = f"{machine_path}/opt/scripts"
sys.path.append(os.path.abspath(path_to_functions))

if '' in sys.path:
    sys.path.remove('')
from functions.mandyocIO import read_datasets, change_dataset, plot_property, find_nearest, _extract_interface

####################################################################################################################################
model_path = os.getcwd() # Get local file
model_name = model_path.split('/')[-1]
output_path = '_output'
print(f"Model name: {model_name}\n")
print(f"Model path: {model_path}\n")
print(f"Output path: {output_path}\n")

if not os.path.isdir(output_path):
    os.makedirs(output_path)

plot_isotherms = True
# plot_isotherms = False
plot_melt = True
# plot_melt = False
plot_particles=False

# plot_three_particles = True
plot_three_particles = False

plot_depth_coded = True
# plot_depth_coded = False

# plot_temperature_coded = True
plot_temperature_coded = False

if(plot_isotherms or plot_melt):
    clean_plot=False
else:
    clean_plot = True

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
            #  'density',
            #  'radiogenic_heat',
             'lithology',
            #  'pressure',
            #  'strain',
            #  'strain_rate',
            #  'temperature',
            #  'temperature_anomaly',
            #  'surface',
            #  'viscosity'
             ]

#######################################################
# Read ascii outputs and save them as xarray.Datasets #
#######################################################

new_datasets = change_dataset(properties, datasets)
# print(new_datasets)
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
        # remove_density=True

if (plot_isotherms): #add datasets needed to plot isotherms
    if ('temperature' not in new_datasets):
        properties.append('temperature')
        new_datasets = change_dataset(properties, datasets)
        to_remove.append('temperature')

# print(f"newdatasets: {new_datasets}")

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

#########################################
# Get domain and particles informations #
#########################################

Nx = int(dataset.nx)
Nz = int(dataset.nz)
Lx = float(dataset.lx)
Lz = float(dataset.lz)

x = np.linspace(0, Lx/1000.0, Nx)
z = np.linspace(Lz/1000.0, 0, Nz)
xx, zz  = np.meshgrid(x, z)

trackdataset = xr.open_dataset("_track_xzPT_all_steps.nc")
x_track = trackdataset.xtrack.values[::-1]
z_track = trackdataset.ztrack.values[::-1]
P = trackdataset.ptrack.values[::-1]
T = trackdataset.ttrack.values[::-1]
time = trackdataset.time.values[::-1]
# print(time)
steps = trackdataset.step.values[::-1]
# print(len(steps), len(time))
n = int(trackdataset.ntracked.values)
nTotal = np.size(x_track)
steps = nTotal//n #

# print(list(time))

print(f"len of:\n x_track: {len(x_track)}\n z_track: {len(z_track)}\n P: {len(P)}\n T: {len(T)}\n time: {len(time)}\n")
print(f"n_tracked x len(all_time) = {n}*{len(time)} = {n*len(time)}")
print(f"nTotal: {nTotal}, n: {n}, steps: {steps}")
x_track = np.reshape(x_track,(steps,n))
z_track = np.reshape(z_track,(steps,n))
P = np.reshape(P,(steps,n))
T = np.reshape(T,(steps,n))

####################################################################################################################
# Take the index of particles_layers which corresponds to mlit layer: coldest, hotterst, and the one in the middle #
####################################################################################################################

particles_layers = trackdataset.particles_layers.values[::-1] #code of the tracked layers
asthenosphere_code = 0 #asthenosphere
mantle_lithosphere1_code = 1
seed_code = 2 #seed
mantle_lithosphere2_code = 3 #lithospheric mantle
lower_crust_code = 4 #lower crust
upper_crust_code = 5 #upper crust

T_initial = T[0] #initial temperature of particles
P_initial = P[0] #initial pressure of particles 

def take_three_particles(layer_codes, particles_layers, T_initial):
    if(len(layer_codes) == 1):
        cond = particles_layers == layer_codes[0]
    else:
        cond = (particles_layers == layer_codes[0]) | (particles_layers == layer_codes[1])

    particles_layer = particles_layers[cond]

    T_initial_layer = T_initial[cond] #initial temperature of lithospheric mantle particles
    T_initial_layer_sorted = np.sort(T_initial_layer)

    Ti_layer_max = np.max(T_initial_layer_sorted)
    mid_index = len(T_initial_layer_sorted)//2
    Ti_layer_mid = T_initial_layer_sorted[mid_index] #can bring more than one particle if they have same temperature
    Ti_layer_min = np.min(T_initial_layer_sorted)

    cond2plot = (T_initial == Ti_layer_min) | (T_initial == Ti_layer_mid) | (T_initial == Ti_layer_max)

    return cond2plot, cond, Ti_layer_max, Ti_layer_mid, Ti_layer_min, len(particles_layer)

if(asthenosphere_code in particles_layers):
    cond_ast = particles_layers == asthenosphere_code #condition to select asthenospheric particles
    particles_ast = particles_layers[cond_ast]
    print(f"N asthenospheric particles: {len(particles_ast)}")
    #Selecting particles with P0 <= 4000 MPa
    cond_ast2plot = P_initial <= 4000
    plot_asthenosphere_particles = True

    ast_color = 'xkcd:violet'
    ast_marker = '.'
else:
    plot_asthenosphere_particles = False
    cond_ast2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

if((mantle_lithosphere1_code in particles_layers) | (mantle_lithosphere2_code in particles_layers)):
    cond_mantle_lithosphere2plot, cond_mantle_lithosphere, Ti_mlit_max, Ti_mlit_mid, Ti_mlit_min, n_mantle_lithosphere = take_three_particles([mantle_lithosphere1_code,mantle_lithosphere2_code],
                                                                                               particles_layers, T_initial)
    
    print(f"N lithospheric mantle particles: {n_mantle_lithosphere}")

    plot_mantle_lithosphere_particles = True
    
    dict_mantle_lithosphere_markers = {Ti_mlit_max: '*',
                                         Ti_mlit_mid: '^',
                                         Ti_mlit_min: 'D'}

    dict_mantle_lithosphere_colors = {Ti_mlit_min: 'xkcd:cerulean blue',
                                       Ti_mlit_mid: 'xkcd:scarlet',
                                       Ti_mlit_max: 'xkcd:dark green'}
else:
    plot_mantle_lithosphere_particles = False
    cond_mantle_lithosphere2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

if(lower_crust_code in particles_layers):
    cond_lower_crust2plot, cond_lower_crust, Ti_crust_max, Ti_crust_mid, Ti_crust_min, n_lower_crust = take_three_particles([lower_crust_code],
                                                                                               particles_layers, T_initial)

    print(f"N crustal particles: {n_lower_crust}")
    plot_lower_crust_particles = True

else:
    plot_lower_crust_particles = False
    cond_lower_crust2plot = np.arange(0, n, 1) == np.arange(0,n,1) + 1

# print(particles_layers)
############################################################################################################################
# Plotting
plot_colorbar = True
h_air = 40.0

# t0 = dataset.time[0]
# t1 = dataset.time[1]
# dt = int(t1 - t0)

# start = int(t0)
# end = int(dataset.time.size - 1)
# step = 5

# start = 4
# end = 5
# step = 1

start = 0
end = int(trackdataset.time.size)
step = 1

make_videos = True
# make_videos = False

make_gifs = True
# make_gifs = False

zip_files = True
# zip_files = False

print("Generating frames...")
linewidth = 0.85
markersize = 8
color_lower_crust='xkcd:brown'

color_incremental_melt = 'xkcd:bright pink'
color_depleted_mantle='xkcd:bright purple'
# topo_from_density = False
topo_from_density = True

plot_other_particles = True
# plot_other_particles = False

# time = time[::-1]
with pymp.Parallel() as p:
    for i in p.range(start, end, step):
        data = dataset.isel(time=i)

        for prop in properties:
            fig, axs = plt.subplots(1, 2, figsize=(12, 3), constrained_layout=True, gridspec_kw={'width_ratios': [1, 0.4]})
            
            current_time = float(data.time.values)
            xlims = [0, float(data.lx) / 1.0e3]
            # ylims = [-float(data.lz) / 1.0e3 + 40, 40]
            ylims = [-150, 40]
            # ylims = [-400, 40]

            plot_property(data, prop, xlims, ylims, model_path,
                        fig,
                        axs[0],
                        plot_isotherms = plot_isotherms,
                        isotherms = [500, 600, 700, 1300],
                        topo_from_density=topo_from_density,
                        plot_colorbar=plot_colorbar,
                        bbox_to_anchor=(0.85,#horizontal position respective to parent_bbox or "loc" position
                                        0.20,# vertical position
                                        0.12,# width
                                        0.35),
                        plot_melt = plot_melt,
                        color_incremental_melt = color_incremental_melt,
                        color_depleted_mantle = color_depleted_mantle
                        )
            
            for particle, particle_layer in zip(range(n), particles_layers):
                #Plot particles in prop subplot
                if((plot_lower_crust_particles == True) & (particle_layer == lower_crust_code)):
                    if(cond_lower_crust2plot[particle] == True):
                        axs[0].plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air, '.', color=color_lower_crust, markersize=markersize-2, zorder=61)
                        axs[1].plot(T[i, particle], P[i, particle], '.', color=color_lower_crust, markersize=markersize, zorder=60) #current PTt point
                        axs[1].plot(T[:i, particle], P[:i, particle], '-', color=color_lower_crust, linewidth=linewidth, alpha=1.0, zorder=60) #PTt path
                        #plotting points at each 5 Myr
                        for j in np.arange(0, current_time, 5):
                            idx = find_nearest(time, j)
                            axs[1].plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=2, zorder=60)


                if((plot_mantle_lithosphere_particles == True) & ((particle_layer == mantle_lithosphere1_code) | (particle_layer == mantle_lithosphere2_code))): #lithospheric mantle particles

                    if(cond_mantle_lithosphere2plot[particle]==True):
                        # print(f"Particle: {particle}, Layer: {particle_layer}, T_initial: {T_initial[particle]}")
                        axs[0].plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air,
                                    dict_mantle_lithosphere_markers[T_initial[particle]],
                                    color=dict_mantle_lithosphere_colors[T_initial[particle]],
                                    markersize=markersize-2, zorder=61)

                        axs[1].plot(T[i, particle], P[i, particle],
                                    dict_mantle_lithosphere_markers[T_initial[particle]],
                                    color=dict_mantle_lithosphere_colors[T_initial[particle]], markersize=10, zorder=61) #current PTt point

                        axs[1].plot(T[:i, particle], P[:i, particle], '-', color=dict_mantle_lithosphere_colors[T_initial[particle]], linewidth=linewidth, alpha=0.8, zorder=61) #PTt path
                        #plotting points at each 5 Myr
                        
                        for j in np.arange(0, current_time, 5):
                            idx = find_nearest(time, j)
                            axs[1].plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=2, zorder=60)
                    else:
                        if(plot_other_particles == True): #plotting the other lithospheric mantle particles instead only 3
                            axs[0].plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air, '.', color='xkcd:black', markersize=markersize-6, zorder=60)
                            axs[1].plot(T[i, particle], P[i, particle],
                                        '.', color='xkcd:black',
                                        markersize=int(markersize/2), zorder=60) #current PTt point
                            axs[1].plot(T[:i, particle], P[:i, particle],
                                        '-', color='xkcd:black',
                                        linewidth=0.1, alpha = 1.0, zorder=60) #PTt path
                                #plotting points at each 5 Myr
                            for j in np.arange(0, current_time, 5):
                                idx = find_nearest(time, j)
                                axs[1].plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=0.7, zorder=60)

                if((plot_asthenosphere_particles == True) & (particle_layer == asthenosphere_code)):
                    if(cond_ast2plot[particle] == True):
                        axs[0].plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air, '.', color=ast_color, markersize=markersize-6, zorder=60)
                        axs[1].plot(T[i, particle], P[i, particle], '.', color=ast_color, markersize=int(markersize/2), zorder=60)
                        axs[1].plot(T[:i, particle], P[:i, particle], '-', color=ast_color, linewidth=linewidth-1, alpha=1.0, zorder=60) #PTt path
                        #plotting points at each 5 Myr
                        for j in np.arange(0, current_time, 5):
                            idx = find_nearest(time, j)
                            axs[1].plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=0.5, zorder=59)

            # Setting plot details
            fsize = 14
            axs[0].set_xlabel('Distance [km]', fontsize=fsize)
            axs[0].set_ylabel('Depth [km]', fontsize=fsize)
            axs[0].tick_params(axis='both', labelsize=fsize)

            axs[1].set_xlim([0, 1500])
            ylims = np.array([0, 4000])
            axs[1].set_ylim(ylims)
            axs[1].set_xlabel(r'Temperature [$^{\circ}$C]', fontsize=fsize)
            axs[1].set_ylabel('Pressure [MPa]', fontsize=fsize)
            # axs[1].yaxis.set_label_position("right")
            # axs[1].tick_params(axis='y', labelright=True, labelleft=False, labelsize=fsize)
            
            axs[1].tick_params(axis='both', labelsize=fsize-2)
            axs[1].grid('-k', alpha=0.7)

            #creating depth axis to PTt plot
            ax1 = axs[1].twinx()
            ax1.set_ylim(ylims/30)
            # ax1.tick_params(axis='y', labelright=False, labelleft=True, labelsize=fsize)
            ax1.set_ylabel('Depth [km]', fontsize=fsize)
            ax1.tick_params(axis='y', labelsize=fsize-2)
            # nticks = len(axs[1].get_yticks())
            # ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
            # ax1.yaxis.set_label_position("left")

            if(plot_melt):
                #plotting melt legend
                text_fsize = 12
                axs[0].text(0.01, 0.90, r'Melt Fraction $\left(\frac{\partial \phi}{\partial t}\right)$', color='xkcd:bright pink', fontsize=text_fsize, transform=axs[0].transAxes, zorder=60)
                axs[0].text(0.01, 0.80, r'Depleted Mantle ($\phi$)', color='xkcd:bright purple', fontsize=text_fsize, transform=axs[0].transAxes, zorder=60)

                figname = f"{model_name}_{prop}_and_PTt_MeltFrac_{str(int(data.step)).zfill(6)}.png"
            else:
                figname = f"{model_name}_{prop}_and_PTt_{str(int(data.step)).zfill(6)}.png"
            fig.savefig(f"_output/{figname}", dpi=300)
            plt.close('all')

        del data
        gc.collect()

print("Done!")

##############################################################################################################################################################################
if(make_videos):
    print("Generating videos...")

    fps = 24
    for prop in properties:
        videoname = f'{model_path}/_output/{model_name}_{prop}_and_PTt'

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
        gifname = f'{model_path}/_output/{model_name}_{prop}_and_PTt'

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
    os.chdir(f'{model_path}')