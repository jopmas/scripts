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
from matplotlib.colors import ListedColormap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)


matplotlib.use('agg')

path = os.getcwd().split('/')
machine_path = f'/{path[1]}/{path[2]}' #cat the /home/user/ or /Users/user from system using path

path_to_functions = f"{machine_path}/opt/scripts"
sys.path.append(os.path.abspath(path_to_functions))

if '' in sys.path:
    sys.path.remove('')
from functions.mandyocIO import read_datasets, change_dataset, plot_property, plot_tracked_particles, plot_ptt_paths_three_particles, plot_tracked_particles_depth_coded, plot_ptt_paths_depth_coded_frame, find_nearest, _extract_interface

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
# plot_melt = True
plot_melt = False
plot_particles=False

# plot_three_particles = True
plot_three_particles = False

plot_depth_coded = True
# plot_depth_coded = False

# plot_temperature_coded = True
# plot_temperature_coded = False

if("4" in model_name and "0" in model_name):
    hcrust = 40.0e3 #m
else:
    hcrust = 35.0e3 #m

if(plot_isotherms or plot_melt):
    clean_plot=False
else:
    clean_plot = True

plot_type = 'no_particles'
if(plot_three_particles):
    plot_type = 'three_particles'
if(plot_depth_coded):
    plot_type = 'depth_coded'
# if(plot_temperature_coded):
#     plot_type = 'temperature_coded'
print(f"Plot type: {plot_type}")

datasets = [#Properties from mandyoc. Comment/uncomment to select properties of the dataset
            'density',
            'radiogenic_heat',
            'pressure',
            'strain',
            'strain_rate',### Read ascii outputs and save them as xarray.Datasets,
            # 'surface',
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

# if ('surface' not in properties): #used to plot air/curst interface
#         properties.append('surface')
#         new_datasets = change_dataset(properties, datasets)
#         to_remove.append('surface')
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
print(dataset.keys())
# Normalize velocity values
if ("velocity_x" and "velocity_z") in dataset.data_vars:
    v_max = np.max((dataset.velocity_x**2 + dataset.velocity_z**2)**(0.5))    
    dataset.velocity_x[:] = dataset.velocity_x[:] / v_max
    dataset.velocity_z[:] = dataset.velocity_z[:] / v_max

if ('lithology' in properties):
    lithology_dataset = xr.open_dataset(f"{model_path}/_lithology.nc")

#########################################
# Get domain and particles informations #
#########################################

Nx = int(dataset.nx)
Nz = int(dataset.nz)
Lx = float(dataset.lx)
Lz = float(dataset.lz)

x = np.linspace(0, Lx/1000.0, Nx)
z = np.linspace(-Lz/1000.0, 0, Nz)
xx, zz  = np.meshgrid(x, z)

trackdataset = xr.open_dataset("_track_xzPT_all_steps.nc")
x_track = trackdataset.xtrack.values[::-1]
z_track = trackdataset.ztrack.values[::-1]
P = trackdataset.ptrack.values[::-1]
T = trackdataset.ttrack.values[::-1]
time = trackdataset.time.values[::-1]
# print(time)
steps_model = trackdataset.step.values[::-1]
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
P = np.reshape(P,(steps,n))/1.0e3 #GPa
T = np.reshape(T,(steps,n))

# if(plot_temperature_coded):
#     Tmaxs = np.max(T, axis=0)
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
    plot_asthenosphere_particles = True
else:
    plot_asthenosphere_particles = False
    cond_ast2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

if((mantle_lithosphere1_code in particles_layers) | (mantle_lithosphere2_code in particles_layers)):
    plot_mantle_lithosphere_particles = True
else:
    plot_mantle_lithosphere_particles = False
    cond_mantle_lithosphere2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

if(lower_crust_code in particles_layers):
    plot_lower_crust_particles = False

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

color_lower_crust='xkcd:brown'

color_incremental_melt = 'xkcd:bright pink'
color_depleted_mantle='xkcd:bright purple'
# topo_from_density = False
topo_from_density = True

plot_other_particles = True
# plot_other_particles = False

linewidth = 0.1
markersize = 4
line_alpha = 1.0
# color_crust='xkcd:grey'

# color_incremental_melt = 'xkcd:bright pink'
# color_depleted_mantle='xkcd:dark grey'

cr = 255.
color_air = (1.,1.,1.) # 5
color_bas = (250./cr,50./cr,50./cr) # 4
color_uc = (228./cr,156./cr,124./cr) # 3
color_lc = (240./cr,209./cr,188./cr) # 2
color_lit = (155./cr,194./cr,155./cr) # 1
color_ast = (207./cr,226./cr,205./cr) # 0


colors = [color_ast,
          color_lit,
          color_lc,
          color_uc,
        #   color_bas,
          color_air]

#Creating a custom colormap according to the list of colors defined above.
# This colormap will be used to plot the lithology mesh, where each lithology type is represented by a specific color.

cmap = ListedColormap(colors)

# time = time[::-1]
with pymp.Parallel() as p:
    for i in p.range(start, end-step, step):
        data = dataset.isel(time=i)
        for prop in properties:
            fig, axs = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True, gridspec_kw={'width_ratios': [1, 0.4]})
            gs = axs[0, 0].get_gridspec()

            # Remove os eixos da segunda linha
            axs[1, 0].remove()
            axs[1, 1].remove()

            # Cria um eixo ocupando toda a segunda linha
            ax3 = fig.add_subplot(gs[1, :])

            current_time = float(data.time.values)
            # xlims = [0, float(data.lx) / 1.0e3]
            # ylims = [-float(data.lz) / 1.0e3 + 40, 40]
            # ylims = [-float(data.lz) / 1.0e3, 0]
            xlims = [700, 1300]
            ylims = [-150, 40]
            axs[0,0].text(0.01, 1.035, f'{model_name}', bbox=dict(facecolor='white', edgecolor='white', alpha=0.0), fontsize = 14, zorder=52, transform=axs[0,0].transAxes)
            axs[0,0].text(0.5, 1.035, f'Time = {current_time:.2f} Myr', bbox=dict(facecolor='white', edgecolor='white', alpha=0.0), fontsize = 14, zorder=52, transform=axs[0,0].transAxes)
            if(prop != 'lithology'):
                plot_property(data, prop, xlims, ylims, model_path,
                            fig,
                            axs[0,0],
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
            else:
                data = lithology_dataset['lithology'].isel(time=i).to_numpy()[::-1,:]
                # axs[0].pcolormesh(data, cmap=cmap, vmin=0, vmax=5, alpha=1.0)
                axs[0,0].imshow(data, aspect='auto', extent=(0, Lx/1000, -Lz/1000+40, 40), cmap=cmap, vmin=0, vmax=5, alpha=1.0)            
                axs[0,0].imshow(np.log10(dataset.strain.isel(time=i)[::-1,:]), extent=(0, Lx/1000, -Lz/1000+40, 40), cmap="Greys", vmin=-0.5, vmax=0.9, alpha=0.2)
                axs[0,0].contour(xx, zz+40, dataset.temperature.isel(time=i), levels=[500, 600, 700, 800, 900, 1300], colors='r', linewidths=1.0)
                axs[0,0].set_ylim(ylims)
                axs[0,0].set_xlim(xlims)

                ax3.imshow(data, aspect='auto', extent=(0, Lx/1000, -Lz/1000+40, 40), cmap=cmap, vmin=0, vmax=5, alpha=1.0)            
                ax3.imshow(np.log10(dataset.strain.isel(time=i)[::-1,:]), extent=(0, Lx/1000, -Lz/1000+40, 40), cmap="Greys", vmin=-0.5, vmax=0.9, alpha=0.2)
                ax3.contour(xx, zz+40, dataset.temperature.isel(time=i), levels=[500, 600, 700, 800, 900, 1300], colors='r', linewidths=1.0)

                bbox_to_anchor=(0.90,#horizontal position respective to parent_bbox or "loc" position
                                0.20,# vertical position
                                0.08,# width
                                0.25)
            
                bv1 = inset_axes(ax3,
                                loc='lower right',
                                width="100%",  # respective to parent_bbox width
                                height="100%",  # respective to parent_bbox width
                                bbox_to_anchor=bbox_to_anchor,
                                bbox_transform=ax3.transAxes
                                )
                
                A = np.zeros((100, 10))

                A[:25, :] = 2700
                A[25:50, :] = 2800
                A[50:75, :] = 3300
                A[75:100, :] = 3400

                A = A[::-1, :]

                xA = np.linspace(-0.5, 0.9, 10)
                yA = np.linspace(0, 1.5, 100)

                xxA, yyA = np.meshgrid(xA, yA)
                air_threshold = 200
                bv1.contourf(
                    xxA,
                    yyA,
                    A,
                    levels=[air_threshold, 2750, 2900, 3365, 3900],
                    colors=[color_uc, color_lc, color_lit, color_ast],
                    extent=[-0.5, 0.9, 0, 1.5]
                )

                bv1.imshow(
                    xxA[::-1, :],
                    extent=[-0.5, 0.9, 0, 1.5],
                    zorder=100,
                    alpha=0.2,
                    cmap=plt.get_cmap("Greys"),
                    vmin=-0.5,
                    vmax=0.9,
                    aspect='auto'
                )

                bv1.set_yticklabels([])
                bv1.set_xlabel(r"log$(\varepsilon_{II})$", size=10)
                bv1.tick_params(axis='x', which='major', labelsize=10)
                bv1.set_xticks([-0.5, 0, 0.5])
                bv1.set_yticks([])
                bv1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            if(plot_three_particles):
                linewidth = 0.85
                markersize = 8
                plot_other_particles = True
                plot_tracked_particles(trackdataset, axs[0,0], i, plot_other_particles=plot_other_particles, color_other_particles='xkcd:black', size_other_particles=3)
                plot_ptt_paths_three_particles(trackdataset, axs[0,1], i, current_time, plot_other_particles=plot_other_particles, color_other_particles='xkcd:black')
            if(plot_depth_coded):
                # color_lower_crust = 'xkcd:brown'
                color_mlit_upper='xkcd:bright purple'#'xkcd:cerulean blue'
                color_mlit_intermediate='xkcd:dark green'#'xkcd:scarlet'
                color_mlit_lower='xkcd:bright orange'#'xkcd:dark green'
                # linewidth = 0.05
                linewidth = 0.3
                markersize = 8

                plot_tracked_particles_depth_coded(trackdataset, axs[0,0], i, hcrust=hcrust, markersize=4,
                                                   plot_lower_crust_particles=False, plot_mantle_lithosphere_particles=True, plot_asthenosphere_particles=True,
                                                   color_mlit_upper=color_mlit_upper, color_mlit_intermediate=color_mlit_intermediate, color_mlit_lower=color_mlit_lower)
                plot_ptt_paths_depth_coded_frame(trackdataset, axs[0,1], i, current_time, hcrust=hcrust, alpha_lines=0.3, markersize=markersize, linewidth=linewidth,
                                                 plot_lower_crust_particles=False, plot_mantle_lithosphere_particles=True, plot_asthenosphere_particles=True,
                                                 color_mlit_upper=color_mlit_upper, color_mlit_intermediate=color_mlit_intermediate, color_mlit_lower=color_mlit_lower,
                                                 plot_steps=False,
                                                 alpha=0.5)

            # Setting plot details
            fsize = 14
            axs[0,0].set_xlabel('Distance [km]', fontsize=fsize)
            axs[0,0].set_ylabel('Depth [km]', fontsize=fsize)
            axs[0,0].tick_params(axis='both', labelsize=fsize)

            ax3.set_xlabel('Distance [km]', fontsize=fsize)
            ax3.set_ylabel('Depth [km]', fontsize=fsize)
            ax3.tick_params(axis='both', labelsize=fsize)

            axs[0,1].set_xlim([0, 1500])
            ylims = np.array([0, 4000])/1.0e3
            axs[0,1].set_ylim(ylims)
            axs[0,1].set_xlabel(r'Temperature [$^{\circ}$C]', fontsize=fsize)
            axs[0,1].set_ylabel('Pressure [GPa]', fontsize=fsize)
            # axs[0,1].yaxis.set_label_position("right")
            # axs[0,1].tick_params(axis='y', labelright=True, labelleft=False, labelsize=fsize)
            
            axs[0,1].tick_params(axis='both', labelsize=fsize-2)
            axs[0,1].grid('-k', alpha=0.7)

            if(plot_depth_coded):
                axs[0,1].plot([-10,-10], [-10,-10], '-', color=color_mlit_upper, label='Upper\nLithospheric Mantle')
                axs[0,1].plot([-10,-10], [-10,-10], '-', color=color_mlit_intermediate, label='Intermediate\nLithospheric Mantle')
                axs[0,1].plot([-10,-10], [-10,-10], '-', color=color_mlit_lower, label='Lower\nLithospheric Mantle')
                axs[0,1].legend(loc='upper left', ncol=1, fontsize=8, handlelength=0, handletextpad=0, labelcolor='linecolor')

            #creating depth axis to PTt plot
            ax1 = axs[0,1].twinx()
            ax1.set_ylim(ylims*1000/30)
            # ax1.tick_params(axis='y', labelright=False, labelleft=True, labelsize=fsize)
            ax1.set_ylabel('Depth [km]', fontsize=fsize)
            ax1.tick_params(axis='y', labelsize=fsize-2)
            # nticks = len(axs[1].get_yticks())
            # ax1.yaxis.set_major_locator(matplotlib.ticker.LinearLocator(nticks))
            # ax1.yaxis.set_label_position("left")

            if(plot_melt):
                incremental_melt = dataset.dPhi.isel(time=i).to_numpy()[::,:]
                melt = dataset.Phi.isel(time=i).to_numpy()[::,:]

                #Plotting incremental melt
                color_incremental_melt = 'xkcd:bright pink'
                color_depleted_mantle='xkcd:purple'

                meltmin, meltmax = melt.min(), melt.max()
                dmeltmin, dmeltmax = incremental_melt.min(), incremental_melt.max()
                incremental_melt[incremental_melt == 0] = np.nan # Set zero values to NaN to avoid plotting them
                melt[melt == 0] = np.nan # Set zero values to NaN to avoid plotting them

                ax3.contourf(xx, zz+40, incremental_melt, colors=color_incremental_melt, alpha=0.4, zorder=30)
                ax3.contourf(xx, zz+40, melt, colors=color_depleted_mantle, linewidths=0.3, alpha=0.4, zorder=20)
                #plotting melt legend
                text_fsize = 12
                ax3.text(0.01, 1.035, r'Melt Fraction $\left(\frac{\partial \phi}{\partial t}\right)$', color='xkcd:bright pink', fontsize=text_fsize, transform=ax3.transAxes, zorder=60)
                ax3.text(0.21, 1.035, r'Depleted Mantle ($\phi$)', color='xkcd:bright purple', fontsize=text_fsize, transform=ax3.transAxes, zorder=60)

                figname = f"{model_name}_{plot_type}_{prop}_and_PTt_MeltFrac_{str(int(steps_model[i])).zfill(6)}.png"
            else:
                figname = f"{model_name}_{plot_type}_{prop}_and_PTt_{str(int(steps_model[i])).zfill(6)}.png"
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
        videoname = f'{model_path}/_output/{model_name}_{plot_type}_{prop}_and_PTt'

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
        gifname = f'{model_path}/_output/{model_name}_{plot_type}_{prop}_and_PTt'

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