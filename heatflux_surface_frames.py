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

from scipy.ndimage import median_filter
from scipy.ndimage import gaussian_filter1d
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.figure import Figure
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
matplotlib.use('agg')

path = os.getcwd().split('/')
machine_path = f'/{path[1]}/{path[2]}' #cat the /home/user/ or /Users/user from system using path

path_to_functions = f"{machine_path}/opt/scripts"
sys.path.append(os.path.abspath(path_to_functions))

if '' in sys.path:
    sys.path.remove('')
from functions.mandyocIO import read_datasets, change_dataset, plot_property, find_nearest

def extract_interface2(z, Nx, field_datai, value_to_search):
    '''
    Extract interface from Rhoi according to a given density (rho)

    Parameters
    ----------
    z: array_like
        Array representing z direction.

    Z: array_like
        Array representing z direction resampled with higher resolution.

    Nx: int
        Number of points in x direction.

    field_data: array_like (Nz, Nx)
        Mandyoc field from mandyoc (e.g.: density, temperature, etc).

    value_to_search: float
        Value of density to be searched in field_data array.

    Returns
    -------
    mapped_interface: array_like
        Interface extracted from Rhoi field
    '''

    mapped_interface = []

    for j in np.arange(Nx):
        
        idx = np.argmax(field_datai[:,j] < value_to_search) #first occurrence of value to search in the array

        depth = z[idx]

        mapped_interface = np.append(mapped_interface, depth)

    return mapped_interface


def heatflux_interface(temp,interface,xi,zi,thermal_condutivity, old_mode=False, along_interface=False):
    '''
    Calculate heat flux in the interface of a given temperature field.

    Parameters
    ----------
    temp : 2d-array
        temperature.
    interface : 1d-array
        array with z depth of the interface along x.
    xi : 1d-array
        x domain.
    zi : 1d-array
        z domain.
    k : float, optional
        thermal conductivity. The default is 2.25 W/m/k (upper crust)

    Returns
    -------
    heatflux_crust : 1d-array (W/m²)
        heat flux in the given interface.
        
    [tempgrad,heatflux] : 2d-arrays
    gradient of the  (K/m)
    heat flux (W/m²)

    '''
    

    res = (zi[1]-zi[0])*1e3 #constante #km->m
    tempgrad = np.array(np.gradient(temp, res)) #Y,X - gradiente termico dT/dz e dT/dx
    if(old_mode == True):
        heatflux = -thermal_condutivity * tempgrad #respeitando variacao de densidade #2.25 multiplicação assumindo k constante (nao importa porque só vou pegar a crosta)
    
        #modtempgrad = np.sqrt(tempgrad[0]**2 + tempgrad[1]**2) #módulo do gradiente termico
        
        modheatflux = np.sqrt(heatflux[0]**2 + heatflux[1]**2) #módulo do fluxo termico, [0], [1] is to get the gradient in z, x direction!

        print(np.shape(tempgrad), np.shape(heatflux), np.shape(heatflux[0]), np.shape(modheatflux))
    
    heatflux_crust = []
    h_air = 40.0
    for i in range(len(xi)): #iterando colunas ao longo de x

        if(along_interface == True):
            depth_to_search = interface[i] #value of the line interface in that column
        else:
            depth_to_search = -12-h_air #km #isoline of 12 km depth

        if(old_mode == True): #Using grad 2D
            modhf_z = modheatflux[:,i] #flux in the column
            gradhfcrust = modhf_z[zi==depth_to_search]
        else: #Using grad 1D
            temp0 = temp[np.where(zi == depth_to_search)[0] - 0, i] # temperature at the interface of depth_to_search
            temp1 = temp[np.where(zi == depth_to_search)[0] - 1, i] # temperature at 1 index below of the interface of depth_to_search

            z0 = zi[np.where(zi == depth_to_search)[0] - 0]*1.0e3 # km -> m #depth at the interface of depth_to_search
            z1 = zi[np.where(zi == depth_to_search)[0] - 1]*1.0e3 # km -> m #depth at 1 index below of the interface of depth_to_search

            k_new = thermal_condutivity[np.where(zi == -12-40)[0]-0, i] #thermal conductivity in the at interface of depth_to_search
            
            heatflux = -k_new*(temp1 - temp0)/(z1 - z0)
            # modheatflux_grad = np.abs(heatflux_grad)
            gradhfcrust = np.sqrt(heatflux**2)

        # gradhfcrust = float(gradhfcrust)
        heatflux_crust.append(gradhfcrust)
    
    return np.array(heatflux_crust), tempgrad, heatflux

##################################################################################################################
####################################################################################################################################
model_path = os.getcwd() # Get local file
model_name = model_path.split('/')[-1]
output_path = '_output'
print(f"Model name: {model_name}\n")
print(f"Model path: {model_path}\n")
print(f"Output path: {output_path}\n")

#reading datasets
temperature_dataset = xr.open_dataset('_output_temperature.nc')
density_dataset = xr.open_dataset('_output_density.nc')
surface_dataset = xr.open_dataset('_output_surface.nc')

#set domain parameters
Nx = int(temperature_dataset.nx)
Nz = int(temperature_dataset.nz)
Lx = float(temperature_dataset.lx)
Lz = float(temperature_dataset.lz)

x = np.linspace(0, Lx/1000.0, Nx)
z = np.linspace(-Lz/1000.0, 0, Nz)
Z = np.linspace(-Lz/1000.0, 0, 8001) #zi

#create meshgrid
xx, zz = np.meshgrid(x, z)

#constants
h_air = 40.0 #km
thermal_diffusivity_coefficient = 1.0e-6 #kappa [m2/s] #0.75e-6 #default is 1.0e-6 # ok
heat_capacity = 1250 #c #ok #default is 1250 [J/kg/K] = [W.s/kg/K] # ok

#Setting instant

t0 = temperature_dataset.time[0]
t1 = temperature_dataset.time[1]
dt = int(t1 - t0)

start = int(t0)
end = int(temperature_dataset.time.size - 1)
step = 1

# start = 4
# end = 5
# step = 1

make_videos = True
# make_videos = False

make_gifs = True
# make_gifs = False

zip_files = True
# zip_files = False

with pymp.Parallel() as p:
    for i in p.range(start, end+step, step):
        fig, axs = plt.subplots(2, 1, figsize=(12, 4), constrained_layout=True, sharex=True, gridspec_kw={'height_ratios': [0.5, 1]})

        time = np.round(temperature_dataset.isel(time=i).time.values, 2)
        step = temperature_dataset.isel(time=i).step.values
        Rhoi = density_dataset.isel(time=i).density.values.T
        Temperi = temperature_dataset.isel(time=i).temperature.values.T

        #extracting topography from density field
        # rho_to_map = 200
        rho_to_map = 2700.0
        # surface_interface = _extract_interface(z, Z, Nx, Rhoi, rho_to_map) #interpoled interface
        surface_interface2 = extract_interface2(z, Nx, Rhoi, rho_to_map) #not interpolated interface

        surface_interface_from_mandyoc = surface_dataset.isel(time=i).surface.values/1.0e3 #km

        condx = (x >= 100) & (x <= 300)
        z_mean = np.mean(surface_interface_from_mandyoc[condx]) + h_air
        surface_interface_from_mandyoc = surface_interface_from_mandyoc + h_air - np.abs(z_mean)

        thermal_condutivity =  thermal_diffusivity_coefficient * heat_capacity * Rhoi #2.25 #W/m/K - condutividade termica
            
        heatflux_crust, tempgrad, heatflux = heatflux_interface(Temperi, surface_interface2, x, z,
                                                                thermal_condutivity=thermal_condutivity,
                                                                old_mode=False,
                                                                along_interface=False)

        # smooth_hf = True
        # smooth_hf = False
        # if(smooth_hf):
        #     sigma = 5
        #     heatflux_crust = gaussian_filter1d(heatflux_crust, sigma, mode='nearest')
        #     heatflux_crust_grad = gaussian_filter1d(heatflux_crust_grad, sigma, mode='nearest')

        #plotting

        axs[0].plot(x, heatflux_crust*1.0e3, 'k', lw=2)
        axs[1].plot(x, surface_interface_from_mandyoc, 'xkcd:purple', lw=2)

        fsize = 14
        axs[0].text(0.01, 1.03, f'{time} Myr', transform=axs[0].transAxes, fontsize=fsize)
        axs[1].set_xlim(0, Lx/1000.0)
        axs[0].set_ylim(0, 300)
        axs[1].set_ylim(-6, 6)
        axs[0].set_ylabel(r'q [mW/m$^2$]', fontsize=fsize)
        axs[1].set_ylabel(r'Depth [km]', fontsize=fsize)
        axs[1].set_xlabel(r'Distance [km]', fontsize=fsize)

        for ax in axs:
            ax.grid()

        figname = f'{model_name}_heatflux_and_surface_{str(int(step)).zfill(6)}.png'
        fig.savefig(f'_output/{figname}', dpi=300)
        plt.close('all')

        del Rhoi
        del Temperi
        gc.collect()

print("Done!")

##############################################################################################################################################################################
if(make_videos):
    print("Generating videos...")

    fps = 24
    
    videoname = f'{model_path}/_output/{model_name}_heatflux_and_surface'
        
    try:
        comand = f"rm {videoname}.mp4"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
        print(f"\tRemoving previous heatflux_and_surface video.")
    except:
        print(f"\tNo heatflux_and_surface video to remove.")

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
    
    gifname = f'{model_path}/_output/{model_name}_heatflux_and_surface'

    try:
        comand = f"rm {gifname}.gif"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
        print(f"\tRemoving previous heatflux_and_surface gif.")
    except:
        print(f"\tNo heatflux_and_surface gif to remove.")
    
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

print("Done!")