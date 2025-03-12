##########################################################
# YOU MUST UNZIP THE step*.txt FILES BEFORE RUNNING THIS #
##########################################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import xarray as xr
import subprocess
import multiprocessing #needed to run pymp in mac
multiprocessing.set_start_method('fork') #needed to run pymp in mac
import pymp
import glob
#reading dataset
Tdataset = xr.open_dataset("_output_temperature.nc")
Pdataset = xr.open_dataset("_output_pressure.nc")

#getting the dimensions
Nx = int(Tdataset.nx)
Nz = int(Tdataset.nz)
Lx = float(Tdataset.lx)
Lz = float(Tdataset.lz)

print(
    "nx:", Nx, "\n",
    "nz:", Nz, "\n",
    "Lx:", Lx, "\n",
    "Lz:", Lz
)

xi = np.linspace(0,Lx,Nx)
zi = np.linspace(-Lz,0,Nz)
xx,zz = np.meshgrid(xi,zi)

dx = Lx/(Nx-1)
dz = Lz/(Nz-1)

xs = []
zs = []

verif=0

# take_middle = True
take_middle = False

take_asthenosphere = True
# take_asthenosphere = False


if(take_middle):
    instant_to_take = 20 #Myr
    print(f"Track particles until {instant_to_take} Myr")
    idx = (np.abs(Tdataset.time.values - instant_to_take)).argmin()
    step_final = Tdataset.step.values[idx]
else:
    step_final = Tdataset.step.values[-1] #14000

print("Final step: ", step_final)

x=[]
z=[]
id_vec=[]
layer_vec=[]

steps0 = sorted(glob.glob("step_0_*.txt"), key=os.path.getmtime)
ncores = len(steps0) #20
print(f"Number of cores: {ncores}")

# ncores = 16
#Reading the data of final step
for rank in range(ncores):
    file_name = f"step_{step_final}_{rank}.txt"

    if os.path.getsize(file_name)>0:
        step_rank = pd.read_csv(
            file_name,
            delimiter=" ",
            comment="P",
            skiprows=2,
            header=None,
        )

        step_rank = step_rank.to_numpy()
        x1 = step_rank[:,0]
        z1 = step_rank[:,1]
        id = step_rank[:,2]
        layer = step_rank[:,3]
        epsilom = step_rank[:,4]

        id_vec = np.append(id_vec,id)
        layer_vec = np.append(layer_vec,layer)
        x = np.append(x,x1)
        z = np.append(z,z1)


###########################################################################
#Selecting the particles to track                                         #
# All particles above search_thickness and between x_begin and x_end [km] #
# and does not belong to air (cc==6)                                      #
###########################################################################

h_air = 40.0e3
# search_thickness = 5.0e3 
search_thickness = 10.0e3
# search_thickness = 50.0e3
# search_thickness = 35.0e3
# x_begin = 0.0e3
# x_end = Lx
x_begin = 400.0e3
x_end = 1000.0e3
# x_end = 1200.0e3

asthenosphere_code = 0
mantle_lithosphere1_code = 1
seed_code = 2
mantle_lithosphere2_code = 3
lower_mantle_code = 4
upper_crust_code = 5
air_code = 6

if(take_middle):
    if(take_asthenosphere):
        cond = (z>-(h_air + search_thickness)) & (x>=x_begin) & (x<=x_end) & (layer_vec<upper_crust_code) & (layer_vec>=asthenosphere_code) #to take particles from asthenosphere and lithosphere.
    else:
        cond = (z>-(h_air + search_thickness)) & (x>=x_begin) & (x<=x_end) & (layer_vec<upper_crust_code) & (layer_vec>asthenosphere_code) #to take particles from lithosphere
    # cond = (z>-(h_air-2.0e3 + search_thickness)) & (x>=x_begin) & (x<=x_end) & (layer_vec==0)
else:
    if(take_asthenosphere):
        cond = (z>-(h_air + search_thickness)) & (x>=x_begin) & (x<=x_end) & (layer_vec<upper_crust_code) & (layer_vec>=asthenosphere_code) #to take particles from lithosphere
    else:
        cond = (z>-(h_air + search_thickness)) & (x>=x_begin) & (x<=x_end) & (layer_vec<upper_crust_code) & (layer_vec>asthenosphere_code) #to take particles from lithosphere

# cond = (z>-(h_air + search_thickness)) & (layer_vec<5) & (layer_vec>0)

part_selec = id_vec[cond]
layers_selec = layer_vec[cond] #get the layer numbers of the selected particles
n_tracked = np.size(part_selec)

print(f"x size: {np.size(x)}")
print(f"Number of tracked particles (n): {np.size(part_selec)}")

pressure = []
temperature = []

#Reading backwards in time the P and T data
start = int(Tdataset.time.values[0])
end = int(Tdataset.time[:idx+1].size) if take_middle else int(Tdataset.time.size - 1)
# end = int(Tdataset.time[:idx+15].size)
print(f'Start: {start}, End: {end}, time: {Tdataset.time.values[end]}')
step = 1

# all_vecx_track = []
# all_vecz_track = []
# all_present_pressure = []
# all_present_temperature = []
# all_steps = []
# all_times = []

#we need a shared pymp dictionary to store data.
vecx_track_evolution = pymp.shared.dict()
vecz_track_evolution = pymp.shared.dict()
present_pressure_evolution = pymp.shared.dict()
present_temperature_evolution = pymp.shared.dict()

with pymp.Parallel() as p:
    for i in p.range(end, start-step, -step):
        # print(f"Step: {Tdataset.step.values[i]}")
        temperature = Tdataset.temperature[i].values.T
        pressure = Pdataset.pressure[i].values.T

        x=[]
        z=[]
        id_vec=[]
        layer_vec=[]

        #getting data from step files
        for rank in range(ncores):
            file_name = f"step_{Tdataset.step.values[i]}_{rank}.txt"
            if os.path.getsize(file_name)>0:
                A = pd.read_csv(
                    file_name,
                    delimiter=" ",
                    comment="P",
                    # skiprows=2,
                    header=None,
                )
                A = A.to_numpy()
                x1=A[:,0]
                z1=A[:,1]
                id=A[:,2]
                layer=A[:,3]
                epsilom=A[:,4]

                id_vec = np.append(id_vec,id)
                layer_vec = np.append(layer_vec,layer)
                x = np.append(x,x1)
                z = np.append(z,z1)

        
        if(take_asthenosphere):
            cond = (layer_vec<air_code) & (layer_vec>=asthenosphere_code) #to take particles from asthenosphere and lithosphere.
        else:
            cond = (layer_vec<air_code) & (layer_vec>asthenosphere_code) #to take particles from lithosphere

        x = x[cond]
        z = z[cond]
        id_vec = id_vec[cond]
        
        #tracking x, z position
        vecx_track = []
        vecz_track = []
        for ii in range(np.size(part_selec)):
            id_selec, = np.where(id_vec==part_selec[ii])
            xaux = x[id_selec]
            zaux = z[id_selec]
            vecx_track = np.append(vecx_track,xaux)
            vecz_track = np.append(vecz_track,zaux)

        jm = ((vecx_track-0)//dx).astype(int)
        im = ((vecz_track+Lz)//dz).astype(int)

        xw = (vecx_track - xx[im,jm])/dx
        zw = (vecz_track - zz[im,jm])/dz

        present_temperature =	temperature[im,jm]*    (1.-xw)*(1.-zw)+\
                                temperature[im,jm+1]*  (xw   )*(1.-zw)+\
                                temperature[im+1,jm]*  (1.-xw)*(   zw)+\
                                temperature[im+1,jm+1]*(   xw)*(   zw)

        present_pressure =  pressure[im,jm]*    (1.-xw)*(1.-zw)+\
                            pressure[im,jm+1]*  (xw   )*(1.-zw)+\
                            pressure[im+1,jm]*  (1.-xw)*(   zw)+\
                            pressure[im+1,jm+1]*(   xw)*(   zw)
        
        vecx_track_evolution[i] = vecx_track
        vecz_track_evolution[i] = vecz_track
        present_pressure_evolution[i] = present_pressure / 1.0e6
        present_temperature_evolution[i] = present_temperature
        
        np.savetxt(f"track_xzPT_step_{str(Tdataset.step.values[i]).zfill(6)}.txt", np.c_[vecx_track, vecz_track, present_pressure/1.0E6, present_temperature], fmt="%.5f")


vecx_track_dict = dict(vecx_track_evolution)
vecz_track_dict = dict(vecz_track_evolution)
present_pressure_dict = dict(present_pressure_evolution)
present_temperature_dict = dict(present_temperature_evolution)

all_vecx_track = []
all_vecz_track = []
all_present_pressure = []
all_present_temperature = []

for i,j,k,l in zip(sorted(vecx_track_dict), sorted(vecz_track_dict), sorted(present_pressure_dict), sorted(present_temperature_dict)):
    all_vecx_track.extend(vecx_track_dict[i])
    all_vecz_track.extend(vecz_track_dict[j])
    all_present_pressure.extend(present_pressure_dict[k])
    all_present_temperature.extend(present_temperature_dict[l])

# Creatiing the xarray dataset with the tracked particles
ds = xr.Dataset(
    {
        "xtrack": (["index"], all_vecx_track[::-1]),
        "ztrack": (["index"], all_vecz_track[::-1]),
        "ptrack": (["index"], all_present_pressure[::-1]),
        "ttrack": (["index"], all_present_temperature[::-1]),
        "step": Tdataset.step[:end].values[::-1],
        "time": Tdataset.time[:end].values[::-1],
        "ntracked": int(n_tracked),
        "particles_layers": layers_selec[::-1]
    },
    coords={
        "index": np.arange(len(all_vecx_track))
    }
)

ds.to_netcdf("_track_xzPT_all_steps.nc")


# Compressing the files 
print("Zipping files")
model_path = os.getcwd() # Get local file
model_name = model_path.split('/')[-1]
subprocess.run(f"zip {model_name}.zip track*.txt", shell=True, check=True, capture_output=True, text=True)
print("Files zipped")
