import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import xarray as xr
import subprocess

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

step_final = Tdataset.step.values[-1] #14000
print("Final step: ", step_final)

x=[]
z=[]
id_vec=[]
layer_vec=[]

#Reading the data of final step
for rank in range(20):
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

h_air = 40.0e3
search_thickness = 5.0e3

cond = (z>-(h_air + search_thickness)) & (layer_vec<5) & (layer_vec>0) # todas as partículas nos últimos 5km e que não pertençam ao ar (cc==6)

part_selec = id_vec[cond]
layers_selec = layer_vec[cond] #get the layer numbers of the selected particles
n_tracked = np.size(part_selec)

print(f"x size: {np.size(x)}")
print(f"Number of tracked particles (n): {np.size(part_selec)}")

pressure = []
temperature = []

#Reading backwards in time the P and T data
start = int(Tdataset.time.values[0])
end = int(Tdataset.time.size - 1)
step = 1

all_vecx_track = []
all_vecz_track = []
all_present_pressure = []
all_present_temperature = []
all_steps = []
all_times = []

for i in range(end, start-step, -step):
    print(f"Step: {Tdataset.step.values[i]}")
    temperature = Tdataset.temperature[i].values.T
    pressure = Pdataset.pressure[i].values.T

    x=[]
    z=[]
    id_vec=[]
    layer_vec=[]

    #getting data from step files
    for rank in range(20):
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


    cond = (layer_vec<6) & (layer_vec>0) #lithospheric layers
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
    
    all_vecx_track.extend(vecx_track)
    all_vecz_track.extend(vecz_track)
    all_present_pressure.extend(present_pressure / 1.0E6)
    all_present_temperature.extend(present_temperature)
    all_steps.extend([Tdataset.step.values[i]] * len(vecx_track))
    all_times.extend([Tdataset.time.values[i]] * len(vecx_track))

    np.savetxt(f"track_xzPT_step_{str(Tdataset.step.values[i]).zfill(6)}.txt", np.c_[vecx_track, vecz_track, present_pressure/1.0E6, present_temperature], fmt="%.5f")
    

# Criar o xarray.Dataset com todos os dados coletados
ds = xr.Dataset(
    {
        "xtrack": (["index"], all_vecx_track),
        "ztrack": (["index"], all_vecz_track),
        "ptrack": (["index"], all_present_pressure),
        "ttrack": (["index"], all_present_temperature),
        "step": Tdataset.step.values[::-1],
        "time": Tdataset.time.values[::-1],
        "ntracked": int(n_tracked),
        "particles_layers": layers_selec
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
