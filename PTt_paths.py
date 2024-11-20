import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import matplotlib.collections as mcollections
import xarray as xr

argv = sys.argv

model_path = os.getcwd() # Get local file
model_name = model_path.split('/')[-1]
output_path = '_output'
print(f"Model name: {model_name}\n")
print(f"Model path: {model_path}\n")
print(f"Output path: {output_path}\n")

Tdataset = xr.open_dataset("_output_temperature.nc")
# Pdataset = xr.open_dataset("_output_pressure.nc")


if len(argv)>1:
    step_initial = int(sys.argv[1])
    step_final = int(sys.argv[2])
    d_step = int(sys.argv[3])
else:
    step_initial = Tdataset.step.values[0]
    step_final = Tdataset.step.values[-1] 
    d_step = Tdataset.step.values[1] - Tdataset.step.values[0]

x=[]
z=[]
P=[]
T=[]

for cont in range(step_initial, step_final + d_step, d_step): 
    xp,zp,Pp,Tp = np.loadtxt(f"track_xzPT_step_{str(cont).zfill(6)}.txt",unpack=True)
    n = np.size(xp)

    x = np.append(x,xp)
    z = np.append(z,zp)
    P = np.append(P,Pp)
    T = np.append(T,Tp)


nTotal = np.size(x)
steps = nTotal//n

print(f"Number of particles: {n}, Steps: {steps}")

x = np.reshape(x,(steps,n))
z = np.reshape(z,(steps,n))
P = np.reshape(P,(steps,n))
T = np.reshape(T,(steps,n))

plt.close()
fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)

# time = np.arange(step_initial, step_final + d_step, d_step)
time = Tdataset.time.values

for i in range(n):
    #organizing the data 
    points = np.array([T[:,i], P[:,i]]).T.reshape(-1, 1, 2) #-1 tells numpy to figure out the length by itself

    segments = np.concatenate([points[:-1], points[1:]], axis=1) #creating the segments of consecutive points

    
    norm = plt.Normalize(time.min(), time.max()) #mapping the time dimenstion according to a colomap
    lc = mcollections.LineCollection(segments, cmap='inferno', norm=norm, linewidths=1.0) #mapping the segments to a colormap
    lc.set_array(time) #setting the dimension to be colorized

    # plt.plot(T[:,i], P[:,i])
    ax.add_collection(lc)
    ax.autoscale()

#adding colorbar
clb = fig.colorbar(lc)
clb.set_label("Time (Myr)")

#plot details
ax.set_xlabel("Temperature ($^\circ$C)")
ax.set_ylabel("Pressure (MPa)")
ax.grid('-k', alpha=0.3)

#saving plot
figname = f"{model_name}_PTt_Paths"
fig.savefig(f"_output/{figname}.png", dpi=300)
fig.savefig(f"_output/{figname}.pdf", dpi=300)



