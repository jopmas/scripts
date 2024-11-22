import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import matplotlib.collections as mcollections
import xarray as xr

def find_nearest(array, value):
    '''Return the index in array nearest to a given value.
    
    Parameters
    ----------
    
    array: array_like
        1D array used to find the index
        
    value: float
        Value to be seached
    '''
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

argv = sys.argv

model_path = os.getcwd() # Get local file
model_name = model_path.split('/')[-1]
output_path = '_output'
print(f"Model name: {model_name}\n")
print(f"Model path: {model_path}\n")
print(f"Output path: {output_path}\n")

# Tdataset = xr.open_dataset("_output_temperature.nc")
# Pdataset = xr.open_dataset("_output_pressure.nc")
trackdataset = xr.open_dataset("_track_xzPT_all_steps.nc")
x = trackdataset.xtrack.values[::-1]
z = trackdataset.ztrack.values[::-1]
P = trackdataset.ptrack.values[::-1]
T = trackdataset.ttrack.values[::-1]
time = trackdataset.time.values[::-1]
steps = trackdataset.step.values[::-1]

# if len(argv)>1:
#     begin = int(sys.argv[1])
#     end = int(sys.argv[2])
#     d_step = int(sys.argv[3])

#     idx_step_initial = find_nearest(steps, begin)
#     idx_step_final = find_nearest(steps, end)
#     didx = d_step//(steps[1] - steps[0])

#     x = trackdataset.xtrack.values[idx_step_initial:idx_step_final:didx]
#     z = trackdataset.ztrack.values[idx_step_initial:idx_step_final:didx]
#     P = trackdataset.ptrack.values[idx_step_initial:idx_step_final:didx]
#     T = trackdataset.ttrack.values[idx_step_initial:idx_step_final:didx]
#     time = trackdataset.time.values[idx_step_initial:idx_step_final:didx]
#     steps = trackdataset.step.values[idx_step_initial:idx_step_final:didx]

# else:
#     step_initial = steps[0]
#     step_final = steps[-1] 
#     d_step = steps[1] - steps[0]

# x=[]
# z=[]
# P=[]
# T=[]

# for cont in range(step_initial, step_final + d_step, d_step): 
#     xp, zp, Pp, Tp = np.loadtxt(f"track_xzPT_step_{str(cont).zfill(6)}.txt",unpack=True)
#     n = np.size(xp)

#     x = np.append(x,xp)
#     z = np.append(z,zp)
#     P = np.append(P,Pp)
#     T = np.append(T,Tp)

n = int(trackdataset.ntracked.values)
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

dt = 5 #Myr

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

    #plotting points at each 5 Myr
    for j in np.arange(0, time.max()+dt, dt):
        idx = find_nearest(time, j)
        if(j==0):
            ax.plot(T[idx,i], P[idx,i], '*', color='xkcd:blue', markersize=10.0)
        else:
            ax.plot(T[idx,i], P[idx,i], 'o', color='xkcd:blue', markersize=2.0)

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



