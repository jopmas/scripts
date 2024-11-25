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

#take the index of particles_layers which corresponds to mlit layer: coldest, hotterst, and the one in the middle
particles_layers = trackdataset.particles_layers.values[::-1]
mlit_code = 1
crust_code = 4 #lower crust

cond_mlit = particles_layers == mlit_code
cond_crust = particles_layers == crust_code

particles_mlit = particles_layers[cond_mlit]
particles_crust = particles_layers[cond_crust]

T_initial = T[0]

T_initial_crust = T_initial[cond_crust] #initial temperature of crustal particles
T_initial_crust_sorted = np.sort(T_initial_crust)

Ti_crust_max = np.max(T_initial_crust_sorted)
mid_index = len(T_initial_crust_sorted)//2
Ti_crust_mid = T_initial_crust_sorted[mid_index]
Ti_crust_min = np.min(T_initial_crust_sorted)

cond_crust2plot = (T_initial == Ti_crust_min) | (T_initial == Ti_crust_mid) | (T_initial == Ti_crust_max)

T_initial_mlit = T_initial[cond_mlit] #initial temperature of lithospheric mantle particles
T_initial_mlit_sorted = np.sort(T_initial_mlit)

Ti_mlit_max = np.max(T_initial_mlit_sorted)
mid_index = len(T_initial_mlit_sorted)//2
Ti_mlit_mid = T_initial_mlit_sorted[mid_index]
Ti_mlit_min = np.min(T_initial_mlit_sorted)

cond_mlit2plot = (T_initial == Ti_mlit_min) | (T_initial == Ti_mlit_mid) | (T_initial == Ti_mlit_max)

dict_mlit_markers = {Ti_mlit_max: '*',
                     Ti_mlit_mid: '^',
                     Ti_mlit_min: 'D'}

plt.close()
fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)

# time = np.arange(step_initial, step_final + d_step, d_step)

dt = 5 #Myr

for i, particle_layer, crust2plot, mlit2plot in zip(range(n), particles_layers, cond_crust2plot, cond_mlit2plot):

    if(particle_layer != mlit_code): #crustal layers
        if(crust2plot == True):
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
                    ax.plot(T[idx,i], P[idx,i], '^', color='xkcd:blue', markersize=10.0)
                else:
                    ax.plot(T[idx,i], P[idx,i], 'o', color='xkcd:blue', markersize=2.0)

    if(mlit2plot == True):
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

#plot ghost points to add the legend
ax.plot(-100, -100, '^', color='xkcd:blue', markersize=10.0, label=r'Crustal Particles (0 Myr)')
ax.plot(-100, -100, '*', color='xkcd:blue', markersize=10.0, label=r'Lithospheric Mantle Particles (0 Myr)')
ax.plot(-100, -100, 'o', color='xkcd:blue', markersize=2.0, label=r'$\Delta t = 5$ Myr')

ax.legend(loc='upper left', fontsize=8)

#adding colorbar
clb = fig.colorbar(lc)
clb.set_label("Time (Myr)")

#plot details
ax.set_xlabel("Temperature ($^\circ$C)")
ax.set_ylabel("Pressure (MPa)")
ax.grid('-k', alpha=0.3)
ax.set_xlim(0, 1400)
ax.set_ylim(0, 4000)

#saving plot
figname = f"{model_name}_PTt_Paths"
fig.savefig(f"_output/{figname}.png", dpi=300)
fig.savefig(f"_output/{figname}.pdf", dpi=300)



