import os
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axisartist
import scipy as sp
import glob
import astropy.units as u
import astropy.table as tb

from matplotlib.ticker import FormatStrFormatter
from shutil import copyfile
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from astropy.table import QTable

path = os.getcwd().split('/')

machine_path = '/' + path[1] + '/' + path[2]

plt.style.use(machine_path+'/Doutorado/cenarios/mandyoc/scripts/des.mplstyle')

def read_params():
	'''
	Read Nx, Nz, Lx, Lz from param.txt
	'''
	with open("param.txt","r") as f:
		line = f.readline()
		line = line.split() #split by space a string to a list of strings
		Nx = int(line[-1])

		line = f.readline()
		line = line.split()
		Nz = int(line[-1])

		line = f.readline()
		line = line.split()
		Lx = float(line[-1])

		line = f.readline()
		line = line.split()
		Lz = float(line[-1])

	return Nx, Nz, Lx, Lz

def trim_axs(axs, N):
	"""little helper to massage the axs list to have correct length..."""
	axs = axs.flat
	for ax in axs[N:]:
	    ax.remove()
	return axs[:N]

def read_density(step, Nx, Nz):
	'''
	Read density data from density_step.txt to extract interfaces
	'''

	Rho = np.loadtxt("density_"+str(step)+".txt",skiprows=2, unpack=True, comments="P")
	Rho = np.reshape(Rho, (Nz, Nx))

	return Rho

def read_data(prop, step, Nz, Nx):
	'''
	Read and process data according to parameters
	'''

	#build filename
	filename = prop + "_" + str(step) + ".txt"

	data = np.loadtxt(filename, skiprows=2, unpack=True, comments="P")
	data = np.reshape(data, (Nz, Nx))

	return data

def extract_interface(z, Z, Nx, Datai, interface):
	#Extract the depth of a given interface 
	topo_aux = []

	for j in np.arange(Nx):
		topoi = interp1d(z, Datai[:,j])
		idx = (np.abs(topoi(Z)-interface)).argmin()
		topo = Z[idx]
		topo_aux = np.append(topo_aux, topo)

	return topo_aux


Nx, Nz, Lx, Lz = read_params()
scenario = path[-1]
#Used to map interface
z = np.linspace(Lz/1000.0, 0, Nz)
Z = np.linspace(Lz/1000.0, 0, 8001) #zi
#read time_*.txt files
steps = sorted(glob.glob("time_*.txt"), key=os.path.getmtime)
step_initial = int(steps[0].split('/')[-1][5:-4])
t0 = int(steps[0].split('/')[-1][5:-4])
t1 = int(steps[1].split('/')[-1][5:-4])
d_step = (t1 - t0)*1
step_final = int(steps[-1].split('/')[-1][5:-4])
nrows=len(range(step_initial, step_final+1, d_step))

prop = ['temperature']

times = []
steps1 = []
interfs = [] #list of interfaces to be mapped
isots_evolution = []
for step in range(step_initial, step_final+1, d_step):

	time_fname = 'time_' + str(step) + '.txt'
	time = np.loadtxt(time_fname, usecols=2, max_rows=1)
	time = round(time/1.0E6, 2)
	times.append(time)

	isots_evolution.append(time)
	isots_evolution.append(step)

	steps1.append(step)
	time = str(time)
	#     print(time, step)
	#read prop data
	Datai = read_data(prop[0], step, Nz, Nx)
	interfaces = [0, 500, 800, 1000, 1300, 1400, 1500, 1600, 1700] # Moho, LAB and other temperatures °C

	for interface in interfaces:
	    interf = extract_interface(z, Z, Nx, Datai, interface)
	    interf_mean = round(np.mean(interf), 2)
	    interfs.append(interf_mean)
	    isots_evolution.append(interf_mean)

	print(time, step)

isots_evolution = np.asarray(isots_evolution)
isots_evolution = isots_evolution.reshape(nrows, len(interfaces)+2)# +2 is to add step and time

np.savetxt(scenario + '_isotherms_evolution.txt', isots_evolution)

#plot data
print('Plot data')
plt.close()

label_fsize = 18
plt.rc('xtick', labelsize=label_fsize)
plt.rc('ytick', labelsize=label_fsize)

fig, ax = plt.subplots(1, 1, figsize=(12,8), sharex=True, sharey=True, constrained_layout=True)

#read_data
#time, step, 0, 500, 800, 1000, 1300, 1400, 1500, 1600, 1700
times, isot0, isot500, mohos, isot1000, labs = np.loadtxt(scenario + '_isotherms_evolution.txt', unpack=True,
                                                          usecols=(0,2,3,4,5,6))

ax.plot(times, isot0-40, '-', label='0 °C')
ax.plot(times, isot500-40, '-', label='500 °C')
ax.plot(times, mohos-40, '-', label='800 °C')
ax.plot(times, isot1000-40, '-', label='1000 °C')
ax.plot(times, labs-40, '-', label='LAB (1300 °C)')

#set plot details
ax.set_xlim([0,500])
ax.set_xticks(np.linspace(0,1000,11))
ax.set_yticks(np.linspace(-50,650,15))
ax.set_ylim([660, -50])
ax.grid(':k', alpha=0.7)
ax.set_xlabel('Time (Myr)', fontsize=label_fsize)
ax.set_ylabel('Depth (km)', fontsize=label_fsize)
ax.legend(loc='best')

scenario = path[-1]
figname = scenario + '_isotherms_evolution.png'

plt.savefig(figname, dpi=300)