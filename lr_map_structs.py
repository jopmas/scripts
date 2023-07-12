import os
import glob
import numpy as np
import scipy as sp
import pandas as pd
import astropy.units as u
import astropy.table as tb
from astropy.table import QTable
from scipy.interpolate import interp1d

##############################################################################
"""
This code creates a .fits table containing the amplitude of the
rift flanks and the topographic evolution over time for a simulated scenario.

Run this script inside the diretory of a simulated scenario.

"""

path = os.getcwd().split('/')

#################
#   FUNCTIONS   #
#################
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

def read_rho(cont, Nx, Nz):
	filename = "density_" + str(cont) + ".txt"

	Rho = np.loadtxt(filename, skiprows=2, unpack=True,
					comments="P")
	Rho = np.reshape(Rho, (Nz, Nx))

	return Rho

def read_tempo(cont):
	filename = "time_" + str(cont) + ".txt"

	time = np.loadtxt(filename, usecols=2, max_rows=1)
	time = str(round(time/1.0E6, 2))

	return time

def map_struct(x_side, topo_side, scarp_top, var_max, x_scarp_tops, scarp_tops, scarp_lengths):

	#Scarp must be higher than background variation
	if(scarp_top<=var_max):

		scarp_tops = np.append(scarp_tops, round(scarp_top*1,2))
		# scarp_tops = np.append(scarp_tops, round(var_max,2))

		if(float(tempo)==0):
			x_scarp_tops = np.append(x_scarp_tops, x_side[-1])
			scarp_length = 0.0
			scarp_lengths = np.append(scarp_lengths, scarp_length)
			return x_scarp_tops, scarp_tops, scarp_lengths

		else:
			x_scarp_top = x_scarp_tops[-1]
			x_scarp_tops = np.append(x_scarp_tops, x_scarp_top)

			scarp_length = scarp_lengths[-1]*0.0
			scarp_lengths = np.append(scarp_lengths, scarp_length)
			return x_scarp_tops, scarp_tops, scarp_lengths


	#Find the Index of scarp top #find_peaks take the 1st in case of a plateu
	idx_scarp_top = np.where(topo_side == scarp_top)[0][-1]

	x_scarp_top = x_side[idx_scarp_top] #scarp position
	x_scarp_tops = np.append(x_scarp_tops, x_scarp_top)
	scarp_tops = np.append(scarp_tops, round(scarp_top,6))

	#Find the left base of the scarp
	cond = x_side <= x_side[idx_scarp_top]

	if(len(np.where(topo_side[cond]<=0)[0])==0):#case if there are no value <= 0 on the left side
		topo_half_left_scarp = topo_side[cond]
		x_half_left_scarp = x_side[cond]

		topo_min_half_left_scarp = np.min(topo_half_left_scarp)
		idx_left_base = np.where(topo_half_left_scarp==topo_min_half_left_scarp)[0][0]
		x_left_base = x_half_left_scarp[idx_left_base]
	else:
		topo_half_left_scarp = topo_side[cond]
		x_half_left_scarp = x_side[cond]

		below0_left_side_scarp = np.where(topo_half_left_scarp <= 0)[0]
		idx_left_base = below0_left_side_scarp[-1] #the last is the nearest to the scarp
		x_left_base = x_half_left_scarp[idx_left_base]

	#Find the right base of the scarp
	cond1 = x_side >= x_side[idx_scarp_top]
	if(len(np.where(topo_side[cond1]<=0)[0])==0): #case if there are no value <= 0 on the left side
		topo_half_right_scarp = topo_side[cond1]
		x_half_right_scarp = x_side[cond1]

		topo_min_half_right_scarp = np.min(topo_half_right_scarp)
		idx_right_base = np.where(topo_half_right_scarp==topo_min_half_right_scarp)[0][0]
		x_right_base = x_half_right_scarp[idx_right_base]

		#print(tempo, topo_half_right_scarp[idx_topo_min_half_right_scarp],
		#x_half_right_scarp[idx_right_base])
	else:
		topo_half_right_scarp = topo_side[cond1]
		x_half_right_scarp = x_side[cond1]

		below0_right_side_scarp = np.where(topo_half_right_scarp <= 0)[0]
		idx_right_base = below0_right_side_scarp[0]
		x_right_base = x_half_right_scarp[idx_right_base]


	scarp_length = x_right_base - x_left_base
	scarp_lengths = np.append(scarp_lengths, scarp_length)

	#plt.plot(x_left[idx_scarp_top], topo_side[idx_scarp_top], 'x', color=(1-i/total_curves,0,i/total_curves))

	return x_scarp_tops, scarp_tops, scarp_lengths

def read_density(step, Nx, Nz):
    '''
    Read density data from density_step.txt to extract interfaces
    '''

    Rho = np.loadtxt("density_"+str(step)+".txt",skiprows=2, unpack=True, comments="P")
    Rho = np.reshape(Rho, (Nz, Nx))

    return Rho

def extract_interface(z, Z, Nx, Rhoi, rho):
    #Extract interface according to a given rho
    topo_aux = []

    for j in np.arange(Nx):
        topoi = interp1d(z, Rhoi[:,j])
        idx = (np.abs(topoi(Z)-rho)).argmin()
        topo = Z[idx]
        topo_aux = np.append(topo_aux, topo)
        
    return topo_aux

#################
#     MAIN      #
#################

Nx, Nz, Lx, Lz = read_params()
x = np.linspace(0, Lx/1000.0, Nx)
z = np.linspace(0, Lz/1000.0, Nz)
xi = np.linspace(Lx/1000.0, 0, 8001)
zi = np.linspace(Lz/1000.0, 0, 8001)
Z = zi
h_air = 40.
#Sort list of Tempo_*.txt files
ts = sorted(glob.glob("time_*.txt"), key=os.path.getmtime)

total_curves = len(ts)
n_curves = round(total_curves//2)*2 #5

#Find the print_step os scenario
p1 = int(ts[0][5:-4])
p2 = int(ts[1][5:-4])
dpasso = p2 - p1

#Interface to map
rho = 1300.0

#Select the region in x to correct the topography in z using z_mean
#condx = (x >= 400) & (x <= 600)
condx = (x >= 100) & (x <= 400)

#Left and right - Check the videos to specify these limits
cond_left = (x>=400) & (x<800)
cond_right = (x>=800) & (x<1600)

x_left = x[cond_left]
x_right = x[cond_right]

factor = 1.0

tempos = []
Rhos = []
topos = []
ucrusts = []
lcrusts = []
lithos = []
#Left_side
scarp_tops_l = []
x_scarp_tops_l = []
scarp_lengths_l = []

#Right_side
scarp_tops_r = []
x_scarp_tops_r = []
scarp_lengths_r = []
steps = []
z_mean=40.0
#Main Loop
for i in np.linspace(0, total_curves-1, n_curves, dtype='int'):
	# #Read data
	cont = i * dpasso
	steps.append(cont)

	#read time
	tempo = read_tempo(i*(dpasso))
	tempos = np.append(tempos, float(tempo))
	print(tempo)

	#read topographic data
	fname = 'sp_surface_global_' + str(i*dpasso) + '.txt'
	topo = np.loadtxt(fname, unpack=True, skiprows=2, comments='P')/1.0E3

	#correct the air layer
	topo = topo+h_air
	mean = np.mean(topo[condx])
	topoi = topo - np.abs(mean)
	topos.append(topoi)

	#Take the maximum amplitude variation in the background at t=0
	if(cont==0):
		maximo = np.max(topoi)
		minimo = np.min(topoi)
		var_max = factor*np.abs(np.max(topoi) - np.min(topoi))


	#read density file
	Rhoi = read_density(cont, Nx, Nz)

	#map uper crust base topography
	ucrusti = extract_interface(z, Z, Nx, Rhoi, 2750)
	ucrusti -= np.abs(z_mean)
	ucrusti = -1*ucrusti
	ucrusts.append(ucrusti)

	#map lower crust base topography
	lcrusti = extract_interface(z, Z, Nx, Rhoi, 2900)
	lcrusti -= np.abs(z_mean)
	lcrusti = -1*lcrusti
	lcrusts.append(lcrusti)

	#map lithosphere base topography
	lithoi = extract_interface(z, Z, Nx, Rhoi, 3365)
	lithoi -= np.abs(z_mean)
	lithoi = -1*lithoi
	lithos.append(lithoi)

	#Map Structures

	#Left Side
	#Selec area where the left escarpment develops km
	topo_left = topoi[cond_left]
	#Right Side
	topo_right = topoi[cond_right]

	#Find scarp top value inside de selected area km
	scarp_top_l = np.max(topo_left)
	scarp_top_r = np.max(topo_right)

	side = ['left', 'right']
	x_scarp_tops_l, scarp_tops_l, scarp_lengths_l = map_struct(x_left, topo_left, scarp_top_l,
																var_max, x_scarp_tops_l,
																scarp_tops_l, scarp_lengths_l)

	x_scarp_tops_r, scarp_tops_r, scarp_lengths_r = map_struct(x_right, topo_right, scarp_top_r,
																var_max, x_scarp_tops_r,
																scarp_tops_r, scarp_lengths_r)
	print(tempo)

#Creating a file with the data

#Creating a file with the data
scarp_props_l = {'Time':tempos, 'Position':x_scarp_tops_l, 'Amplitude':scarp_tops_l,
              'Wavelength':scarp_lengths_l}
scarp_props_r = {'Time':tempos, 'Position':x_scarp_tops_r, 'Amplitude':scarp_tops_r,
              'Wavelength':scarp_lengths_r}

scarp_data_l = pd.DataFrame(data=scarp_props_l)
scarp_data_r = pd.DataFrame(data=scarp_props_r)

filename_l = path[-1] + "_left_scarp_data.txt"
filename_r = path[-1] + "_right_scarp_data.txt"

scarp_data_l['Amplitude'] = scarp_data_l['Amplitude'].map(lambda x: '%.6f' % x if not pd.isna(x) else '')
scarp_data_l.to_csv(filename_l, sep='\t', header=True, index=False, float_format='%.2f')

scarp_data_r['Amplitude'] = scarp_data_r['Amplitude'].map(lambda x: '%.6f' % x if not pd.isna(x) else '')
scarp_data_r.to_csv(filename_r, sep='\t', header=True, index=False, float_format='%.2f')



x_scarp_tops_l = np.asarray(x_scarp_tops_l, dtype=float)
scarp_tops_l = np.asarray(scarp_tops_l, dtype=float)
x_scarp_tops_r = np.asarray(x_scarp_tops_r, dtype=float)
scarp_tops_r = np.asarray(scarp_tops_r, dtype=float)
topos = np.asarray(topos, dtype=float)
ucrusts = np.asarray(ucrusts, dtype=float)
lcrusts = np.asarray(lcrusts, dtype=float)
lithos = np.asarray(lithos, dtype=float)

x_scarp_tops_l.reshape(n_curves, 1)
scarp_tops_l.reshape(n_curves, 1)
x_scarp_tops_r.reshape(n_curves, 1)
scarp_tops_r.reshape(n_curves, 1)
topos.reshape((n_curves, Nx))
ucrusts.reshape((n_curves, Nx))
lcrusts.reshape((n_curves, Nx))
lithos.reshape((n_curves, Nx))

tab = tb.Table({'Steps':steps,
				'Time':tempos,
				'Position_left':x_scarp_tops_l,
				'Amplitude_left':scarp_tops_l,
				'Position_right':x_scarp_tops_r,
				'Amplitude_right':scarp_tops_r,
				'Topography':topos,
				'UpC':ucrusts,
				'LwC':lcrusts,
				'Litho':lithos
              })
tab['Steps'] = tab['Steps'].astype('int')
fname = path[-1] + '_lr_escarpment_data.fits'
tab.write(fname, overwrite=True)



