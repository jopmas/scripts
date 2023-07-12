'''
This script generates frames to create videos of mandyoc specified mandyoc outputs
'''

#################
#   LIBRARIES   #
#################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from matplotlib.ticker import FormatStrFormatter
import glob
import os

############################
#   SYSTEM HOME AND USER   #
############################

path = os.getcwd().split('/')

machine_path = '/'+path[1]+'/'+path[2]
#machine_path = '/home/user'
#machine_path = '/Users/joao_lab'
#machine_path = '/home/joao_macbook'

plt.style.use(machine_path+'/Mestrado/cenarios/mandyoc/scripts_gera_inputs_imgs/des.mplstyle')

#################
#   FUNCTIONS   #
#################

def read_data(prop, step, Nx, Nz):
	'''
	Read and reshape the read data according to parameters to return a (Nx, Nz) array.

	Parameters
    ----------
    prop: str
        Property of the mandyoc outputs: temperature, density, strain, strain_rate, viscosity, heat or pressure.
        
    step: str
        Time step of numerical scenario.

    Nx: int
        Number of points in x direction.
        
    Nz: int
        Number of points in z direction.

	'''
	
	file_name = prop + "_" + str(step) + ".txt"

	A = np.loadtxt(file_name, unpack=True, comments="P", skiprows=2)
	TT = A*1.0
	TT[np.abs(TT)<1.0E-200]=0
	TT = np.reshape(TT, (Nx, Nz), order='F')
	TTT = TT[:,:]
	return TTT

def read_density(step, Nx, Nz):
	'''
    Read and reshape readed data according to parameters to return a (Nx, Nz) array.
    
    Parameters
    ---------- 
    step: str
        Time step of numerical scenario.
        
    Nx: int
        Number of points in x direction.
        
    Nz: int
        Number of points in z direction.
    '''

	Rho = np.loadtxt("density_"+str(step)+".txt",skiprows=2, unpack=True, comments="P")
	Rho = np.reshape(Rho, (Nz, Nx))

	return Rho

def read_temperature(step, Nx, Nz):
	'''
    Read temperature data from temperature_step.txt  and reshape to a (Nz, Nx) array
    
    Parameters
    ----------        
    step: str
        Time step of numerical scenario.
        
    Nx: int
        Number of points in x direction.
        
    Nz: int
        Number of points in z direction.
    '''

	Temper = np.loadtxt("temperature_"+str(step)+".txt",skiprows=2, unpack=True, comments="P")
	Temper = np.reshape(Temper, (Nz, Nx))

	return Temper

def extract_interface(z, Z, Nx, Rhoi, rho):
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
        
    Rhoi: array_like (Nz, Nx)
        Density field from mandyoc

    rho: int
		Value of density to be searched in Rhoi field
	'''

	topo_aux = []

	for j in np.arange(Nx):
		topoi = interp1d(z, Rhoi[:,j]) #return a "function" of interpolation to apply in other array
		idx = (np.abs(topoi(Z)-rho)).argmin()
		topo = Z[idx]
		topo_aux = np.append(topo_aux, topo)

	return topo_aux
		
def plot_data(Lx, Lz, Nx, Nz, xx, zz, prop, prop_label, val_minmax, data, step, time, points=False):
	'''
	Plot data from mandyoc according to a given property.

	Parameters
    ----------   
    Lx: int
        Length of x direction in meters.
        
    Lz: int
        Length of z direction in meters.

    Nx: int
        Number of points in x direction.
        
    Nz: int
        Number of points in z direction.
        
    xx: array_like
		(Nz, Nx) 2D matrix for x direction from meshgrid

    zz:
		(Nz, Nx) 2D matrix for z direction from meshgrid

    prop: str
		Property from mandyoc

    prop_label: str
		Label used inplots with colorbars

    val_minmax: list_like
		List with the lower val_minmax[0] and upper val_minmax[1] limits of colorbar, respectively

    data: array_like
		(Nz, Nx) matrix with a specified mandyoc data (e.g.: temperature field)

    step: int
    	Step of Mandyoc output files 
		
    time: float
    	Time in Myr correspondent to step
		
	points: bool
		Plot particles from step_*.txt files. Default is False.
	'''

	#Interfaces

	#The air, crust and lithopshere basis are extracted from density data
	Rhoi = read_density(step, Nx, Nz)
	interfaces=[200, 2900, 3365]
	# ax.contour(xx, zz, Rhoi, 100, levels=interfaces, colors=['black', 'black'])
	# ax.fill_between(xx[0], topo, 40, color='white', alpha=1.0)

	#The isotherms of 1300 and 800 °C are extracted from temperature data
	Temperi = read_temperature(step, Nx, Nz)
	isotherms=[550, 800, 1300]
	# isotherms=[800, 900, 1000, 1100, 1200, 1300]

	z = np.linspace(Lz / 1000.0, 0, Nz)
	Z = np.linspace(Lz / 1000.0, 0, 8001) #zi
	
	##Extract layers topography from density data

	# Air/crust interface
	density=interfaces[0] #200 kg/m3
	topo_interface = extract_interface(z, Z, Nx, Rhoi, density)
	condx = (xx[0] >= 100) & (xx[0] <= 600)
	z_mean = np.mean(topo_interface[condx])
	topo_interface -= np.abs(z_mean)
	topo_interface = -1.0*topo_interface

	#Crust base
	density=interfaces[1]#2900 kg/m3
	topo_crust = extract_interface(z, Z, Nx, Rhoi, density)
	topo_crust -= np.abs(z_mean)
	topo_crust = -1.0*topo_crust
	#ax.plot(xx[0], topo_crust, '--k')
	
	#Lithosphere base
	density=interfaces[2]#3370 kg/m3
	topo_lit = extract_interface(z, Z, Nx, Rhoi, density)
	topo_lit -= np.abs(z_mean)
	topo_lit = -1.0*topo_lit
	#ax.plot(xx[0], topo_lit, 'r')

	##Extract isotherms depths from temperature data

	#Isotherm 800 oC
	temp=isotherms[0]#800
	isot_800 = extract_interface(z, Z, Nx, Temperi, temp)
	isot_800 -= np.abs(z_mean)
	isot_800 = -1.0*isot_800
	# ax.plot(xx[0], isot_800, '-r')

	#Isotherm 1300 oC
	temp=isotherms[1]#800
	isot_1300 = extract_interface(z, Z, Nx, Rhoi, temp)
	isot_1300 -= np.abs(z_mean)
	isot_1300 = -1.0*isot_1300
	# ax.plot(xx[0], isot_1300, '-r')

	xmin = 0 #+200
	xmax = Lx / 1.0E3 #-200

	if(prop != 'lithology' and prop != 'topography'):
		plt.close()

		label_size=20
		plt.rc('xtick', labelsize = label_size)
		plt.rc('ytick', labelsize = label_size)

		fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout = True)
		
		#set plot details
		ax.set_xlabel("Distance (km)", fontsize = 14)
		ax.set_ylabel("Depth (km)", fontsize = 14)
		ax.set_xlim(xmin, xmax)
		ax.set_ylim(-(Lz / 1.0E3 - 40), 40)
		# ax.text(Lx/1.0E3-350, 25+40, str(time)+r' Myr', fontsize = 18)
		ax.text(Lx / 1.0E3 - 350, 25 + 40, ' {:01} Myr'.format(time), fontsize = 18)

		#PLOT PARTICULAS
		# if (points == True):

		# 	x  = []
		# 	z  = []
		# 	cc = []

		# 	for rank in range(4):
				
		# 		#x1,z1,c0,c1,c2,c3,c4,c5 = np.loadtxt("step_"+str(cont*dt)+"-rank_new"+str(rank)+".txt",unpack=True)
		# 		A = pd.read_csv("step_"+str(step)+"-rank_new"+str(rank)+".txt",delimiter = " ") 
		# 		AA = A.to_numpy()
		# 		x1 = AA[:,0]
		# 		z1 = AA[:,1]
		# 		c1 = AA[:,3]

		# 		cor =  (0,0,0)
		# 		cor2 = (0,0,0)
		# 		cor3 = (0,0,0)
		# 		#print(cor)
				
		# 		cc = np.append(cc,c1)
		# 		x = np.append(x,x1)
		# 		z = np.append(z,z1)
			
		# 	x = x[(cc<6)]
		# 	z = z[(cc<6)]

		# 	difere = 2
		# 	difere2 = 0.95
			
		# 	#plt.axis("equal")

		# 	ax.plot(x/1000,z/1000,"o",color=cor,markersize=0.05,mfc=cor)
		# 	#plt.plot(x/1000,z/1000,"c.",color=cor,markersize=0.3)
		# 	"""
		# 	cond = (cc>difere2) & (cc<difere)
		# 	plt.plot(x[cond]/1000,z[cond]/1000,"c.",color=cor,markersize=0.3)
		# 	cond = (cc<difere2)
		# 	plt.plot(x[cond]/1000,z[cond]/1000,"c.",color=cor3,markersize=0.3)
		# 	plt.plot(x[cc>difere]/1000,z[cc>difere]/1000,"c.",color=cor2,markersize=0.3)
		# 	"""
		# 	print("Leu os points")

			#ax.contourf(xx, zz, np.transpose(data), 100)

		if(prop != 'strain' and prop != 'pressure' and prop != 'temperature_anomaly' and prop != 'vs_anomaly'):
			#Plots that need a regular colorbar
			im = ax.imshow(np.transpose(data),
						   cmap = 'viridis',
						   origin =' lower',
						   extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40),
						   vmin = val_minmax[0],
						   vmax = val_minmax[1]
						   )

			fig.colorbar(im,
						 ax = ax,
						 orientation = 'horizontal',
						 label = prop_label,
						 fraction = 0.08
						 )

		elif(prop == 'pressure'):
			#Plot as GPa
			im = ax.imshow(np.transpose(data / 1.0E9),
						   cmap = 'viridis',
						   origin = 'lower',
						   extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40),
						   vmin = val_minmax[0],
						   vmax = val_minmax[1]
						   )
			
			fig.colorbar(im,
						 ax = ax,
					     orientation = 'horizontal',
					     label = prop_label,
					     fraction = 0.08
					     )

		elif(prop == 'temperature_anomaly'):
			#removing horizontal mean temperature
			A = data  #shape: (Nx, Nz)
			B = np.transpose(A) #shape: (Nz, Nx)
			C = np.mean(B, axis=1) #shape: 151==Nz
			D = (B.T - C) #B.T (Nx,Nz) para conseguir subtrair C

			im = ax.imshow(np.transpose(D), 
						   cmap = 'RdBu_r', 
					       origin = 'lower',
                           extent = (0 ,Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40), 
                           vmin = val_minmax[0], 
                           vmax = val_minmax[1]
                           )

			fig.colorbar(im, 
						 ax = ax, 
					     orientation = 'horizontal', 
					     label = prop_label, 
					     fraction = 0.08
					    )

		elif(prop == 'vs_anomaly'):
			#removing horizontal mean temperature
			A = data  #shape: (Nx, Nz)
			B = np.transpose(A) #shape: (Nz, Nx)
			C = np.mean(B, axis=1) #shape: 151==Nz
			D = (B.T - C) #B.T (Nx,Nz) is necessary to subtract C
			VsAn = D/(-1.0e4)

			im = ax.imshow(np.transpose(VsAn),
						   cmap = 'RdBu',
						   origin = 'lower',
                           extent = (0, Lx / 1.0E3,-Lz / 1.0E3 + 40, 0 + 40),
                           vmin = val_minmax[0],
                           vmax = val_minmax[1]
                           )

			fig.colorbar(im,
						 ax = ax,
						 orientation = 'horizontal',
						 label = prop_label,
						 fraction = 0.08
						 )
		else:
			#Plot that does not need a colorbar
			ax.imshow(np.transpose(data),
					  cmap = 'viridis',
					  origin = 'lower',
					  extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40)
					  )
	
	elif(prop == 'topography'):
		plt.close()
		
		label_size=20
		plt.rc('xtick', labelsize = label_size)
		plt.rc('ytick', labelsize = label_size)

		fig, ax = plt.subplots(1, 1, figsize = (12,6), constrained_layout = True)
		# ax.plot(xx[0], data, alpha = 1, linewidth = 2.0, color = "blueviolet")
		ax.plot(xx[0], data, alpha = 1, linewidth = 2.0, color = "blueviolet")

		# ax.text(Lx/1.0E3-350, 6.5, str(time)+r' Myr', fontsize = 22)
		ax.text(Lx / 1.0E3 - 350, 6.5, '{:01} Myr'.format(float(time)), fontsize = 22)

		xmin = 0 + 200
		xmax = Lx / 1.0E3 - 200
		ax.set_xlim(xmin, xmax)
		ax.set_ylim(-8, 8)
		ax.grid('-k', alpha = 0.5)

		ax.set_xlabel("Distance (km)", fontsize = 24)
		ax.set_ylabel("Topography (km)", fontsize = 24)

	else: #Shaded lithology plots
		#read data

		h_air = 40.0
		A = pd.read_csv(
			"density_" + str(step) + ".txt",
			delimiter = " ",
			comment = "P",
			skiprows = 2,
			header = None,
		)
		A = A.to_numpy()
		TT = A * 1.0
		TT[np.abs(TT) < 1.0e-200] = 0
		TT = np.reshape(TT, (Nx, Nz), order="F")
		TTT = TT[:, :]
		TTT = np.transpose(TTT)
		rho = np.copy(TTT)

		A = pd.read_csv(
			"strain_" + str(step) + ".txt",
			delimiter = " ",
			comment = "P",
			skiprows = 2,
			header = None,
		)
		A = A.to_numpy()
		TT = A * 1.0
		TT[np.abs(TT) < 1.0e-200] = 1.0E-28
		TT = np.reshape(TT, (Nx, Nz), order = "F")
		TTT = np.transpose(TT)
		TTT[rho < 200] = 1.0E-28
		TTT = np.log10(TTT)
		stc = np.copy(TTT)

		#creating figure
		
		plt.close()
		label_size=20
		plt.rc('xtick', labelsize = label_size)
		plt.rc('ytick', labelsize = label_size)
		
		fig, ax = plt.subplots(1, 1, figsize = (12, 6), constrained_layout = True)

		ax.set_xlabel("Distance (km)", fontsize = 14)
		ax.set_ylabel("Depth (km)", fontsize = 14)
		ax.set_xlim(xmin, xmax)
		ax.set_ylim(-(Lz / 1.0E3 - 40), 40)
		# plt.text(Lx/1.0E3-350, 25+40, str(time)+r' Myr', fontsize = 18)
		ax.text(Lx / 1.0E3 - 350, 25 + 40, '{:01} Myr'.format(time), fontsize = 18)

		cr = 255.0
		color_uc = (228.0 / cr, 156.0 / cr, 124.0 / cr)
		color_lc = (240.0 / cr, 209.0 / cr, 188.0 / cr)
		color_lit = (155.0 / cr, 194.0 / cr, 155.0 / cr)
		color_ast = (207.0 / cr, 226.0 / cr, 205.0 / cr)

		ax.contourf(
			xx,
			zz,
			rho,
			levels=[200.0, 2750, 2900, 3365, 3900],
			colors=[color_uc, color_lc, color_lit, color_ast],
		)

		# print("stc", np.min(stc), np.max(stc))

		# print("stc(log)", np.min(stc), np.max(stc))
		im = ax.imshow(
			stc[::-1, :],
			extent=[0, Lx / 1000, -Lz / 1000 + h_air, 0+h_air],
			zorder=100,
			alpha=0.2,
			cmap=plt.get_cmap("Greys"),
			vmin=-0.5,
			vmax=0.9,
		)
		# ax.fill_between(xx[0], topo, 40, color='white', alpha=1.0)
		ax.fill_between(xx[0], topo_interface, 40, color='white', alpha=1.0)
		#plt.text(100, 10, "%.2lf Myr" % (time[0] / 1.0e6))

		#legend box
		b1 = [0.84, 0.41, 0.15, 0.15]
		bv1 = plt.axes(b1)

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
		plt.contourf(
			xxA,
			yyA,
			A,
			levels=[air_threshold, 2750, 2900, 3365, 3900],
			colors=[color_uc, color_lc, color_lit, color_ast],
		)

		plt.imshow(
			xxA[::-1, :],
			extent=[-0.5, 0.9, 0, 1.5],
			zorder=100,
			alpha=0.2,
			cmap=plt.get_cmap("Greys"),
			vmin=-0.5,
			vmax=0.9,
		)

		bv1.set_yticklabels([])

		# plt.xlabel(r"log$(\epsilon_{II})$", size=14)
		plt.xlabel(prop_label, size=14)
		bv1.tick_params(axis='x', which='major', labelsize=10)
		bv1.set_xticks([-0.5, 0, 0.5])
		bv1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

	if(prop != 'topography'): #Plot isotherms and fill_betweeen
		# #Plot interfaces

		# #The crust and lithopshere basis are extracted from density data
		# Rhoi = read_density(step, Nx, Nz)
		# interfaces=[200, 2900, 3365]
		# # ax.contour(xx, zz, Rhoi, 100, levels=interfaces, colors=['black', 'black'])
		# # ax.fill_between(xx[0], topo, 40, color='white', alpha=1.0)

		# #The isotherms of 1300 and 800 °C are extracted from temperature data
		# Temperi = read_temperature(step, Nx, Nz)
		isotherms=[550, 800, 1300]
		# # isotherms=[800, 900, 1000, 1100, 1200, 1300]
		cs = ax.contour(xx, zz, Temperi, 100, levels=isotherms, colors=['red', 'red', 'red'])
		
		# fmt = {}
		# for level, isot in zip(cs.levels, isotherms):
		# 	fmt[level] = str(level) + r'$^{\circ}$C'
		# ax.clabel(cs, cs.levels, fmt=fmt, inline=True, use_clabeltext=True)
		# ax.clabel(cs, cs.levels, inline=True, use_clabeltext=True)

		# z = np.linspace(Lz/1000.0, 0, Nz)
		# Z = np.linspace(Lz/1000.0, 0, 8001) #zi
		# ##Extract layer topography
		# density=interfaces[0]#200   
		# topo_interface = extract_interface(z, Z, Nx, Rhoi, density) #200 kg/m3 = air/crust interface
		# topo_interface -= np.abs(z_mean)
		# topo_interface = -1.0*topo_interface

		# #Crust base
		# density=interfaces[1]#2900
		# topo_crust = extract_interface(z, Z, Nx, Rhoi, density)
		# topo_crust -= np.abs(z_mean)
		# topo_crust = -1.0*topo_crust
		# #ax.plot(xx[0], topo_crust, '--k')
		
		# #Lithosphere base
		# density=interfaces[2]#3370
		# topo_lit = extract_interface(z, Z, Nx, Rhoi, density)
		# topo_lit -= np.abs(z_mean)
		# topo_lit = -1.0*topo_lit
		# #ax.plot(xx[0], topo_lit, 'r')

		# #Isotherm 800
		# temp=isotherms[0]#800
		# isot_800 = extract_interface(z, Z, Nx, Temperi, temp)
		# isot_800 -= np.abs(z_mean)
		# isot_800 = -1.0*isot_800
		# # ax.plot(xx[0], isot_800, '-r')

		# #Isotherm 1300
		# temp=isotherms[1]#800
		# isot_1300 = extract_interface(z, Z, Nx, Rhoi, temp)
		# isot_1300 -= np.abs(z_mean)
		# isot_1300 = -1.0*isot_1300
		# # ax.plot(xx[0], isot_1300, '-r')

		ax.fill_between(xx[0], topo_interface, 40, color='white', alpha=1.0)

	#Saving figure
	fig_name = prop + "_{:07}.png".format(step)
	plt.savefig(fig_name, dpi=400)

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

######################
#     MAIN CODE      #
######################

#Build the grid to plot countour plots
Nx, Nz, Lx, Lz = read_params()
print('Nx:', Nx,
	  'Nz:', Nz,
	  'Lx:', Lx,
	  'Lz:', Lz) #Lz/(Nz-1)

xi = np.linspace(0, Lx / 1000, Nx)
zi = np.linspace(-Lz / 1000 + 40, 0 + 40, Nz)
xx, zz = np.meshgrid(xi, zi)

#Lists e Dictionaries
props = [#Properties from mandyoc. Comment/uncomment to select with ones you would like to plot
		 # 'density',
		 # 'heat',
		 'lithology',
		 # 'pressure',
		 'strain',
		 'strain_rate',
		 # 'temperature',
		 'temperature_anomaly',
		 'topography',
		 # 'viscosity',
		 # 'vs_anomaly'
		 ]

#label of colorbars
props_label = {'density':              r'$\mathrm{kg/m^3}$',
			   'heat':                 'log(W/kg)',
			   'lithology':            r'log$(\epsilon_{II})$',
			   'pressure':             'GPa',
			   'strain':               'Accumulated strain',
			   'strain_rate':          r'log($\dot{\varepsilon}$)',
			   'temperature':          r'$^{\circ}\mathrm{C}$',
			   'temperature_anomaly':  'Temperature deviation from horizontal mean (°C)',
			   'topography':           '',
			   'viscosity':            'log(Pa.s)',
			   'vs_anomaly':           r'dVs/Vs (\%)'
			   }

#limits of colorbars			   
val_minmax = {'density':             [0.0, 3378.],
			  'heat':                [np.log10(1.0E-13), np.log10(1.0E-9)],
			  'lithology':           [None, None],
			  'pressure':            [-1.0E-3, 1.0],
			  'strain':              [None, None],
			  'strain_rate':         [np.log10(1.0E-19), np.log10(1.0E-12)],
			  'temperature':         [0, 1600],
			  'temperature_anomaly': [-150, 150],
			  'topography':          [-5, 5],
			  # 'viscosity':           [np.log10(1.0E16), np.log10(1.0E25)],
			  # 'viscosity':           [np.log10(1.0E22), np.log10(1.0E25)],
			  'viscosity':           [np.log10(1.0E18), np.log10(1.0E25)],
			  'vs_anomaly':          [-0.03, 0.03]
			  }

#Read and sort the time files list
ts = sorted(glob.glob("time_*.txt"), key=os.path.getmtime)

t1 = int(ts[0][5: -4]) #cat only the numeric section of time files
t2 = int(ts[1][5: -4])
dt = t2 - t1
# dt=50

# step_initial = t1 #145900 #28450
# step_final = int(ts[-1][5:-4]) #200000 #28450
step_initial = 10000
step_final = step_initial
d_step = dt 

for cont in range(step_initial, step_final + d_step, d_step):

	# step = cont*dt
	step = cont
	
	time_fname = "time_" + str(step) + ".txt"
	time = np.loadtxt(time_fname, usecols=2, max_rows=1)
	time = str(round(time/1.0E6, 2))

	print(time, step)

	for prop in props:
		print(prop)

		if (prop != 'lithology' and prop != 'topography'):
			if(prop == 'temperature_anomaly' or prop == 'vs_anomaly'):
				data = read_data('temperature', step, Nx, Nz)
				plot_data(Lx, Lz, Nx, Nz, xx, zz, prop, props_label[prop],
							  val_minmax[prop], data, step, float(time))
			elif(prop == 'vs_anomaly'):
				data = read_data('temperature', step, Nx, Nz)
				plot_data(Lx, Lz, Nx, Nz, xx, zz, prop, props_label[prop],
							  val_minmax[prop], data, step, float(time))
			else:
				data = read_data(prop, step, Nx, Nz)
				
				if (prop == "strain" or prop == "strain_rate" or prop == "viscosity" or prop == "heat"): #log properties
					data[data==0] = 1.0E-28 #avoid log(0)
					data = np.log10(data)
					
					plot_data(Lx, Lz, Nx, Nz, xx, zz, prop, props_label[prop],
							  val_minmax[prop], data, step, float(time))

				else: #not log
					plot_data(Lx, Lz, Nx, Nz, xx, zz, prop, props_label[prop], 
							  val_minmax[prop], data, step, float(time))

		elif(prop == 'topography'):
			##Extract layer topography from  density data or from sp_surface data
			read_from = 'density'
			# read_from = 'sp_surface_global'

			if(read_from == 'density'):
				Rhoi = read_density(step, Nx, Nz)
				z = np.linspace(Lz/1000.0, 0, Nz)
				Z = np.linspace(Lz/1000.0, 0, 8001)
				density = 300
				topo_interface = extract_interface(z, Z, Nx, Rhoi, density) #200 kg/m3 = air/crust interface
				condx = (xi >= 100) & (xi <= 600)
				z_mean = np.mean(topo_interface[condx])
				topo_interface -= np.abs(z_mean)
				topo_interface = -1.0*topo_interface
			else:
				fname = 'sp_surface_global_' + str(step) + '.txt'
				topo = np.loadtxt(fname, unpack=True, skiprows=2, comments='P')/1.0E3
				condx = (xi >= 100) & (xi <= 600)
				z_mean = np.mean(topo[condx])
				topo += np.abs(z_mean)
				topo_interface = topo

			data = topo_interface

			plot_data(Lx, Lz, Nx, Nz, xx, zz, prop, props_label[prop],
					  val_minmax[prop], data, step, float(time))
		else:
			if(prop == 'lithology'):
				data = []
				plot_data(Lx, Lz, Nx, Nz, xx, zz, prop, props_label[prop],
						  val_minmax[prop], data, step, float(time))


print("running make_video\n")
# os.system('bash '+machine_path+'/Doutorado/cenarios/mandyoc/scripts/make_video.sh')

print("running zipper\n")
# os.system('bash '+machine_path+'/Doutorado/cenarios/mandyoc/scripts/zipper.sh')

# filename = os.getcwd()[-6::] #pegar o nome da pasta SimXXX

# print('Zipping videos\n')
# os.system('zip ' + filename + '_videos.zip *.mp4')

#print('Zipping PDFs\n')
#os.system('zip ' + filename + '_PDFs.zip *.pdf')
