import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import glob
from matplotlib import ticker
import matplotlib as mpl
label_size=20
mpl.rcParams['xtick.labelsize'] = label_size
mpl.rcParams['ytick.labelsize'] = label_size
from matplotlib.ticker import LogFormatterExponent
import matplotlib.ticker as tk

step_initial = int(sys.argv[1])
step_final = int(sys.argv[2])

if len(sys.argv) > 3:
    d_step = int(sys.argv[3])
else:
    d_step = 10

with open("param.txt", "r") as f:
    line = f.readline()
    line = line.split()
    Nx = int(line[2])
    line = f.readline()
    line = line.split()
    Nz = int(line[2])
    line = f.readline()
    line = line.split()
    Lx = float(line[2])
    line = f.readline()
    line = line.split()
    Lz = float(line[2])

print(Nx, Nz, Lx, Lz)

xi = np.linspace(0, Lx / 1000, Nx)
zi = np.linspace(-Lz / 1000, 0, Nz)
xx, zz = np.meshgrid(xi, zi)

h_air = 40.0

for cont in range(step_initial, step_final, d_step):  #
    print(cont)

    A = np.loadtxt("time_" + str(cont) + ".txt", dtype="str")
    AA = A[:, 2:]
    AAA = AA.astype("float")
    tempo = np.copy(AAA)

    print("Time = %.1lf Myr\n\n" % (tempo[0] / 1.0e6))


    A = pd.read_csv(
        "density_" + str(cont) + ".txt",
        delimiter=" ",
        comment="P",
        skiprows=2,
        header=None,
    )
    A = A.to_numpy()
    TT = A * 1.0
    TT[np.abs(TT) < 1.0e-200] = 0
    TT = np.reshape(TT, (Nx, Nz), order="F")
    TTT = TT[:, :]
    TTT = np.transpose(TTT)
    rho = np.copy(TTT)

    A = pd.read_csv(
        "viscosity_" + str(cont) + ".txt",
        delimiter=" ",
        comment="P",
        skiprows=2,
        header=None,
    )
    A = A.to_numpy()
    TT = A * 1.0
    TT[np.abs(TT) < 1.0e-200] = 0
    TT = np.reshape(TT, (Nx, Nz), order="F")
    TTT = TT[:, :]
    TTT = np.transpose(TTT)
    eta = np.copy(TTT)
    print("visc",eta.shape)

    A = pd.read_csv(
        "velocity_" + str(cont) + ".txt",
        delimiter=" ",
        comment="P",
        skiprows=2,
        header=None,
    )
    A = A.to_numpy()
    vxx = A[::2]    
    TT = vxx * 1.0
    TT[np.abs(TT) < 1.0e-200] = 0
    TT = np.reshape(TT, (Nx, Nz), order="F")
    TTT = np.transpose(TT)
   #TTT = np.log10(TTT)
    velo = np.copy(TTT)
    print("velo", velo.shape)


    strain_rate = np.zeros((Nz,Nx))
    stress_field = np.zeros((Nz,Nx))
    

    for i in range (Nx-1):
        strain_rate[:,i] = (velo[:,i+1] - velo[:,i] ) / (Lx/(Nx-1))  
    print("strain",strain_rate.shape)


    stress_field = -eta * strain_rate
    print("stress",stress_field.shape)
    stress_field[rho<200.0] = float('NaN')


    plt.close()
    plt.figure(figsize=(20, 7))

    plt.contourf(
        xx,
        zz + h_air,
        stress_field/1.0e6,
        levels = np.linspace(-8,8,31)
    )


    cbar = plt.colorbar()
    #plt.colorbar(label=r"$(\sigma_{xx})$ (MPa)",)
    cbar.set_label(label=r"$(\sigma_{xx})$ (MPa)",fontsize=20)
    plt.text(100, 45, "%.1lf Myr" % (tempo[0] / 1.0e6),fontsize = 20)
    plt.xlabel("Distance (km)",fontsize = 20)
    plt.ylabel("Depth (km)",fontsize = 20)


    plt.savefig("stress_field_{:05}.png".format(cont * 1))

    plt.close()
    plt.figure(figsize=(20, 7))
    eta[rho<200.0] = float('NaN')

    plt.contourf(
        xx,
        zz + h_air,
        eta,
        locator=ticker.LogLocator(),
	#levels=[1e17,1e18,1e19,1e20,1e21,1e22,1e23,1e24,1e25,1E26],    
        #levels = np.linspace(1.0e19,1.0e26,8),
        #norm=mpl.colors.LogNorm(),
    )
    cbar = plt.colorbar(format='%.0e')
    #plt.colorbar(label=r"$(\eta)$ (Pa.s)",)
    cbar.set_label(label=r"$(\eta)$ (Pa.s)",fontsize=20)
    
    #cbformat = tk.ScalarFormatter()
    #cbar.formatter = LogFormatterExponent(base=10)
    plt.text(100, 45, "%.1lf Myr" % (tempo[0] / 1.0e6),fontsize=20)
    plt.xlabel("Distance (km)",fontsize = 20)
    plt.ylabel("Depth (km)",fontsize = 20)

    plt.savefig("viscosity_{:05}.png".format(cont * 1))





