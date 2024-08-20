import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import glob


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

dx = (Lx/1000)/(Nx-1)
dz = (Lz/1000)/(Nz-1)

h_air = 40.0

PPhi = []
tt = []


for cont in range(step_initial, step_final, d_step):  #
    print(cont)

    A = np.loadtxt("time_" + str(cont) + ".txt", dtype="str")
    AA = A[:, 2:]
    AAA = AA.astype("float")
    tempo = np.copy(AAA)

    print("Time = %.1lf Myr\n\n" % (tempo[0] / 1.0e6))

    tt = np.append(tt,(tempo[0] / 1.0e6))

    A = pd.read_csv(
        "Phi_" + str(cont) + ".txt",
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
    Phi = np.copy(TTT)


    Phi_sum = np.sum(Phi)*dx*dz


    PPhi = np.append(PPhi,Phi_sum)

    print(Phi_sum*1.0E6)

plt.figure(figsize=(8,3))

plt.axes([0.1,0.2,0.8,0.7])
ylim = 200
plt.plot(tt[1:],(PPhi[1:]-PPhi[:-1])/(tt[1]-tt[0]))
plt.plot([27,27],[0,ylim],"k")
plt.plot([10,10],[0,ylim],"k")

plt.text(26,22,"crust rupture",rotation=90)
plt.text(9,22,"mantle rupture",rotation=90)

plt.xlabel("Myr")
plt.ylabel("Melt production (km$^2$/Myr)")

plt.ylim(0,ylim)
plt.xlim(0,50)

plt.savefig("melt_rate_from_Phi.png")



np.savetxt("melt_rate_from_Phi.txt",np.c_[tt,PPhi*1.0E6])
