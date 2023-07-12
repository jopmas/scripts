"""
This example simulates the evolution of divergent margins, taking into account the plastic rheology and the sin-rift geodynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import os

path = os.getcwd().split('/')
machine_path = '/'+path[1]+'/'+path[2]

label_size=18
plt.rc('xtick', labelsize=label_size)
plt.rc('ytick', labelsize=label_size)

# total model horizontal extent (m)
Lx = 1600 * 1.0e3
# total model vertical extent (m)
Lz = 300 * 1.0e3 #400 * 1.0e3
# number of points in horizontal direction
Nx = 801 #1601
# number of points in vertical direction
Nz = 151 #301 #401
# thickness of sticky air layer (m)
H_sa = 40 * 1.0e3
# thickness of upper crust (m)
H_upper_crust = 20 * 1.0e3
# thickness of lower crust (m)
H_lower_crust = 15 * 1.0e3
# total thickness of lithosphere (m)
H_litho = 130 * 1.0e3
# seed depth bellow base of lower crust (m)
seed_depth = 9 * 1.0e3

x = np.linspace(0, Lx, Nx)
z = np.linspace(Lz, 0, Nz)
X, Z = np.meshgrid(x, z)


##############################################################################
# Interfaces (bottom first)
##############################################################################
interfaces = {
    "litho": np.ones(Nx) * (H_litho + H_sa),
    "seed_base": np.ones(Nx) * (seed_depth + H_lower_crust + H_upper_crust + H_sa),
    "seed_top": np.ones(Nx) * (seed_depth + H_lower_crust + H_upper_crust + H_sa),
    "lower_crust": np.ones(Nx) * (H_lower_crust + H_upper_crust + H_sa),
    "upper_crust": np.ones(Nx) * (H_upper_crust + H_sa),
    "air": np.ones(Nx) * (H_sa),
}

# seed thickness (m)
H_seed = 6 * 1.0e3
# seed horizontal position (m)
x_seed = 800 * 1.0e3
# seed: number of points of horizontal extent
n_seed = 6

interfaces["seed_base"][
    int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)
] = (
    interfaces["seed_base"][
        int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)
    ]
    + H_seed // 2
)
interfaces["seed_top"][
    int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)
] = (
    interfaces["seed_top"][
        int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)
    ]
    - H_seed // 2
)

Huc = 2.5e-6 / 2700.0 #9.259E-10
Hlc = 0.8e-6 / 2800.0 #2.85E-10

# Create the interface file

with open("interfaces.txt", "w") as f:
    rheology_mlit = 'dry' #rheology of lithospheric mantle: dry olivine or wet olivine

    if(rheology_mlit == 'dry'):
        layer_properties = f"""
            C   1.0       1.0        0.1        1.0        10.0         1.0         1.0
            rho 3378.0    3354.0     3354.0     3354.0     2800.0      2700.0      1.0
            H   0.0       9.0e-12    9.0e-12    9.0e-12    {Hlc}       {Huc}       0.0
            A   1.393e-14 2.4168e-15 2.4168e-15 2.4168e-15 8.574e-28   8.574e-28   1.0e-18
            n   3.0       3.5        3.5        3.5        4.0         4.0         1.0
            Q   429.0e3   540.0e3    540.0e3    540.0e3    222.0e3     222.0e3     0.0
            V   15.0e-6   25.0e-6    25.0e-6    25.0e-6    0.0         0.0         0.0
        """

    if(rheology_mlit == 'wet'):
        layer_properties = f"""
            C   1.0       5.0        0.1        5.0        10.0         1.0         1.0
            rho 3378.0    3354.0     3354.0     3354.0     2800.0      2700.0      1.0
            H   0.0       9.0e-12    9.0e-12    9.0e-12    {Hlc}       {Huc}       0.0
            A   1.393e-14 1.393e-14  1.393e-14  1.393e-14  8.574e-28   8.574e-28   1.0e-18
            n   3.0       3.0        3.0        3.0        4.0         4.0         1.0
            Q   429.0e3   429.0e3    429.0e3    429.0e3    222.0e3     222.0e3     0.0
            V   15.0e-6   15.0e-6    15.0e-6    15.0e-6    0.0         0.0         0.0
        """

    for line in layer_properties.split("\n"):
        line = line.strip()
        if len(line):
            f.write(" ".join(line.split()) + "\n")

    # layer interfaces
    data = -1 * np.array(tuple(interfaces.values())).T
    np.savetxt(f, data, fmt="%.1f")

# Plot interfaces
##############################################################################
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

for label, layer in interfaces.items():
    print(label, ":", np.size(layer))
    ax.plot(x/1.0E3, layer/1.0E3, label=f"{label}")
ax.set_xlim([0, Lx/1.0E3])
ax.set_ylim([Lz/1.0E3, 0])
ax.set_xlabel("km", fontsize=label_size)
ax.set_ylabel("km", fontsize=label_size)
ax.legend()
plt.savefig("interfaces_teste.png")
plt.close()


##############################################################################
# Parameters file
##############################################################################
params = f"""
nx = {Nx}
nz = {Nz}
lx = {Lx}
lz = {Lz}
# Simulation options
multigrid                           = 1             # ok -> soon to be on the command line only
solver                              = direct        # default is direct [direct/iterative]
denok                               = 1.0e-15       # default is 1.0E-4
particles_per_element               = 400          # default is 81
particles_perturb_factor            = 0.7           # default is 0.5 [values are between 0 and 1]
rtol                                = 1.0e-7        # the absolute size of the residual norm (relevant only for iterative methods), default is 1.0E-5
RK4                                 = Euler         # default is Euler [Euler/Runge-Kutta]
Xi_min                              = 1.0e-7       # default is 1.0E-14
random_initial_strain               = 0.2           # default is 0.0
pressure_const                      = -1.0          # default is -1.0 (not used) - useful only in horizontal 2D models
initial_dynamic_range               = True         # default is False [True/False]
periodic_boundary                   = False         # default is False [True/False]
high_kappa_in_asthenosphere         = False         # default is False [True/False]
K_fluvial                           = 2.0e-7        # default is 2.0E-7
m_fluvial                           = 1.0           # default is 1.0
sea_level                           = -1000.0           # default is 0.0
basal_heat                          = 0.0          # default is -1.0
# Surface processes
sp_surface_tracking                 = True         # default is False [True/False]
sp_surface_processes                = True         # default is False [True/False]
sp_dt                               = 2.0e5        # default is 0.0
sp_d_c                              = 1.0          # default is 0.0
plot_sediment                       = False         # default is False [True/False]
a2l                                 = False          # default is True [True/False]
free_surface_stab                   = True          # default is True [True/False]
theta_FSSA                          = 0.5           # default is 0.5 (only relevant when free_surface_stab = True)
# Time constrains
step_max                            = 100000          # Maximum time-step of the simulation
time_max                            = 130.0e6     # Maximum time of the simulation [years]
dt_max                              = 15.0e3      # Maximum time between steps of the simulation [years]
step_print                          = 400            # Make file every <step_print>
sub_division_time_step              = 0.5           # default is 1.0
initial_print_step                  = 0             # default is 0
initial_print_max_time              = 1.0e6         # default is 1.0E6 [years]
# Viscosity
viscosity_reference                 = 1.0e26        # Reference viscosity [Pa.s]
viscosity_max                       = 1.0e25        # Maximum viscosity [Pa.s]
viscosity_min                       = 1.0e18        # Minimum viscosity [Pa.s]
viscosity_per_element               = constant      # default is variable [constant/variable]
viscosity_mean_method               = arithmetic      # default is harmonic [harmonic/arithmetic]
viscosity_dependence                = pressure      # default is depth [pressure/depth]
# External ASCII inputs/outputs
interfaces_from_ascii               = True          # default is False [True/False]
n_interfaces                        = {len(interfaces.keys())}           # Number of interfaces int the interfaces.txt file
variable_bcv                        = False         # default is False [True/False]
temperature_from_ascii              = True         # default is False [True/False]
velocity_from_ascii                 = False         # default is False [True/False]
binary_output                       = False         # default is False [True/False]
sticky_blanket_air                  = True         # default is False [True/False]
precipitation_profile_from_ascii    = True         # default is False [True/False]
climate_change_from_ascii           = True         # default is False [True/False]
print_step_files                    = False          # default is True [True/False]
checkered                           = False         # Print one element in the print_step_files (default is False [True/False])
sp_mode                             = 5             # default is 1 [0/1/2]
geoq                                = on            # ok
geoq_fac                            = 100.0           # ok
# Physical parameters
temperature_difference              = 1500.         # ok
thermal_expansion_coefficient       = 3.28e-5       # ok
thermal_diffusivity_coefficient     = 1.0e-6        # ok
gravity_acceleration                = 10.0          # ok
density_mantle                      = 3300.         # ok
external_heat                       = 0.0e-12       # ok
heat_capacity                       = 1250.         # ok
non_linear_method                   = on            # ok
adiabatic_component                 = on            # ok
radiogenic_component                = on            # ok
# Velocity boundary conditions
top_normal_velocity                 = fixed         # ok
top_tangential_velocity             = free          # ok
bot_normal_velocity                 = fixed         # ok
bot_tangential_velocity             = free          # ok
left_normal_velocity                = fixed         # ok
left_tangential_velocity            = free          # ok
right_normal_velocity               = fixed         # ok
right_tangential_velocity           = fixed         # ok
surface_velocity                    = 0.0e-2        # ok
multi_velocity                      = False         # default is False [True/False]
# Temperature boundary conditions
top_temperature                     = fixed         # ok
bot_temperature                     = fixed         # ok
left_temperature                    = free          # ok
right_temperature                   = free          # ok
rheology_model                      = 9             # ok
T_initial                           = 3             # ok
"""
# Create the parameter file
with open("param.txt", "w") as f:
    for line in params.split("\n"):
        line = line.strip()
        if len(line):
            f.write(" ".join(line.split()) + "\n")

##############################################################################
# Initial temperature field
##############################################################################

# T = 1300 * (z - H_sa) / (H_litho)  # Temperature
T = 1300 * (z - H_sa) / (130*1.0E3)  # Temperature

Ta = 1262 / np.exp(-10 * 3.28e-5 * (z - H_sa) / 1250)

T[T < 0.0] = 0.0
T[T > Ta] = Ta[T > Ta]

kappa = 1.0e-6

ccapacity = 1250

H = np.zeros_like(T)

cond = (z >= H_sa) & (z < H_upper_crust + H_sa)  # upper crust
H[cond] = Huc

cond = (z >= H_upper_crust + H_sa) & (
    z < H_lower_crust + H_upper_crust + H_sa
)  # lower crust
H[cond] = Hlc

Taux = np.copy(T)
t = 0
dt = 5000
dt_sec = dt * 365 * 24 * 3600
cond = (z > H_sa + H_litho) | (T == 0)  # (T > 1300) | (T == 0)
dz = Lz / (Nz - 1)

while t < 500.0e6:
    T[1:-1] += (
        kappa * dt_sec * ((T[2:] + T[:-2] - 2 * T[1:-1]) / dz ** 2)
        + H[1:-1] * dt_sec / ccapacity
    )
    T[cond] = Taux[cond]
    t = t + dt

T = np.ones_like(X) * T[:, None]

print(np.shape(T))

# Save the initial temperature file
np.savetxt("input_temperature_0.txt", np.reshape(T, (Nx * Nz)), header="T1\nT2\nT3\nT4")

# Plot temperature field and thermal profile
##############################################################################

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(16, 8), sharey=True)

ax0.contour(X / 1.0e3, (Z - H_sa) / 1.0e3, T, levels=np.arange(0, 1610, 100))
ax0.set_ylim((Lz - H_sa) / 1.0e3, -H_sa / 1000)
ax0.set_xlabel("km", fontsize=label_size)
ax0.set_ylabel("km", fontsize=label_size)
ax1.set_xlabel("$^\circ$C", fontsize=label_size)

ax1.plot(T[:, 0], (z - H_sa) / 1.0e3, "-k")

code = 0
for label in list(interfaces.keys()):
    color = "C" + str(code)
    code += 1
    ax1.hlines(
        (interfaces[label][0] - H_sa) / 1.0e3,
        np.min(T[:, 0]),
        np.max(T[:, 0]),
        label=f"{label}",
        color=color
    )

ax1.set_ylim((Lz - H_sa) / 1.0e3, -H_sa / 1000)
ax1.set_xlim(0, 1400)
ax0.grid(':k', alpha=0.7)
ax1.grid(':k', alpha=0.7)
ax1.legend(loc='lower left', fontsize=14)
plt.savefig("initial_temperature_field.png")
plt.close()


##############################################################################
# Boundary condition - velocity
##############################################################################

fac_air = 10.0e3

# 1 cm/year
vL = 0.005 / (365 * 24 * 3600)  # m/s

h_v_const = H_litho + 20.0e3  #thickness with constant velocity 
ha = Lz - H_sa - h_v_const  # difference

vR = 2 * vL * (h_v_const + fac_air + ha) / ha  # this is to ensure integral equals zero

VX = np.zeros_like(X)
cond = (Z > h_v_const + H_sa) & (X == 0)
VX[cond] = vR * (Z[cond] - h_v_const - H_sa) / ha

cond = (Z > h_v_const + H_sa) & (X == Lx)
VX[cond] = -vR * (Z[cond] - h_v_const - H_sa) / ha

cond = X == Lx
VX[cond] += +2 * vL

cond = Z <= H_sa - fac_air
VX[cond] = 0

# print(np.sum(VX))

v0 = VX[(X == 0)]
vf = VX[(X == Lx)]
sv0 = np.sum(v0[1:-1]) + (v0[0] + v0[-1]) / 2.0
svf = np.sum(vf[1:-1]) + (vf[0] + vf[-1]) / 2.0
# print(sv0, svf, svf - sv0)

diff = (svf - sv0) * dz

vv = -diff / Lx
# print(vv, diff, svf, sv0, dz, Lx)

VZ = np.zeros_like(X)

cond = Z == 0
VZ[cond] = vv
#save bc to plot arraows in numerical setup
vels_bc = np.array([v0, vf])
vz0 = VZ[(z == 0)]
np.savetxt("vel_bc.txt", vels_bc.T)
np.savetxt("velz_bc.txt", vz0.T)
# print(np.sum(v0))

VVX = np.copy(np.reshape(VX, Nx * Nz))
VVZ = np.copy(np.reshape(VZ, Nx * Nz))

v = np.zeros((2, Nx * Nz))

v[0, :] = VVX
v[1, :] = VVZ

v = np.reshape(v.T, (np.size(v)))

# Create the initial velocity file
np.savetxt("input_velocity_0.txt", v, header="v1\nv2\nv3\nv4")

# Plot veolocity
##############################################################################
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(9, 9), constrained_layout=True, sharey=True)

ax0.plot(VX[:, 0]*1e10, (z - H_sa) / 1000, "k-", label="left side")
ax1.plot(VZ[:, 0]*1e10, (z - H_sa) / 1000, "k-", label="left side")

ax0.plot(VX[:, -1]*1e10, (z - H_sa) / 1000, "r-", label="right side")
ax1.plot(VZ[:, -1]*1e10, (z - H_sa) / 1000, "r-", label="right side")

ax0.legend(loc='upper left', fontsize=14)
ax1.legend(loc='upper right', fontsize=14)

ax0_xlim = ax0.get_xlim()
ax1_xlim = ax1.get_xlim()

ax0.set_yticks(np.arange(0, Lz / 1000, 40))
ax1.set_yticks(np.arange(0, Lz / 1000, 40))

ax0.set_ylim([Lz / 1000 - H_sa / 1000, -H_sa / 1000])
ax1.set_ylim([Lz / 1000 - H_sa / 1000, -H_sa / 1000])

ax0.set_xlim([-8, 8])
ax1.set_xlim([-8, 8])
ax0.set_xticks(np.arange(-8, 9, 4))
ax1.set_xticks(np.arange(-8, 9, 4))
ax0.grid(':k', alpha=0.7)
ax1.grid(':k', alpha=0.7)

ax0.set_xlabel("$10^{-10}$ (m/s)", fontsize=label_size)
ax1.set_xlabel("$10^{-10}$ (m/s)", fontsize=label_size)
ax0.set_ylabel("Depth (km)", fontsize=label_size)

ax0.set_title("Horizontal component of velocity")

ax1.set_title("Vertical component of velocity")

plt.savefig("velocity.png")
plt.close()

#When climate effects will start to act - scaling to 1
climate = f'''
        2
        0 0.0
        10 0.02
    '''

with open('climate.txt', 'w') as f:
    for line in climate.split('\n'):
        line = line.strip()
        if len(line):
            f.write(' '.join(line.split()) + '\n')

#Creating precipitation profile

# prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/8)**6) #original
# prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/(8*2))**6) #100 km
prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/(8*4))**6) #50 km

plt.figure(figsize=(12, 9), constrained_layout=True)
plt.xlim([0, Lx/1.0E3])
plt.ylim([0, np.max(prec)])
plt.xlabel("km", fontsize=label_size)
plt.ylabel("Precipitation", fontsize=label_size)
plt.plot(x/1000,prec)
plt.grid(':k', alpha=0.7)

figname='precipitation_profile.png'
plt.savefig(figname, dpi=300)

np.savetxt("precipitation.txt", prec, fmt="%.8f")


#Creating run files

run_gcloud = f'''
        #!/bin/bash
        MPI_PATH=$HOME/opt/petsc/arch-0-fast/bin
        MANDYOC_PATH=$HOME/opt/mandyoc
        NUMBER_OF_CORES=6
        touch FD.out
        $MPI_PATH/mpirun -n $NUMBER_OF_CORES $MANDYOC_PATH/mandyoc -seed 0,2 -strain_seed 0.0,1.0 | tee FD.out
        bash /home/joao_macedo/Mestrado/cenarios/mandyoc/scripts/zipper_gcloud.sh
        bash /home/joao_macedo/Mestrado/cenarios/mandyoc/scripts/clean_gcloud.sh
        sudo poweroff
    '''
with open('run_gcloud.sh', 'w') as f:
    for line in run_gcloud.split('\n'):
        line = line.strip()
        if len(line):
            f.write(' '.join(line.split()) + '\n')

run_linux = f'''
        #!/bin/bash
        #!/bin/bash
        MPI_PATH=$HOME/opt/petsc/arch-0-fast/bin
        MANDYOC_PATH=$HOME/opt/mandyoc
        NUMBER_OF_CORES=12
        touch FD.out
        $MPI_PATH/mpirun -n $NUMBER_OF_CORES $MANDYOC_PATH/mandyoc -seed 0,2 -strain_seed 0.0,1.0 | tee FD.out
    '''
with open('run-linux.sh', 'w') as f:
    for line in run_linux.split('\n'):
        line = line.strip()
        if len(line):
            f.write(' '.join(line.split()) + '\n')

#zip input files
filename = 'inputs_'+path[-1]+'.zip'
files_list = ' interfaces.txt param.txt input*_0.txt run*.sh vel*.txt *.png precipitation.txt climate.txt'
os.system('zip '+filename+files_list)
