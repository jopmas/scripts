import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import subprocess
import xarray as xr
import multiprocessing #needed to run pymp in mac
multiprocessing.set_start_method('fork') #needed to run pymp in mac
import pymp

path = os.getcwd().split('/')
machine_path = f'/{path[1]}/{path[2]}' #cat the /home/user/ or /Users/user from system using path

path_to_functions = f"{machine_path}/opt/scripts"
sys.path.append(os.path.abspath(path_to_functions))

if '' in sys.path:
    sys.path.remove('')
from functions.mandyocIO import read_datasets, change_dataset, single_plot

model_path = os.getcwd() # Get local file
model_name = model_path.split('/')[-1]
output_path = '_output'
print(f"Model name: {model_name}\n")
print(f"Model path: {model_path}\n")
print(f"Output path: {output_path}\n")

make_videos = True
make_gifs = True
zip_files = True

Tdataset = xr.open_dataset("_output_temperature.nc")
trackdataset = xr.open_dataset("_track_xzPT_all_steps.nc")

#reading param files
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

print(
    "nx:", Nx, "\n",
    "nz:", Nz, "\n",
    "Lx:", Lx, "\n",
    "Lz:", Lz
)

#xx,zz = np.mgrid[0:Lx:(Nx)*1j,-Lz:0:(Nz)*1j]

xi = np.linspace(0,Lx/1.0E3,Nx)
zi = np.linspace(-Lz/1.0E3,0,Nz)
xx,zz = np.meshgrid(xi,zi)

dx = Lx/(Nx-1)
dz = Lz/(Nz-1)

thickness_air = 40.0

argv = sys.argv

if len(argv)>1:
    step_initial = int(sys.argv[1])
    step_final = int(sys.argv[2])
    d_step = int(sys.argv[3])
else:
    step_initial = Tdataset.step.values[0]
    step_final = Tdataset.step.values[-1] 
    d_step = Tdataset.step.values[1] - Tdataset.step.values[0]


for cont in range(step_initial, step_final + d_step, d_step): 
    print("Step: ",cont) 
    
    # Read time
    time = np.loadtxt("time_" + str(cont) + ".txt", dtype="str")
    time = time[:, 2:]
    time = time.astype("float")
   
    # Read density
    rho = pd.read_csv(
        "density_" + str(cont) + ".txt",
        delimiter=" ",
        comment="P",
        skiprows=2,
        header=None,
    )
    rho = rho.to_numpy()
    rho[np.abs(rho) < 1.0e-200] = 0
    rho = np.reshape(rho, (Nx, Nz), order="F")
    rho = np.transpose(rho)
    
    # Read strain
    strain = pd.read_csv(
        "strain_" + str(cont) + ".txt",
        delimiter=" ",
        comment="P",
        skiprows=2,
        header=None,
    )
    strain = strain.to_numpy()
    strain[np.abs(strain) < 1.0e-200] = 0
    strain = np.reshape(strain, (Nx, Nz), order="F")
    strain = np.transpose(strain)
    strain[rho < 200] = 0
    strain_log = np.log10(strain)
    
    print("Step =", cont)
    print("Time = %.1lf Myr\n\n" % (time[0] / 1.0e6))
    print("strain(log)", np.min(strain_log), np.max(strain_log))
    print("strain", np.min(strain), np.max(strain))
    
    plt.figure(figsize=(20, 5))
    plt.title("Time = %.1lf Myr\n\n" % (time[0] / 1.0e6))

    # Create the colors to plot the density
    cr = 255.0
    color_upper_crust = (228.0 / cr, 156.0 / cr, 124.0 / cr)
    color_lower_crust = (240.0 / cr, 209.0 / cr, 188.0 / cr)
    color_lithosphere = (155.0 / cr, 194.0 / cr, 155.0 / cr)
    color_asthenosphere = (207.0 / cr, 226.0 / cr, 205.0 / cr)
    colors = [
        color_upper_crust, 
        color_lower_crust, 
        color_lithosphere, 
        color_asthenosphere
    ]
    # Plot density
    plt.contourf(
        xx,
        zz + thickness_air,
        rho,
        levels=[200.0, 2750, 2900, 3365, 3900],
        colors=colors,
    )  
    
    # Plot strain_log
    plt.imshow(
        strain_log[::-1, :],
        extent=[0, Lx / 1e3, -Lz / 1e3 + thickness_air, thickness_air],
        zorder=100,
        alpha=0.2,
        cmap=plt.get_cmap("Greys"),
        vmin=-0.5,
        vmax=0.9,
    )
    plt.xlabel("x [km]")
    plt.ylabel("Depth [km]")



    xp,zp,Pp,Tp = np.loadtxt(f"track_xzPT_step_{str(cont).zfill(6)}.txt", unpack=True)

    plt.plot(xp/1.0E3,zp/1.0E3+thickness_air,".")
    
    b1 = [0.67, 0.31, 0.2, 0.2]
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
        colors=colors,
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
    
    plt.xlabel("$log_{10}(\epsilon_{II})$", size=18)

    plt.savefig(f"_output/{model_name}_litho_particles_{str(cont).zfill(6)}.png")

##############################################################################################################################################################################
if(make_videos):
    print("Generating videos...")

    fps = 30
    videoname = f'{model_path}/_output/{model_name}_litho_particles'
        
    try:
        comand = f"rm {videoname}.mp4"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
        print(f"\tRemoving previous litho_particles video.")
    except:
        print(f"\tNo litho_particles video to remove.")

    comand = f"ffmpeg -r {fps} -f image2 -s 1920x1080 -pattern_type glob -i \"{videoname}*.png\" -vcodec libx264 -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -r 24 -y -an -crf 25 -pix_fmt yuv420p {videoname}.mp4"
    result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
    print("\tDone!")


##########################################################################################################################################################################

# # Converting videos to gifs
# 
# ss: skip seconds
# 
# t: duration time of the output
# 
# i: inputs format
# 
# vf: filtergraph (video filters)
# 
#     - fps: frames per second
# 
#     - scale: resize accordint to given pixels (e.g. 1080 = 1080p wide)
#     
#     - lanczos: scaling algorithm
#     
#     - palettegen and palette use: filters that generate a custom palette
#     
#     - split: filter that allows everything to be done in one command
# 
# loop: number of loops
# 
#     - 0: infinite
# 
#     - -1: no looping
# 
#     - for numbers n >= 0, create n+1 loops


if(make_gifs):
    print("Converting videos to gifs...")
    gifname = f'{model_path}/_output/{model_name}_litho_particles'
        
    try:
        comand = f"rm {gifname}.gif"
        result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True)
        print(f"\tRemoving previous litho_particles gif.")
    except:
        print(f"\tNo litho_particles gif to remove.")
    
    comand = f"ffmpeg -ss 0 -t 15 -i '{gifname}.mp4' -vf \"fps=30,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse\" -loop 0 {gifname}.gif"
    result = subprocess.run(comand, shell=True, check=True, capture_output=True, text=True) 
    print("\tDone!")

##########################################################################################################################################################################

if(zip_files):
    #zip plots, videos and gifs
    print('Zipping figures, videos and gifs...')
    outputs_path = f'{model_path}/_output/'
    os.chdir(outputs_path)
    subprocess.run(f"zip {model_name}_imgs.zip *litho_particles*.png", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"zip {model_name}_videos.zip *litho_particles.mp4", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"zip {model_name}_gifs.zip *litho_particles.gif", shell=True, check=True, capture_output=True, text=True)
    subprocess.run(f"rm *.png", shell=True, check=True, capture_output=True, text=True)
    print('Zipping complete!')

