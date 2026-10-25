import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil
import argparse
import re
import glob
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

import xarray as xr

path = os.getcwd().split('/')
machine_path = '/'+path[1]+'/'+path[2]

###############################################################################################################################################
#Functions
###############################################################################################################################################

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

def layer_indices(layers, label):
    '''Return the indices (position in the stack) of the layers with the given label.

    Parameters
    ----------

    layers: list of MandyocLayer
        Layers from bottom to top

    label: str
        layer_label of the layers (label of the shapes in the SVG)
    '''
    return [i for i, layer in enumerate(layers) if layer.layer_label == label]

def slot_thickness(layers, i, ix):
    '''
    Thickness [m] of the i-th layer of the stack at column ix
    '''
    top = layers[i+1].interface[ix] if i < len(layers)-1 else 0.0
    return layers[i].interface[ix] - top

def layer_thickness(layers, label, ix):
    '''
    Total thickness [m] of the layers with the given label at column ix
    '''
    return sum(slot_thickness(layers, i, ix) for i in layer_indices(layers, label))

def layer_grid(layers, z):
    '''
    Return the index of the layer at each node as a (Nz, Nx) array.
    z is the depth of the nodes (positive downwards). Nodes on an interface belong to the layer below.
    '''
    bases = np.array([layer.interface for layer in layers[1:]]) #(n_layers-1, Nx)
    return (z[:, None, None] < bases[None, :, :]).sum(axis=1)

def layer_property(layers, grid, prop):
    '''
    Map a property of the layers (e.g. 'density') to the nodes of grid (output of layer_grid)
    '''
    return np.array([layer.__dict__[prop] for layer in layers], dtype=float)[grid]

def read_params(fpath):
    '''
    Read Nx, Nz, Lx, Lz from param.txt
    '''
    with open(fpath+"param.txt","r") as f:
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

def read_data(prop, step, Nz, Nx, fpath):
    '''
    Read and reshape readed data according to parameters to return a (Nx, Nz) array
    '''
    
    #build filename
    filename = fpath + prop + "_" + str(step) + ".txt"

    data = np.loadtxt(filename, skiprows=2, unpack=True, comments="P")
    data = np.reshape(data, (Nz, Nx))
    
    return data

def calc_mean_temperaure_region(data, Nz, xx, begin, end):
    '''
    This funcition select a region in x direction in a 2D array and calculates the horizontal mean

    Parameters
    ----------

    data: `numpy.ndarray`

    Nz: int
        Number of points in Z direction

    xx: numpy.ndarray
        2D grid with x cordinates

    begin: float
        Start point

    end: float
        End point

    Returns
    -------
    arr: `numpy.ndarray`
        Array containing the horizontal mean of selected region
    '''

    x_region = (xx >= begin) & (xx <= end)
    Nx_aux = len(x_region[0][x_region[0]==True])
    data_sel = data[x_region].reshape(Nz, Nx_aux)
    data_sel_mean = np.mean(data_sel, axis=1)

    return data_sel_mean

def read_preset_temperature(path):
    '''
    Horizontal mean of the final temperature field of a preliminary scenario.

    Parameters
    ----------

    path: str
        Folder of the preliminary scenario (with _output_temperature.nc, or with param.txt and temperature_*.txt)
        or the _output_temperature.nc file itself

    Returns
    -------
    depth: depth of the nodes from the top of the domain [m] (increasing)
    profile: horizontal mean of the temperature [oC]
    thickness_sa: thickness of the sticky air [m]
    thickness_litho: thickness of the lithosphere [m] (None if the interfaces.txt of the scenario is not found)
    source: description of the field used
    '''
    folder = os.path.dirname(path) if(os.path.isfile(path)) else path
    nc_file = path if(os.path.isfile(path)) else os.path.join(folder, '_output_temperature.nc')

    if(os.path.isfile(nc_file)):
        dataset = xr.open_dataset(nc_file)
        temperature = dataset.temperature.isel(time=-1) #(x, z), final step
        depth = -dataset.z.values
        profile = temperature.mean(dim='x').values
        source = f"{nc_file} (time = {float(dataset.time[-1]):g} Myr)"
    else:
        if(not os.path.isfile(os.path.join(folder, 'param.txt'))):
            sys.exit(f"preset_initial_temperature: no _output_temperature.nc nor param.txt in {folder}")
        Nx_aux, Nz_aux, Lx_aux, Lz_aux = read_params(folder + '/')
        steps = [int(m.group(1)) for m in (re.fullmatch(r'temperature_(\d+)\.txt', os.path.basename(f))
                 for f in glob.glob(os.path.join(folder, 'temperature_*.txt'))) if m]
        if(len(steps) == 0):
            sys.exit(f"preset_initial_temperature: no temperature_*.txt in {folder}")
        step_final = max(steps) #step of final thermal structure
        Datai = read_data('temperature', step_final, Nz_aux, Nx_aux, folder + '/') #(Nz, Nx), from the bottom to the top
        depth = np.linspace(Lz_aux, 0, Nz_aux)
        profile = np.mean(Datai, axis=1)
        source = f"{folder}/temperature_{step_final}.txt"

    order = np.argsort(depth)
    depth, profile = depth[order], profile[order]

    #thickness of the sticky air (last interface) and of the lithosphere (first interface = top of the asthenosphere)
    #from the interfaces.txt of the preliminary scenario (most common values along x). Without it, the sticky air is
    #the depth where the temperature starts to increase and the lithosphere is unknown (None).
    thickness_sa, thickness_litho = None, None
    for fname in [os.path.join(folder, 'interfaces.txt')] + sorted(glob.glob(os.path.join(folder, 'inputs*', 'interfaces.txt'))):
        if(os.path.isfile(fname)):
            try:
                interfaces_preset = -np.loadtxt(fname, skiprows=7, ndmin=2)
            except ValueError:
                continue
            def most_common(values):
                v, c = np.unique(np.round(values, -2), return_counts=True)
                return v[np.argmax(c)]
            thickness_sa = most_common(interfaces_preset[:, -1])
            thickness_litho = most_common(interfaces_preset[:, 0]) - thickness_sa
            source += f", interfaces from {fname}"
            break
    if(thickness_sa is None):
        thickness_sa = depth[max(np.argmax(profile > 1.0) - 1, 0)]

    return depth, profile, thickness_sa, thickness_litho, source

def adapt_preset_temperature(depth_ref, T_ref, sa_ref, litho_ref, z, thickness_sa, thickness_litho, thickness_mantle):
    '''
    Initial temperature field (Nz, Nx) from the horizontal mean profile of a preliminary scenario,
    adapted to the lithosphere of each column (generalization of keel_adjust):
      - columns with the reference lithosphere thickness litho_ref: the profile itself (as a function of the
        depth below the surface);
      - other columns (keels, thinned or oceanic lithosphere, ...): the profile in the crust, a linear geotherm
        in the lithospheric mantle from the Moho down to the base of the lithosphere, and the asthenospheric
        profile below it. Where the lithosphere is thinner than litho_ref, the asthenospheric profile is
        extended upwards with a linear fit (adiabat) of the profile below litho_ref + 50 km.

    depth_ref, T_ref: depth from the top [m] and temperature [oC] of the profile; sa_ref: its sticky air [m]
    litho_ref: lithosphere thickness of the preliminary scenario [m]
    z: depth of the nodes [m]; thickness_sa, thickness_litho, thickness_mantle: along x [m]
    '''
    d_ref = depth_ref - sa_ref #depth below the surface in the preliminary scenario
    cond = (d_ref >= litho_ref + 50.0e3) & (d_ref <= d_ref.max() - 50.0e3)
    a, b = np.polyfit(d_ref[cond], T_ref[cond], 1) #asthenospheric profile

    def T_profile(d):
        '''Profile at depth d below the surface (extended below the preliminary domain with the fit)'''
        return np.where(d > d_ref.max(), a*d + b, np.interp(d, d_ref, T_ref))

    T = np.zeros((len(z), len(thickness_sa)))
    columns = np.round(np.vstack([thickness_sa, thickness_litho, thickness_mantle]), 1)
    columns_unique, idx_columns = np.unique(columns, axis=1, return_inverse=True)
    T_unique = np.zeros((len(z), columns_unique.shape[1]))
    for k, (sa, litho, mantle) in enumerate(columns_unique.T):
        d = z - sa #depth below the surface
        if(abs(litho - litho_ref) < 1.0e3): #reference lithosphere
            Tk = T_profile(d)
        else:
            moho = litho - mantle
            T_below = T_profile(d) if(litho >= litho_ref) else np.maximum(T_profile(d), a*d + b) #asthenosphere
            T_lab = T_profile(litho) if(litho >= litho_ref) else max(float(T_profile(litho)), a*litho + b)
            T_moho = T_profile(moho)
            Tk = np.where(d <= moho, T_profile(d), T_below)
            in_mantle = (d > moho) & (d < litho)
            if(litho > moho):
                Tk[in_mantle] = T_moho + (T_lab - T_moho) * (d[in_mantle] - moho) / (litho - moho)
        Tk[d <= 0] = 0.0 #sticky air
        T_unique[:, k] = Tk
    T = T_unique[:, idx_columns.ravel()]
    T[T <= 1.0e-7] = 0.0 #dealing with <=0 values inherited from interpolation
    return T

###############################################################################################################################################
#Customizing matplotlib 
label_size=18
plt.rc('xtick', labelsize=label_size)
plt.rc('ytick', labelsize=label_size)

#Install the following package from (https://www.fabiocrameri.ch/colourmaps/) for inclusive color palletes
#or comment set crameri_colors as False

# crameri_colors=True
crameri_colors=False
if(crameri_colors):
    from cmcrameri import cm as cr
    def get_cycle(cmap, N=None, use_index="auto"):
        if isinstance(cmap, str):
            if use_index == "auto":
                if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                            'Dark2', 'Set1', 'Set2', 'Set3',
                            'tab10', 'tab20', 'tab20b', 'tab20c']:
                    use_index=True
                else:
                    use_index=False
            cmap = matplotlib.cm.get_cmap(cmap)
        if not N:
            N = cmap.N
        if use_index=="auto":
            if cmap.N > 100:
                use_index=False
            elif isinstance(cmap, LinearSegmentedColormap):
                use_index=False
            elif isinstance(cmap, ListedColormap):
                use_index=True
        if use_index:
            ind = np.arange(int(N)) % cmap.N
            return cycler("color",cmap(ind))
        else:
            colors = cmap(np.linspace(0,1,N))
            return cycler("color",colors)

    n_colors = 10
    # plt.rcParams["axes.prop_cycle"] = get_cycle(cr.romaO, n_colors)
    # plt.rcParams["axes.prop_cycle"] = get_cycle(cr.oslo, n_colors)
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cr.batlowKS(np.linspace(0, 1, 10)))
    #From Color Universal Design (CUD): https://jfly.uni-koeln.de/color/
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"])
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ['#88CCEE', '#44AA99', '#117733', '#332288', '#DDCC77', '#999933','#CC6677', '#882255', '#AA4499', '#585858'])
    


###############################################################################################################################################
#Setting the kind of tectonic scenario and number of cores
###############################################################################################################################################

# ncores = 20
# ncores = 64
# ncores = 70
ncores = 90
# ncores = 128
# ncores = 192
# ncores = 256
# ncores = 384
cores_per_node = 96


#Estimating number of nodes needed according to number of cores
nodes = (ncores + cores_per_node - 1) // cores_per_node #ceil division

###############################################################################################################################################
# Domain and interfaces
###############################################################################################################################################

#Scenario geometry drawn in Inkscape.
#Each shape of the SVG is a MandyocLayer described by its label (layer_label), title (name in the legend)
#and description (rheology, C, rho, H, ...). See svg2interfaces.py.
#The shape labelled 'domain' gives the domain (Lx, Lz) and the resolution (dx, dz) in its description.
#It may also give the numerical experiment (see "Numerical experiment" below), e.g.
#       regime = accordion
#       velocity = 1 cm/yr
#       moving_boundary = right
#       quiescence_time = 300 Myr
#       time_max = 500 Myr
#       magmatism = off
#       magmatism_extraction = off
#       magmatism_Ts0 = 1080
#       preset_initial_temperature = ../preliminary     (folder of a preliminary scenario, relative to the SVG)
#Usage: python generate_input_files_svg.py cenario.svg [--dx 1] [--dz 1] [--regime rifting] [--velocity 1] [--moving-boundary right]
#                                          [--quiescence-time 300] [--time-max 500]
#                                          [--magmatism off] [--magmatism-extraction off] [--magmatism-Ts0 1080]
#                                          [--preset-initial-temperature path/to/preliminary]
#       the command line options override the SVG, e.g. for resolution tests (--dx, --dz in km)

functions_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'functions') #folder with svg2interfaces.py and mandyoc_layers.py
sys.path.insert(0, functions_path) #first, so other copies of the modules are not imported
from svg2interfaces import layers_from_svg, write_interfaces, read_domain, points_from_spacing
from mandyoc_layers import MandyocLayer, Basalt

parser = argparse.ArgumentParser(description='Create the Mandyoc input files of a scenario drawn in an SVG file')
parser.add_argument('svg_file', help='SVG file with the scenario')
parser.add_argument('--dx', type=float, help='horizontal resolution [km] (default: from the domain shape of the SVG)')
parser.add_argument('--dz', type=float, help='vertical resolution [km] (default: from the domain shape of the SVG)')
parser.add_argument('--regime', choices=['rifting', 'compression', 'accordion', 'quiescence'],
                    help='tectonic regime of the numerical experiment (default: from the SVG, or rifting)')
parser.add_argument('--velocity', type=float, help='velocity at the boundary [cm/yr] (default: from the SVG, or 1 cm/yr)')
parser.add_argument('--moving-boundary', choices=['left', 'right'], help='boundary that moves (default: from the SVG, or right)')
parser.add_argument('--quiescence-time', type=float, help='duration of the quiescence [Myr] in the accordion and quiescence regimes '
                                                          '(default: from the SVG, or 0 Myr)')
parser.add_argument('--time-max', type=float, help='duration of the simulation [Myr] (default: from the SVG, or 60 Myr in rifting and '
                                                   'compression, the end of the second extension in accordion and the quiescence time '
                                                   '(or 1000 Myr) in quiescence; in the accordion regime it sets the end of the second extension)')
parser.add_argument('--magmatism', choices=['on', 'off'], help='melting in Mandyoc (default: from the SVG, or off)')
parser.add_argument('--magmatism-extraction', choices=['on', 'off'],
                    help='extraction of the melt to the basalt layer below the sticky air (default: from the SVG, or off)')
parser.add_argument('--magmatism-Ts0', type=float, help='magmatism_Ts0 in param.txt [oC] (default: from the SVG, or 1080)')
parser.add_argument('--preset-initial-temperature', help='folder (or _output_temperature.nc) of a preliminary scenario: the horizontal '
                                                         'mean of its final temperature, adapted to the lithosphere of the SVG, is the '
                                                         'initial temperature (default: from the SVG, or none: conductive model)')
args = parser.parse_args()

svg_file = args.svg_file
if(not os.path.isfile(svg_file)):
    sys.exit(f"SVG file not found: {svg_file}")

domain = read_domain(svg_file)
if(domain is None):
    sys.exit(f"No shape labelled 'domain' in {svg_file}. Draw a rectangle over the model domain, label it 'domain' "
             "and write Lx, Lz, dx and dz in its description (see svg2interfaces.py).")

# total model horizontal and vertical extent (m)
Lx = domain['Lx']
Lz = domain['Lz']
# number of points in horizontal and vertical direction
Nx = points_from_spacing(Lx, args.dx * 1.0e3, 'dx') if(args.dx) else domain['Nx']
Nz = points_from_spacing(Lz, args.dz * 1.0e3, 'dz') if(args.dz) else domain['Nz']

##############################################################################
# Numerical experiment: command line > SVG (domain description) > default
##############################################################################

def parse_number_with_unit(text, units, name):
    '''Read a number with an optional unit, e.g. "1 cm/yr", and return it in the first unit of units ({unit: factor})'''
    match = re.fullmatch(r"\s*([-+0-9.eE]+)\s*([A-Za-z/]*)\s*", text)
    if(match is None or (match.group(2) and match.group(2) not in units)):
        sys.exit(f"Could not read {name} = {text} in {svg_file}. Units: {', '.join(units)}")
    try:
        number = float(match.group(1))
    except ValueError:
        sys.exit(f"Could not read {name} = {text} in {svg_file}.")
    return number * units[match.group(2)] if(match.group(2)) else number

experiment = domain.get('experiment', {})
experiment_source = {}
def experiment_setting(name, cli_value, default):
    if(cli_value is not None):
        experiment_source[name] = 'command line'
        return cli_value
    if(name in experiment):
        experiment_source[name] = 'SVG'
        return experiment[name]
    experiment_source[name] = 'default'
    return default

regime = experiment_setting('regime', args.regime, 'rifting').strip().lower()
if(regime not in ['rifting', 'compression', 'accordion', 'quiescence']):
    sys.exit(f"Unknown regime '{regime}' in {svg_file}. Use rifting, compression, accordion or quiescence.")

velocity = experiment_setting('velocity', args.velocity, 1.0) #cm/yr
if(isinstance(velocity, str)):
    velocity = parse_number_with_unit(velocity, {'cm/yr': 1.0, 'cm/y': 1.0, 'mm/yr': 0.1, 'mm/y': 0.1}, 'velocity')

moving_boundary = experiment_setting('moving_boundary', args.moving_boundary, 'right').strip().lower()
if(moving_boundary not in ['left', 'right']):
    sys.exit(f"Unknown moving_boundary '{moving_boundary}' in {svg_file}. Use left or right.")

time_max_setting = experiment_setting('time_max', args.time_max, None) #Myr
if(isinstance(time_max_setting, str)):
    time_max_setting = parse_number_with_unit(time_max_setting, {'Myr': 1.0, 'My': 1.0, 'kyr': 1.0e-3, 'yr': 1.0e-6}, 'time_max')

quiescence_time = experiment_setting('quiescence_time', args.quiescence_time, 0.0) #Myr
if(isinstance(quiescence_time, str)):
    quiescence_time = parse_number_with_unit(quiescence_time, {'Myr': 1.0, 'My': 1.0, 'kyr': 1.0e-3, 'yr': 1.0e-6}, 'quiescence_time')
if(quiescence_time < 0):
    sys.exit(f"quiescence_time must not be negative ({quiescence_time:g} Myr).")
if(regime in ['rifting', 'compression'] and experiment_source['quiescence_time'] != 'default'):
    print(f"Warning: quiescence_time is not used in the {regime} regime.")

print(f"Numerical experiment: {regime} ({experiment_source['regime']}), velocity = {velocity:g} cm/yr ({experiment_source['velocity']}), "
      f"moving boundary: {moving_boundary} ({experiment_source['moving_boundary']})")

x = np.linspace(0, Lx, Nx)
z = np.linspace(Lz, 0, Nz)
X, Z = np.meshgrid(x, z)
dz = Lz / (Nz - 1)

#Keep a copy of the SVG in the scenario folder (it is zipped with the input files)
if(not os.path.exists(os.path.basename(svg_file)) or not os.path.samefile(svg_file, os.path.basename(svg_file))):
    shutil.copy(svg_file, os.path.basename(svg_file))

#Labels of the shapes in the SVG used to build the initial temperature, velocity bc and plots
asthenosphere_label = 'asthenosphere'
seed_label = 'weak_seed'

#Vertical resolution used to locate the interfaces in the SVG (m)
dz_svg = 100.0

layers = layers_from_svg(svg_file, Lx, Lz, Nx, dz=dz_svg) #from bottom to top; layer.interface is the depth of the layer base (m)

if(layers[-1].rheology_name != 'air'):
    sys.exit(f"The top layer of {svg_file} must be the sticky air (found '{layers[-1].layer_label}').")
if(len(layer_indices(layers, asthenosphere_label)) == 0):
    sys.exit(f"No shape labelled '{asthenosphere_label}' in {svg_file}.")

#Basalt layer with zero thickness just below the sticky air.
#It can not be drawn in the SVG and receives the melt extracted by Mandyoc (magmatic_layer in param.txt)

#Magmatism (command line > SVG > default: off, off, 1080 oC)
def on_off_setting(name, cli_value):
    value = str(experiment_setting(name, cli_value, 'off')).strip().lower()
    if(value not in ['on', 'off']):
        sys.exit(f"{name} must be on or off (found '{value}' in {svg_file}).")
    return value

magmatism = on_off_setting('magmatism', args.magmatism)
magmatism_extraction = on_off_setting('magmatism_extraction', args.magmatism_extraction)
magmatism_Ts0 = experiment_setting('magmatism_Ts0', args.magmatism_Ts0, 1080.0) #oC
if(isinstance(magmatism_Ts0, str)):
    magmatism_Ts0 = parse_number_with_unit(magmatism_Ts0, {'oC': 1.0, 'C': 1.0}, 'magmatism_Ts0')
magmatism_Ts0 = float(magmatism_Ts0)
if(magmatism_extraction == 'on' and magmatism == 'off'):
    print("Warning: magmatism_extraction is on but magmatism is off.")
print(f"Magmatism: {magmatism} ({experiment_source['magmatism']}), extraction: {magmatism_extraction} "
      f"({experiment_source['magmatism_extraction']}), Ts0 = {magmatism_Ts0:g} oC ({experiment_source['magmatism_Ts0']})")

#Initial temperature from a preliminary scenario (command line > SVG > none: conductive model).
#A relative path in the SVG is relative to the folder of the SVG file.
preset_initial_temperature = experiment_setting('preset_initial_temperature', args.preset_initial_temperature, None)
if(preset_initial_temperature is not None):
    preset_initial_temperature = os.path.expanduser(str(preset_initial_temperature).strip())
    if(experiment_source['preset_initial_temperature'] == 'SVG' and not os.path.isabs(preset_initial_temperature)):
        preset_initial_temperature = os.path.join(os.path.dirname(os.path.abspath(svg_file)), preset_initial_temperature)
    preset_initial_temperature = os.path.normpath(preset_initial_temperature)
    if(not os.path.exists(preset_initial_temperature)):
        sys.exit(f"preset_initial_temperature not found: {preset_initial_temperature}")
    print(f"Initial temperature from the preliminary scenario {preset_initial_temperature} ({experiment_source['preset_initial_temperature']})")

if(magmatism == 'on' and magmatism_extraction == 'on'):
    add_basalt_layer = True
else:
    add_basalt_layer = False

if(add_basalt_layer):
    basalt = MandyocLayer('basalt', Basalt,
                          density=2900.0,
                          effective_viscosity_scale_factor=1.0,
                          radiogenic_heat_production=9.0e-11,
                          Nx=Nx)
    basalt.interface = np.copy(layers[-1].interface) #base of basalt = base of sticky air
    basalt.base_depth = basalt.interface.max()
    basalt.title = 'Basalt'
    basalt.color = 'xkcd:dark grey blue'
    layers.insert(len(layers)-1, basalt)

#layer below the sticky air
magmatic_layer = len(layers) - 2

#Index of the layer at each node of the Mandyoc grid (Nz, Nx)
grid_layers = layer_grid(layers, z)

#Thickness of sticky air and lithosphere along x (m)
thickness_sa = layers[-1].interface #base of the sticky air
idx_ast = max(layer_indices(layers, asthenosphere_label))
thickness_litho = layers[idx_ast+1].interface - thickness_sa #from the surface to the top of the asthenosphere

#Position of the vertical profile used in the plots (temperature and YSE) and in the scenario infos
# x_profile = Lx / 2.0
x_profile = Lx / 4.0
idx_profile = find_nearest(x, x_profile)
thickness_sa_profile = thickness_sa[idx_profile]

#Temperature and YSE profiles in the plot: one for each lithospheric domain (e.g. oceanic, non-cratonic, cratonic),
#i.e. each region where the lithosphere has a constant thickness over at least min_profile_width.
#Each profile is at the centre of the widest region with that thickness (outside the weak seed) and is labelled with the title of the
#lithospheric mantle unit above the asthenosphere there.
min_profile_width = 0.05 * Lx
profile_colors = ['tab:orange', 'tab:purple', 'tab:green', 'tab:blue', 'r', 'k'] #the thickest lithosphere is the last one (black)

litho_km = np.round(thickness_litho / 1.0e3, 1)
#columns with the weak seed are left out, so the profiles do not cross it
has_seed = np.zeros(Nx, dtype=bool)
for i in layer_indices(layers, seed_label):
    has_seed |= slot_thickness(layers, i, slice(None)) > 0
#unit at the base of the lithosphere in each column (e.g. a mobile belt inside a cratonic keel is a domain of its own)
base_unit = np.array([next((layers[i].layer_label for i in range(idx_ast + 1, len(layers)) if slot_thickness(layers, i, ix) > 0), '')
                      for ix in range(Nx)])
litho_key = np.array([f"{litho_km[ix]}|{base_unit[ix]}" if(not has_seed[ix]) else '' for ix in range(Nx)])
edges = np.r_[0, np.nonzero(litho_key[1:] != litho_key[:-1])[0] + 1, Nx]
regions = {} #(lithosphere thickness, unit at its base): (first column, last column + 1)
for begin, end in zip(edges[:-1], edges[1:]):
    key = (litho_km[begin], base_unit[begin])
    if(litho_key[begin] != '' and (end - begin) * Lx / (Nx - 1) >= min_profile_width):
        if((key not in regions) or (end - begin > np.diff(regions[key])[0])):
            regions[key] = (begin, end)

profiles = [] #(column, label, color)
colors = profile_colors[-len(regions):] if(len(regions) <= len(profile_colors)) else profile_colors
for k, ((thickness, unit), (begin, end)) in enumerate(sorted(regions.items())):
    ix = (begin + end - 1) // 2
    i_mantle = next((i for i in range(idx_ast + 1, len(layers)) if slot_thickness(layers, i, ix) > 0), idx_ast + 1)
    name = layers[i_mantle].title.replace(' lith. mantle', '').replace(' lithospheric mantle', '')
    profiles.append((ix, f"{name} ({thickness:.0f} km)", colors[k % len(colors)]))
if(len(profiles) == 0):
    profiles = [(idx_profile, f"x = {x_profile/1.0e3:.0f} km", 'r')]

##############################################################################
# Save interfaces.txt to be used in Mandyoc code
##############################################################################

write_interfaces(layers, "interfaces.txt")

##############################################################################
# Creating parameter file
##############################################################################

# high_kappa_in_asthenosphere = True
high_kappa_in_asthenosphere = False #default

#Convergence criteria
denok                            = 1.0e-14
particles_per_element            = 40

#Surface constrains
sp_surface_tracking              = True
# sp_surface_tracking              = False
sp_surface_processes             = False


#External inputs: bc velocity, velocity field, precipitation and
#climate change

#Numerical experiment (regime, velocity and moving_boundary are read above from the command line or the SVG):
#   rifting: extension, the moving boundary moves outwards with velocity
#   compression: convergence, the moving boundary pushes the lithosphere inwards with velocity
#   accordion (as in M300): the boundary velocity of the rifting regime is scaled in time by scale_bcv.txt
#       0 - ti_convergence: extension
#       ti_convergence - ti_quiescence: convergence (velocity x -1)
#       ti_quiescence - tf_quiescence: quiescence (velocity x 0.01)
#       tf_quiescence - time_max: extension again (dt_rifting2)
#   quiescence: no velocity imposed at the boundaries (velocity_from_ascii = False)
#The quiescence time (quiescence_time, 0 Myr by default) sets dt_quiescence. Without time_max, the duration of the simulation is:
#   rifting and compression: 60 Myr
#   accordion: tf_quiescence + dt_rifting2
#   quiescence: quiescence_time, or 1000 Myr if it is 0
velocity_from_ascii              = (regime != 'quiescence')
variable_bcv                     = (regime == 'accordion')

#time constrains
dt_quiescence = quiescence_time #Myr (e.g. 300 in M300)
if(variable_bcv == True):
    ti_convergence = 25.0
    ti_quiescence = 75.0
    tf_quiescence = ti_quiescence + dt_quiescence
    dt_rifting2 = 60.0
    if(time_max_setting is not None): #time_max sets the duration of the second extension
        dt_rifting2 = time_max_setting - tf_quiescence
        if(dt_rifting2 <= 0):
            sys.exit(f"time_max = {time_max_setting:g} Myr ends before the second extension starts ({tf_quiescence:g} Myr).")

    time_max = (tf_quiescence + dt_rifting2)*1.0e6 #210.0e6
elif(time_max_setting is not None):
    time_max = time_max_setting * 1.0e6
elif(regime == 'quiescence'):
    time_max = dt_quiescence*1.0e6 if(dt_quiescence > 0) else 1000.0e6 #default: 1 Gyr
else: #rifting and compression
    time_max = 60.0e6

dt_max                           = 10.0e3 #default
step_print                       = 50 #25

if(sp_surface_processes == True):
    precipitation_profile_from_ascii = True #False
    climate_change_from_ascii        = True #False
else:
    precipitation_profile_from_ascii = False
    climate_change_from_ascii        = False 

#step files
print_step_files                 = True

checkered = False

rheol = 19

#velocity bc
top_normal_velocity                 = 'fixed'         # ok
top_tangential_velocity             = 'free '         # ok
bot_normal_velocity                 = 'fixed'         # ok
bot_tangential_velocity             = 'free'         # ok
left_normal_velocity                = 'fixed'         # ok
left_tangential_velocity            = 'fixed'         # ok
right_normal_velocity               = 'fixed'         # ok
right_tangential_velocity           = 'fixed'         # ok

# periodic_boundary = True
periodic_boundary = False

if(periodic_boundary == True):
    left_normal_velocity                = 'free'         # ok
    left_tangential_velocity            = 'free '         # ok
    right_normal_velocity               = 'free'         # ok
    right_tangential_velocity           = 'free'         # ok

#temperature bc
top_temperature                     = 'fixed'         # ok
bot_temperature                     = 'fixed'         # ok
# left_temperature                    = 'free'          # ok
left_temperature                    = 'fixed'          # ok
# right_temperature                   = 'free'          # ok
right_temperature                   = 'fixed'          # ok

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
denok                               = {denok}       # default is 1.0E-4
particles_per_element               = {particles_per_element}          # default is 81
surface_particles_per_element       = 40           # default is 2
particles_perturb_factor            = 0.7           # default is 0.5 [values are between 0 and 1]
rtol                                = 1.0e-7        # the absolute size of the residual norm (relevant only for iterative methods), default is 1.0E-5
RK4                                 = Euler         # default is Euler [Euler/Runge-Kutta]
Xi_min                              = 1.0e-5       # default is 1.0E-14
random_initial_strain               = 0.3           # default is 0.0
pressure_const                      = -1.0          # default is -1.0 (not used) - useful only in horizontal 2D models
initial_dynamic_range               = True         # default is False [True/False]
periodic_boundary                   = False         # default is False [True/False]
high_kappa_in_asthenosphere         = {high_kappa_in_asthenosphere}         # default is False [True/False]
K_fluvial                           = 2.0e-7        # default is 2.0E-7
m_fluvial                           = 1.0           # default is 1.0
sea_level                           = 0.0           # default is 0.0
basal_heat                          = 0.0          # default is -1.0
# Surface processes
sp_surface_tracking                 = {sp_surface_tracking}         # default is False [True/False]
sp_surface_processes                = {sp_surface_processes}        # default is False [True/False]
plot_sediment                       = False         # default is False [True/False]
a2l                                 = True          # default is True [True/False]
free_surface_stab                   = True         # default is True [True/False]
theta_FSSA                          = 0.5          # default is 0.5 (only relevant when free_surface_stab = True)
# Time constrains
step_max                            = 800000        # Maximum time-step of the simulation
time_max                            = {time_max}    # Maximum time of the simulation [years]
dt_max                              = {dt_max}      # Maximum time between steps of the simulation [years]
step_print                          = {step_print}  # Make file every <step_print>
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
n_interfaces                        = {len(layers)-1}           # Number of interfaces int the interfaces.txt file
variable_bcv                        = {variable_bcv} #False         # default is False [True/False]
temperature_from_ascii              = True         # default is False [True/False]
velocity_from_ascii                 = {velocity_from_ascii} #False      # default is False [True/False]
binary_output                       = False         # default is False [True/False]
sticky_blanket_air                  = True         # default is False [True/False]
# precipitation_profile_from_ascii    = {precipitation_profile_from_ascii}         # default is False [True/False]
# climate_change_from_ascii           = {climate_change_from_ascii}         # default is False [True/False]
print_step_files                    = {print_step_files}          # default is True [True/False]
checkered                           = {checkered}         # Print one element in the print_step_files (default is False [True/False])
geoq                                = on            # ok
geoq_fac                            = 100.0           # ok
# Physical parameters
temperature_difference              = 1500.         # ok
thermal_expansion_coefficient       = 3.28e-5       # ok
thermal_diffusivity_coefficient     = 1.0e-6 #0.75e-6       #default is 1.0e-6        # ok
gravity_acceleration                = 10.0          # ok
density_mantle                      = 3300.         # ok
external_heat                       = 0.0e-12       # ok
heat_capacity                       = 1250.         # ok #default is 1250
non_linear_method                   = on            # ok
adiabatic_component                 = on            # ok
radiogenic_component                = on            # ok
magmatism                           = {magmatism}           # ok
magmatism_extraction = {magmatism_extraction}
magmatism_Ts0 = {magmatism_Ts0} #oC
export_lithology = True
magmatic_layer = {magmatic_layer}
# Velocity boundary conditions
top_normal_velocity                 = fixed         # ok
top_tangential_velocity             = free          # ok
bot_normal_velocity                 = fixed         # ok
bot_tangential_velocity             = free          # ok
left_normal_velocity                = {left_normal_velocity}         # ok
left_tangential_velocity            = {left_tangential_velocity}          # ok
right_normal_velocity               = {right_normal_velocity}         # ok
right_tangential_velocity           = {right_tangential_velocity}         # ok
surface_velocity                    = 0.0e-2        # ok
multi_velocity                      = False         # default is False [True/False]
# Temperature boundary conditions
top_temperature                     = {top_temperature}         # ok
bot_temperature                     = {bot_temperature}         # ok
left_temperature                    = {left_temperature}         # ok
right_temperature                   = {right_temperature}         # ok
rheology_model                      = {rheol}             # ok

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

# scenario = '/Doutorado/cenarios/mandyoc/stable/lit80km/stable_PT200_rheol19_c1250_C1_HprodAst/'
# scenario_name = scenario.split('/')[-2]
# print(scenario_name)

DeltaT = 0
# DeltaT = 200
# DeltaT = 290
# DeltaT = 350

#preset: initial temperature from the final thermal state of a preliminary scenario (preset_initial_temperature)
preset = (preset_initial_temperature is not None)

if(preset == False):
    ccapacity = 1250*1.0 #937.5=75% #J/kg/K? #DEFAULT

    # TP = 1262 #mantle potential temperature
    # TP = 1350
    TP = 1400
    # TP = 1450

    # kappa = 0.75*1.0e-6 #thermal diffusivity
    kappa = 1.0e-6 #thermal diffusivity

    #Radiogenic heat production of the layers at each node
    H = layer_property(layers, grid_layers, 'radiogenic_heat_production') #(Nz, Nx)

    #The conductive model is 1D (vertical): it is solved only for the distinct columns
    #(thickness of sticky air, thickness of lithosphere and radiogenic heat profile)
    columns = np.vstack([thickness_sa, thickness_litho, H])
    columns_unique, idx_columns = np.unique(columns, axis=1, return_inverse=True)
    idx_columns = idx_columns.ravel()
    sa_col, litho_col, H = columns_unique[0], columns_unique[1], columns_unique[2:]
    print(f"Solving the conductive model for {len(sa_col)} distinct column(s)")

    zcol = z[:, None]
    T = 1300 * (zcol - sa_col) / (litho_col)  # Temperature

    # T = 1300 * (z - thickness_sa) / (130*1.0E3)  # Temperature of 1300 isotherm bellow the lithosphere

    Ta = (TP / np.exp(-10 * 3.28e-5 * (zcol - sa_col) / ccapacity)) + DeltaT #Temperature profile for asthenosphere

    T[T < 0.0] = 0.0 #forcing negative temperatures to 0
    cond1 = Ta<T #VICTOR - selecting where asthenosphere temperature is lower than lithospheric temperature
    T[T > Ta] = Ta[T > Ta] #apply the temperature of asthenosphere (Ta) where temperature (T) is greater than Ta

    #Conductive model
    Taux = np.copy(T)

    t = 0
    dt = 5000
    dt_sec = dt * 365 * 24 * 3600
    # cond = (z > thickness_sa + thickness_litho) | (T == 0)  # (T > 1300) | (T == 0) #OLD
    cond = cond1 | (T == 0)  # (T > 1300) | (T == 0) #VICTOR
    # dz = Lz / (Nz - 1)
    while t < 2000.0e6:
        T[1:-1] += (
            kappa * dt_sec * ((T[2:] + T[:-2] - 2 * T[1:-1]) / dz ** 2)
            + H[1:-1] * dt_sec / ccapacity
        )
        T[cond] = Taux[cond]

        t = t + dt

    T = T[:, idx_columns] #(Nz, Nx)

    # Save the initial temperature file
    np.savetxt("input_temperature_0.txt", np.reshape(T, (Nx * Nz)), header="T1\nT2\nT3\nT4")

else:
    #Horizontal mean of the final temperature of the preliminary scenario
    depth_preset, T_preset, thickness_sa_preset, thickness_litho_preset, preset_source = read_preset_temperature(preset_initial_temperature)

    #Lithosphere of the preliminary scenario (from its interfaces.txt): the columns with this thickness use the
    #profile itself and the others are adapted to their lithosphere. Without interfaces.txt, the most common
    #lithosphere thickness of this scenario is used.
    if(thickness_litho_preset is None):
        values, counts = np.unique(np.round(thickness_litho / 1.0e3), return_counts=True)
        thickness_litho_preset = values[np.argmax(counts)] * 1.0e3
        print(f"Warning: interfaces.txt of the preliminary scenario not found; its lithosphere is assumed to be "
              f"{thickness_litho_preset/1.0e3:.0f} km thick (the most common in {svg_file}).")

    #thickness of the lithospheric mantle (olivine rheologies above the asthenosphere) along x, to find the Moho
    mantle_rheologies = ['wet_olivine', 'dry_olivine']
    thickness_mantle = np.zeros(Nx)
    for i in range(idx_ast + 1, len(layers)):
        if(layers[i].rheology_name in mantle_rheologies):
            thickness_mantle += slot_thickness(layers, i, slice(None))

    T = adapt_preset_temperature(depth_preset, T_preset, thickness_sa_preset, thickness_litho_preset,
                                 z, thickness_sa, thickness_litho, thickness_mantle)
    print(f"Preset of initial temperature: {preset_source}")
    print(f"\tsticky air of the preliminary scenario: {thickness_sa_preset/1.0e3:.0f} km; "
          f"reference lithosphere: {thickness_litho_preset/1.0e3:.0f} km (other lithosphere thicknesses are adapted)")

    # Save the initial temperature file
    np.savetxt("input_temperature_0.txt", np.reshape(T, (Nx * Nz)), header="T1\nT2\nT3\nT4")

##############################################################################
# Boundary condition - velocity
##############################################################################
if(velocity_from_ascii == True):
    fac_air = 10.0e3

    idx_bc = 0 if(moving_boundary == 'left') else Nx - 1 #column of the moving boundary

    #thickness of sticky air at the moving boundary
    thickness_sa_bc = thickness_sa[idx_bc]
    #Reference thickness for the region with constant velocity at the boundaries (+ 20 km):
    #   'lithosphere': surface to the top of the asthenosphere, crust included (thickness_crat_bot in the double keel scripts)
    #   'lithospheric_mantle': mantle units (olivine rheologies) above the asthenosphere (thickness_mlit in generate_input_files_rft_melt.py)
    #taken where it is thickest in the rifting and accordion regimes (as in M300) and at the moving boundary
    #in the compression regime (e.g. the oceanic plate pushed towards a continent)
    h_v_reference = 'lithosphere'
    # h_v_reference = 'lithospheric_mantle'

    if(h_v_reference == 'lithosphere'):
        thickness_ref_x = thickness_litho
    else:
        mantle_rheologies = ['wet_olivine', 'dry_olivine']
        thickness_ref_x = np.array([sum(slot_thickness(layers, i, ix) for i in range(idx_ast+1, len(layers))
                                        if layers[i].rheology_name in mantle_rheologies) for ix in range(Nx)])
    thickness_ref_bc = thickness_ref_x[idx_bc] if(regime == 'compression') else thickness_ref_x.max()

    # 1 cm/year
    vL = (0.5*velocity/100) / (365 * 24 * 3600)  # m/s

    h_v_const = thickness_ref_bc + 20.0e3  #thickness with constant velocity
    ha = Lz - thickness_sa_bc - h_v_const  # difference

    vR = 2 * vL * (h_v_const + fac_air + ha) / ha  # this is to ensure integral equals zero

    VX = np.zeros_like(X)
    cond = (Z > h_v_const + thickness_sa_bc) & (X == 0)
    VX[cond] = vR * (Z[cond] - h_v_const - thickness_sa_bc) / ha

    cond = (Z > h_v_const + thickness_sa_bc) & (X == Lx)
    VX[cond] = -vR * (Z[cond] - h_v_const - thickness_sa_bc) / ha

    cond = X == Lx
    VX[cond] += +2 * vL

    cond = Z <= thickness_sa_bc - fac_air
    VX[cond] = 0

    #The profile above is extensional, with the right boundary moving outwards (2*vL = velocity) and the
    #return flow below h_v_const. It is changed according to the regime and the moving boundary
    #(the integral of the inflow remains zero):
    if(regime == 'compression'):
        if(moving_boundary == 'left'):
            VX = VX[:, ::-1] #mirror: the left boundary moves to the right, pushing the lithosphere inwards
        else:
            VX = -VX #reverse: the right boundary moves to the left
    elif(moving_boundary == 'left'):
        VX = -VX[:, ::-1] #extension with the left boundary moving outwards (to the left)

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

    if(variable_bcv == True):
        #Mandyoc multiplies the boundary velocity by each factor at the given time (Myr), one per time step
        if(dt_quiescence > 0):
            var_bcv = f""" 4
                        {ti_convergence} -1.0
                        {ti_quiescence} 0.01
                        {tf_quiescence} -100.0
                        {time_max/1.0e6} 0.01
                        """
        else: #no quiescence: convergence followed directly by extension
            var_bcv = f""" 3
                        {ti_convergence} -1.0
                        {ti_quiescence} -1.0
                        {time_max/1.0e6} 0.01
                        """

        # Create the parameter file
        with open("scale_bcv.txt", "w") as f:
            for line in var_bcv.split("\n"):
                line = line.strip()
                if len(line):
                    f.write(" ".join(line.split()) + "\n")

##############################################################################
# Surface processes
##############################################################################

if(sp_surface_processes == True):
    if(climate_change_from_ascii == True):
        #When climate effects will start to act - scaling to 1
        climate = f'''
                2
                0 0.0
                120 0.02
            '''

        with open('climate.txt', 'w') as f:
            for line in climate.split('\n'):
                line = line.strip()
                if len(line):
                    f.write(' '.join(line.split()) + '\n')

    if(precipitation_profile_from_ascii ==True):
        #Creating precipitation profile

        prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/(1))**6) #Lx km
        # prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/8)**6) #original
        # prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/(8*2))**6) #100 km
        # prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/(8*4))**6) #50 km

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




##############################################################################
# Plot of the initial temperature field with the layer interfaces
##############################################################################

plt.close()
fig, ax0 = plt.subplots(nrows=1, ncols=1, constrained_layout=True, figsize=(16, 6))

#depth relative to the surface at x_profile, as in numerical_setup.png
im = ax0.contourf(X / 1.0e3, (-Z + thickness_sa_profile) / 1.0e3, T,
                  levels=np.arange(0, np.max(T) + 100, 100), cmap='inferno')
isotherms = ax0.contour(X / 1.0e3, (-Z + thickness_sa_profile) / 1.0e3, T,
                        levels=[600, 1000, 1300], colors='w', linewidths=1.0, linestyles='--')
ax0.clabel(isotherms, fmt='%d °C', fontsize=10)

#layer interfaces (base of each layer)
for layer in layers[1:]:
    ax0.plot(x / 1.0e3, (-layer.interface + thickness_sa_profile) / 1.0e3, color='k', lw=0.6)

ax0.set_xlim(0, Lx / 1.0e3)
ax0.set_ylim(-Lz/1000+thickness_sa_profile/1000, 0+thickness_sa_profile/1000)
ax0.tick_params(labelsize=12)
ax0.set_xlabel("Distance [km]", fontsize=14)
ax0.set_ylabel("Depth [km]", fontsize=14)
ax0.set_title(f"Initial temperature field ({os.path.basename(svg_file)})", fontsize=14)
cbar = fig.colorbar(im, orientation='vertical', ax=ax0, pad=0.01)
cbar.set_label("Temperature [°C]", fontsize=12)
cbar.ax.tick_params(labelsize=10)
fig.savefig('initial_temperature.png', dpi=300)
plt.close()

##############################################################################
#
# Creating a single plot with scenario infos
#
##############################################################################

plt.close()
fig, axs = plt.subplots(1, 1, figsize = (14, 6))
ylimplot = [-Lz/1000+thickness_sa_profile/1000, 0+thickness_sa_profile/1000]

#plot scenario layers with the colours of the SVG
for i, layer in enumerate(layers):
    top = layers[i+1].interface if i < len(layers)-1 else np.zeros(Nx)
    axs.fill_between(x/1000, (-layer.interface+thickness_sa_profile)/1000, (-top+thickness_sa_profile)/1000, color=layer.color, lw=0)

dx = Lx/(Nx-1)
axs.set_xlim(0, Lx/1000)
axs.set_xticks([])
axs.set_yticks([])
axs.set_ylim(ylimplot)
axs.set_xlabel(f"Distance = {Lx/1000:.0f} km\nResolution = {dx/1000:.0f} km", fontsize=16)
axs.set_ylabel(f"Depth = {Lz/1000:.0f} km\nResolution = {dz/1000:.0f} km", fontsize=16)

#plotting ghost points to create a legend to the layers (one entry per label, from top to bottom)
#thickness h in the legend:
#   bodies (units that do not reach both sides of the domain, e.g. a basin): maximum thickness (depocentre)
#   bottom unit (asthenosphere): range of thickness along x
#   other layers: most common thickness along x (their reference thickness)
legend_layers = {}
for layer in layers[::-1]:
    if(layer.layer_label != seed_label):
        legend_layers.setdefault(layer.layer_label, layer)

def legend_thickness(label):
    th = layer_thickness(layers, label, slice(None)) / 1.0E3 #km, along x
    if(th[0] <= 0 or th[-1] <= 0):
        return f"{th.max():.0f} km"
    if(label == layers[0].layer_label and np.round(th.min()) != np.round(th.max())):
        return f"{th.min():.0f}-{th.max():.0f} km"
    values, counts = np.unique(np.round(th[th > 0], 1), return_counts=True)
    return f"{values[np.argmax(counts)]:.0f} km" if(len(values)) else "0 km"

legend_elements = []
for label, layer in legend_layers.items():
    legend_label = f"{layer.title}\n{layer.effective_viscosity_scale_factor:g} x {layer.rheology_name.replace('_', ' ')}\n"+fr"$\rho$ = {layer.density:.0f} kg/m³"+f"\n$h$ = {legend_thickness(label)}"
    legend_elements.append(Patch(facecolor=layer.color, edgecolor='black', label=legend_label))

fig.subplots_adjust(bottom=0.25)

leg = axs.legend(handles=legend_elements,
    ncol=len(legend_elements),
    loc='lower center',
    bbox_to_anchor=(0.5, -0.45),
    frameon=False,
    title='Layers properties',
    title_fontsize=15,
    fontsize=8,
    columnspacing=1.0,
)

#Indicating weak seed position (one annotation per body, i.e. per group of contiguous columns)
idx_seed = layer_indices(layers, seed_label)
if(len(idx_seed) > 0):
    seed_bases = np.array([layers[i].interface for i in idx_seed])
    seed_tops = np.array([layers[i+1].interface if i < len(layers)-1 else np.zeros(Nx) for i in idx_seed])
    seed_present = seed_bases > seed_tops
    seed_top = np.where(seed_present, seed_tops, np.inf).min(axis=0)
    seed_base = np.where(seed_present, seed_bases, -np.inf).max(axis=0)
    seed_cols = seed_present.any(axis=0)
    seed_title = layers[idx_seed[0]].title.lower()

    edges = np.diff(np.r_[0, seed_cols.astype(int), 0])
    for begin, end in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        cols = slice(begin, end)
        x_seed = np.mean(x[cols])
        width_seed = (end - begin) * dx
        thickness_seed = np.max(seed_base[cols] - seed_top[cols])
        depth_seed = np.mean((seed_base[cols] + seed_top[cols]) / 2.0)
        # axs.plot(x_seed/1000, -(depth_seed - thickness_sa_profile)/1000, 'x', color='k', lw=1.0)
        axs.text(x_seed/1000, -(np.max(seed_base[cols]) - thickness_sa_profile)/1000 - 2, f'{seed_title}\n{width_seed/1000:.0f}x{thickness_seed/1000:.0f} km²', color='k', fontsize=12, ha='center', va='top')


#Temperature profiles
axt = axs.inset_axes((0.205,
                        0,
                        0.18,
                        1))

for ix, label, color in profiles:
    axt.plot(T[:, ix], (-z + thickness_sa_profile) / 1.0e3, "-", color=color, label=label)
    #position of the profile on top of the section
    axs.plot(x[ix] / 1.0e3, thickness_sa_profile / 1.0e3 * 0.15, 'v', color=color, ms=6)
axt.legend(loc='lower left', fontsize=8, framealpha=0.9)

axt.grid(visible=True, axis='x',which='both',ls='--',color='red',alpha=0.3)
axt.set_ylim(ylimplot)
axt.set_yticks([])
axt.set_xticks(np.linspace(0,1800,7))
axt.patch.set_alpha(0)
axt.xaxis.set_ticks_position('top')
axt.set_yticks([])
axt.tick_params(labelsize=8)
axt.xaxis.label.set_color('red')
axt.tick_params(axis='x', colors='k')
axt.spines['left'].set_visible(False)
axt.spines['right'].set_visible(False)
axt.set_title('Temperature [°C]',color='k')
# axt.legend(loc='lower left', fontsize=10, framealpha=0.9)
#axt.spines['bottom'].set_visible(False)

################################
#    Yield Strength Envelope   #
################################

sr = 1.0E-15 #strain rate - s-1
# sr = 1.0E-14
R = 8.314 #gas constant - J K−1 mol−1
g = 10.0

def yield_strength_envelope(ix):
    '''YSE [Pa] along the vertical profile at column ix'''
    layers_profile = grid_layers[:, ix]
    Cnc = layer_property(layers, layers_profile, 'effective_viscosity_scale_factor')
    rhonc = layer_property(layers, layers_profile, 'density')
    Anc = layer_property(layers, layers_profile, 'pre_exponential_constant')
    nnc = layer_property(layers, layers_profile, 'power_law_exponent')
    Qnc = layer_property(layers, layers_profile, 'activation_energy')
    Vnc = layer_property(layers, layers_profile, 'activation_volume')

    Pnc = rhonc[::-1].cumsum()[::-1]*g*dz

    phi = 15.0*np.pi/180.0
    c0 = 20.0E6
    sigmanc_max = c0 * np.cos(phi) + Pnc * np.sin(phi)

    TKnc = T[:, ix] + 273

    viscnc = Cnc * Anc**(-1./nnc) * sr**((1.0-nnc)/nnc)*np.exp((Qnc + Vnc*Pnc)/(nnc*R*TKnc))
    sigmanc_v = viscnc * sr
    condnc = sigmanc_v>sigmanc_max
    sigmanc_v[condnc]=sigmanc_max[condnc]
    return sigmanc_v


axsg = axs.inset_axes((0.605,
                       0,
                       0.13,
                       1))
for ix, label, color in profiles:
    axsg.plot(yield_strength_envelope(ix)/1e9, -(z-thickness_sa_profile)/1e3, '-', color=color, label=label)

axsg.grid(visible=True, axis='x',which='both',ls='--',color='gray',alpha=0.8)
axsg.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axsg.set_xlim(-0.1,1.1)
axsg.set_ylim(ylimplot)
axsg.set_yticks([])
axsg.patch.set_alpha(0)
axsg.xaxis.set_ticks_position('top')
axsg.spines['left'].set_visible(False)
axsg.spines['right'].set_visible(False)
axsg.set_title('$\sigma_{YSE}$ [GPa]')
axsg.tick_params(labelsize=8)

############################
# Effective friction angle #
############################

axsf = axs.inset_axes((0.800,
                       0,
                       0.10,
                       0.30))
axsf.patch.set_alpha(0.4)

# xdata = np.array([0, 0.05, 1.05, 1.1])
xdata = np.array([0, 0.25, 0.75, 1.0])
ydata = np.array([15, 15, 2, 2])

fisize = 10
axsf.plot(xdata, ydata, 'k-')
axsf.set_xlim([0.10, 0.9])
axsf.set_ylim([0, 17])
axsf.set_xticks([0.25, 0.5, 0.75])
axsf.set_xticklabels([0.05, ' ', 1.05])
axsf.tick_params('x', top=True, labeltop=False)
axsf.xaxis.set_ticks_position('bottom')
axsf.set_xlabel(r"$\varepsilon$", fontsize=fisize)

axsf.set_yticks([2, 15])
axsf.set_yticklabels(['2°', '15°'])
axsf.set_ylabel('$\Phi_{\mathrm{eff}}$', fontsize=fisize)
axsf.tick_params(labelsize=fisize)

axsfC = axsf.twinx()
axsfC.patch.set_visible(False) #the background of the twin of an inset axes would hide the curve (matplotlib >= 3.11)
axsfC.set_ylim([0, 17])
axsfC.set_yticks([2, 15])
axsfC.set_yticklabels([4, 20])
axsfC.set_ylabel('Cohesion [MPa]', fontsize=fisize, rotation=90)
axsfC.tick_params(labelsize=fisize)

#plot velocity bc
if(velocity_from_ascii == True):
    #Velocity = Right side
    vr_plot = np.round(max(abs(VX[:, -1]* (100.0*365.0 * 24.0 * 3600.0))),0)
    vr_plot = vr_plot if(vr_plot > 0) else 1.0
    fac = 0.96
    axr = axs.inset_axes((fac,
                         0,
                         (1-fac)*2,
                         1))

    scale_veloc = 100.0*365.0*24.0*3600.0
    crt=0

    #Right side velocity arrows
    axr.fill_betweenx((-z + thickness_sa_profile) / 1.0e3, VX[:, -1]* (100.0*365.0 * 24.0 * 3600.0), 0, facecolor='none', edgecolor='k', linewidth=0, hatch='---') #only the hatch (with alpha=0 the hatch is also invisible in matplotlib >= 3.11)
    axr.set_ylim(ylimplot)
    axr.set_yticks([])
    axr.set_xticks([])
    axr.patch.set_alpha(0)
    axr.set_title(f'v = {velocity} [cm/y]', fontsize=8)
    # axr.xaxis.set_ticks_position('top')
    axr.tick_params(labelsize=8)
    axr.set_xlim([-vr_plot, vr_plot])
    axr.spines['left'].set_visible(False)
    axr.spines['right'].set_visible(False)
    axr.spines['bottom'].set_visible(False)

    # Left side velocity arrows
    vl_plot = np.round(max(abs(VX[:, 0]* (100*365.0 * 24.0 * 3600.0))), 0)
    vl_plot = vl_plot if(vl_plot > 0) else 1.0
    #inset centred on the left edge (as the right one), so v = 0 is on the boundary:
    #inflow is drawn inside the domain and outflow outside
    axl = axs.inset_axes((-(1-fac),
                          0,
                          (1-fac)*2,
                          1))
    
    axl.fill_betweenx((-z + thickness_sa_profile) / 1.0e3, VX[:, 0]* (100*365 * 24 * 3600), 0, facecolor='none', edgecolor='k', linewidth=0, hatch='---') #only the hatch (with alpha=0 the hatch is also invisible in matplotlib >= 3.11)

    axl.set_ylim(-Lz/1000+thickness_sa_profile/1000, 0+thickness_sa_profile/1000)
    axl.set_yticks([])
    axl.set_xticks([])
    axl.patch.set_alpha(0)
    # axl.set_title(f'v = {v} cm/y')
    # axl.xaxis.set_ticks_position('top')
    # axl.set_xlim([0, 2*velocity])
    axl.set_xlim([-vl_plot, vl_plot])
    axl.tick_params(labelsize=8)
    axl.spines['left'].set_visible(False)
    axl.spines['right'].set_visible(False)
    axl.spines['top'].set_visible(False)
    axl.spines['bottom'].set_visible(False)

figname = 'numerical_setup'
fig.savefig(f"{figname}.png", bbox_inches="tight", dpi=300)
plt.close()

##############################################################################
# Scenario infos
##############################################################################

#Description of the layers at x_profile
layers_infos = [f'Layers (bottom to top) from {svg_file}, thickness at x = {x_profile/1.0e3:.0f} km:']
for i, layer in enumerate(layers):
    layers_infos.append(f"\t{i}: {layer.layer_label} ({layer.rheology_name}): h = {slot_thickness(layers, i, idx_profile)/1.0e3:.1f} km, "
                        f"C = {layer.effective_viscosity_scale_factor}, rho = {layer.density} kg/m3, H = {layer.radiogenic_heat_production:.4g} W/kg")
layers_infos.append(f"\tlithosphere: {thickness_litho[idx_profile]/1.0e3:.1f} km")
layers_infos.append(f"\tmagmatic layer: {magmatic_layer} ({layers[magmatic_layer].layer_label})")
layers_infos.append("Temperature and YSE profiles in numerical_setup.png:")
for ix, label, color in profiles:
    layers_infos.append(f"\t{label}: x = {x[ix]/1.0e3:.0f} km")

print(f"N cores: {ncores}")
print('Domain parameters:')
print(f"\tLx: {Lx*1.0e-3} km")
print(f"\tLz: {Lz*1.0e-3} km")
print(f"\tNx: {Nx}")
print(f"\tNz: {Nz}")
print(f"Resolution dx x dz: {1.0e-3*Lx/(Nx-1)} x {1.0e-3*Lz/(Nz-1)} km2")
print(f'Time limit: {time_max/1.0e6} Myr')
if(variable_bcv == True):
    print(f"Extension: 0 - {ti_convergence} Myr")
    print(f"Convergence: {ti_convergence} - {ti_quiescence} Myr")
    print(f"Time of quiescence: {dt_quiescence} Myr ({ti_quiescence} - {tf_quiescence} Myr)")
    print(f"Second extension: {dt_rifting2} Myr ({tf_quiescence} - {time_max/1.0e6} Myr)")
for line in layers_infos:
    print(line)
print(f"Preset of initial temperature field: {preset}")
print(f"Surface process: {sp_surface_processes}")
print(f"Numerical experiment: {regime}")
print(f"Velocity field: {velocity_from_ascii}")
if(velocity_from_ascii==True):
    print(f"\tvelocity: {velocity:g} cm/yr, moving boundary: {moving_boundary}, constant velocity down to {h_v_const/1.0e3:.0f} km below the surface")
print(f"Variable velocity field: {variable_bcv}")
print(f"Climate change: {climate_change_from_ascii}")
print(f"Periodic Boundary: {periodic_boundary}")
print('Initial temperature field setup:')
print(f"\tPreset of initial temperature field: {preset}")
if(preset==False):
    print(f"\tIncrease in mantle basal temperature (Ta): {DeltaT} oC")
    print(f"\tAssumed mantle Potential Temperature for diffusive model: {TP} oC")
else:
    print(f"\tPreliminary scenario: {preset_source}")
    print(f"\tReference lithosphere of the preliminary scenario: {thickness_litho_preset/1.0e3:.0f} km")
print(f'magmatism: {magmatism}, extraction: {magmatism_extraction}, Ts0: {magmatism_Ts0:g} oC')
print(f'rheology model in param file: {rheol}')

#Save scenario infos
scenario_infos = ['SCENARIO INFOS:']
scenario_infos.append(' ')
scenario_infos.append('Name: ' + path[-1])
scenario_infos.append(f"N cores: {ncores}")
scenario_infos.append(' ')
scenario_infos.append('Domain parameters:')
scenario_infos.append(f"\tLx: {Lx*1.0e-3} km")
scenario_infos.append(f"\tLz: {Lz*1.0e-3} km")
scenario_infos.append(f"\tNx: {Nx}")
scenario_infos.append(f"\tNz: {Nz}")
scenario_infos.append(f"Resolution dx x dz: {1.0e-3*Lx/(Nx-1)} x {1.0e-3*Lz/(Nz-1)} km2")
scenario_infos.append(' ')
scenario_infos.append(f'Time limit: {time_max/1.0e6} Myr')
if(variable_bcv == True):
    scenario_infos.append(f"Extension: 0 - {ti_convergence} Myr")
    scenario_infos.append(f"Convergence: {ti_convergence} - {ti_quiescence} Myr")
    scenario_infos.append(f"Time of quiescence: {dt_quiescence} Myr ({ti_quiescence} - {tf_quiescence} Myr)")
    scenario_infos.append(f"Second extension: {dt_rifting2} Myr ({tf_quiescence} - {time_max/1.0e6} Myr)")
    scenario_infos.append(' ')
scenario_infos += layers_infos
scenario_infos.append(' ')
scenario_infos.append(f"Preset of initial temperature field: {preset}")
scenario_infos.append(f"Surface process: {sp_surface_processes}")
scenario_infos.append(f"Numerical experiment: {regime}")
scenario_infos.append(f"Velocity field: {velocity_from_ascii}")
if(velocity_from_ascii==True):
    scenario_infos.append(f"inital velocity: {vL*(365 * 24 * 3600)*2*100} cm/yr")
    scenario_infos.append(f"\tmoving boundary: {moving_boundary}, constant velocity down to {h_v_const/1.0e3:.0f} km below the surface")
scenario_infos.append(f"Variable velocity field: {variable_bcv}")
scenario_infos.append(f"Climate change: {climate_change_from_ascii}")
scenario_infos.append(f"Periodic Boundary: {periodic_boundary}")
scenario_infos.append('Initial temperature field setup:')
scenario_infos.append(f"\tPreset of initial temperature field: {preset}")
if(preset==False):
    scenario_infos.append(f"\tIncrease in mantle basal temperature (Ta): {DeltaT} oC")
    scenario_infos.append(f"\tAssumed mantle Potential Temperature for diffusive model: {TP} oC")
else:
    scenario_infos.append(f"\tPreliminary scenario: {preset_source}")
    scenario_infos.append(f"\tReference lithosphere of the preliminary scenario: {thickness_litho_preset/1.0e3:.0f} km")
scenario_infos.append(' ')
scenario_infos.append(f'magmatism: {magmatism}, extraction: {magmatism_extraction}, Ts0: {magmatism_Ts0:g} oC')
scenario_infos.append(f'rheology model in param file: {rheol}')

np.savetxt('infos_'+path[-1] + '.txt', scenario_infos, fmt="%s")

##############################################################################
#Creating run_scripts
##############################################################################

linux = False
mac = True
aguia = False
hypatia = True


#Initial strain of the layers given by label (-seed and -strain_seed options of Mandyoc)
#Layers not listed here get the random initial strain (random_initial_strain in param.txt)
strain_seed = {asthenosphere_label: 0.0,
               seed_label: 1.0}

seed_layers = [(i, strain) for label, strain in strain_seed.items() for i in layer_indices(layers, label)]
mandyoc_options = f"-seed {','.join(str(i) for i, strain in seed_layers)} -strain_seed {','.join(str(strain) for i, strain in seed_layers)}"


if(linux):
    run_linux = f'''
            #!/bin/bash
            MPI_PATH=$HOME/opt/petsc/arch-label-optimized/bin
            MANDYOC_PATH=$HOME/opt/mandyoc
            NUMBER_OF_CORES=20
            touch FD.out
            $MPI_PATH/mpirun -n $NUMBER_OF_CORES $MANDYOC_PATH/bin/mandyoc {mandyoc_options} | tee FD.out
        '''
    with open('run-linux.sh', 'w') as f:
        for line in run_linux.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')

if(mac):
    # claudio=True
    claudio=False
    dirname = '${PWD##*/}'

    if(claudio):
        PETSC_DIR = '/Users/claudiomora/Documents/petsc'
        PETSC_ARCH = 'arch-label-optimized/bin/mpiexec'
        MANDYOC = '/Users/claudiomora/Documents/mandyoc/bin/mandyoc'

        run_mac = f'''
                    #!/bin/bash
                    touch FD.out
                    {PETSC_DIR}/{PETSC_ARCH} -n 16 {MANDYOC} {mandyoc_options} | tee FD.log
                    
                    DIRNAME={dirname}

                    zip $DIRNAME.zip interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh
                    zip -u $DIRNAME.zip bc_velocity_*.txt
                    zip -u $DIRNAME.zip density_*.txt
                    zip -u $DIRNAME.zip heat_*.txt
                    zip -u $DIRNAME.zip pressure_*.txt
                    zip -u $DIRNAME.zip sp_surface_global_*.txt
                    zip -u $DIRNAME.zip strain_*.txt
                    zip -u $DIRNAME.zip temperature_*.txt
                    zip -u $DIRNAME.zip time_*.txt
                    zip -u $DIRNAME.zip velocity_*.txt
                    zip -u $DIRNAME.zip viscosity_*.txt
                    zip -u $DIRNAME.zip scale_bcv.txt
                    zip -u $DIRNAME.zip step*.txt
                    zip -u $DIRNAME.zip *.bin*
                    zip -u $DIRNAME.zip *.log
                    

                    #rm *.log
                    rm vel_bc*
                    rm velz*
                    rm bc_velocity*
                    rm velocity*
                    rm step*
                    rm temperature*
                    rm density*
                    rm viscosity*
                    rm heat*
                    rm strain_*
                    rm time*
                    rm pressure_*
                    rm sp_surface_global*
                    rm scale_bcv.txt
                    rm *.bin*

                '''
    else:
        run_mac = f'''
                #!/bin/bash
                MPI_PATH=/opt/miniconda3/envs/petsc-env/bin/
                MANDYOC_PATH=$HOME/opt/mandyoc
                NUMBER_OF_CORES=6
                touch FD.out
                $MPI_PATH/mpiexec -n $NUMBER_OF_CORES $MANDYOC_PATH/bin/mandyoc {mandyoc_options} | tee FD.out
                bash zipper.sh
                bash clean.sh
            '''
    with open('run_mac.sh', 'w') as f:
        for line in run_mac.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')

if(aguia):
    dirname = '${PWD##*/}'

    aguia_machine = 'aguia4'
    # aguia_machine = 'aguia3'

    if(aguia_machine == 'aguia4'):
        partition = 'SP2'
        main_folders = '/temporario2/8672526'

    if(aguia_machine == 'aguia3'):
        partition = 'SP3'
        main_folders =  '/scratch/8672526'

    run_aguia = f'''
            #!/usr/bin/bash

            #SBATCH --partition={partition}
            #SBATCH --ntasks=1
            #SBATCH --nodes=1
            #SBATCH --cpus-per-task={str(int(ncores))}
            #SBATCH --time 192:00:00 #16horas/"2-" para 2 dias com max 8 dias
            #SBATCH --job-name mandyoc-jpms
            #SBATCH --output slurm_%j.log #ou FD.out/ %j pega o id do job
            #SBATCH --mail-type=BEGIN,FAIL,END
            #SBATCH --mail-user=joao.macedo.silva@usp.br

            export PETSC_DIR={main_folders}/opt/petsc
            export PETSC_ARCH=arch-label-optimized
            MANDYOC={main_folders}/opt/mandyoc/bin/mandyoc
            MANDYOC_OPTIONS={mandyoc_options}

            $PETSC_DIR/$PETSC_ARCH/bin/mpiexec -n {str(int(ncores))} $MANDYOC $MANDYOC_OPTIONS

            DIRNAME={dirname}

            zip $DIRNAME.zip interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh
            zip -u $DIRNAME.zip bc_velocity_*.txt
            zip -u $DIRNAME.zip density_*.txt
            zip -u $DIRNAME.zip heat_*.txt
            zip -u $DIRNAME.zip pressure_*.txt
            zip -u $DIRNAME.zip sp_surface_global_*.txt
            zip -u $DIRNAME.zip strain_*.txt
            zip -u $DIRNAME.zip temperature_*.txt
            zip -u $DIRNAME.zip time_*.txt
            zip -u $DIRNAME.zip velocity_*.txt
            zip -u $DIRNAME.zip viscosity_*.txt
            zip -u $DIRNAME.zip scale_bcv.txt
            zip -u $DIRNAME.zip step*.txt
            zip -u $DIRNAME.zip *.log

            #rm *.log
            rm vel_bc*
            rm velz*
            rm bc_velocity*
            rm velocity*
            rm step*
            rm temperature*
            rm density*
            rm viscosity*
            rm heat*
            rm strain_*
            rm time*
            rm pressure_*
            rm sp_surface_global*
            rm scale_bcv.txt
        '''
    with open('run_aguia.sh', 'w') as f:
        for line in run_aguia.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')

if(hypatia):

    dirname = '${PWD##*/}'
    current_dir = '${PWD}'
    # main_folders = '/scratch/jpmacedo'
    main_folders = '/home/jpmacedo'
    run_hypatia = f'''
    #!/usr/bin/env bash
    #SBATCH --mail-type=BEGIN,END,FAIL         			# Mail events (NONE, BEGIN, END, FAIL, ALL)
    #SBATCH --mail-user=joao.macedo.silva@usp.br		# Where to send mail
    #SBATCH --ntasks={str(int(ncores))}
    #SBATCH --nodes={str(int(nodes))}
    #SBATCH --cpus-per-task=1
    #SBATCH --time 72:00:00 # 16 horas; poderia ser “2-” para 2 dias; máximo “8-”
    #SBATCH --job-name mandyoc-jpms
    #SBATCH --output slurm_%j.log
    #SBATCH --error=log_error_%j.log
    #SBATCH --no-requeue

    module purge
    module load gcc/13.2.0-gcc-8.5.0-tnbqzki
    module load openmpi/5.0.3-gcc-8.5.0-no4tqjk
    module load cmake/3.27.9-gcc-8.5.0-33534nt

    #Setup of Mandyoc variables:
    PETSC_DIR='{main_folders}/opt/petsc'
    PETSC_ARCH='optimized-v3.24.1-mpich'

    MANDYOC='{main_folders}/opt/mandyoc/bin/mandyoc'
    MANDYOC_OPTIONS='{mandyoc_options}'

    #run mandyoc
    mpirun -n ${{SLURM_NTASKS}} --map-by :OVERSUBSCRIBE ${{MANDYOC}} ${{MANDYOC_OPTIONS}}

    conda activate mpy
    #Creating directories for the output files
    bash mv /home/jpmacedo/opt/mv-updated.sh

    #Creating netdf files
    julia -t {str(int(ncores))} /home/jpmacedo/opt/convertNETCDF_v2.jl {current_dir}
    julia -t {str(int(ncores))} /home/jpmacedo/opt/LithoNETCDF_v2.jl {current_dir}

    python /home/jpmacedo/opt/track_particles_v3.py {current_dir} 0
    zip {dirname}.zip *.nc

    #run of auxiliary scripts to zip and clean the folder
    # bash zipper.sh
    # bash clean.sh
    '''
    with open('run_hypatia.sh', 'w') as f:
        for line in run_hypatia.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')


zipper = f'''
        #!/usr/bin/env bash
        DIRNAME={dirname}

        # Primeiro zipa os arquivos fixos
        zip "$DIRNAME.zip" interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh

        # Lista de padrões
        patterns=(
                "bc_velocity_*.txt"
                "density"
                "heat"
                "pressure"
                "surface"
                "strain"
                "temperature"
                "time"
                "velocity"
                "viscosity"
                "scale_bcv.txt"
                "steps"
                "lithos"
                "Phi"
                "dPhi"
                "X_depletion"
                "*.bin*.txt"
                "bc*-1.txt"
                )

        # Faz um loop e usa find para evitar o erro "argument list too long"
        for pat in "${{patterns[@]}}"; do
            find . -maxdepth 1 -name "$pat" -exec zip -u "$DIRNAME.zip" {{}} +
        done
    '''
with open('zipper.sh', 'w') as f:
    for line in zipper.split('\n'):
        line = line.strip()
        if len(line):
            f.write(' '.join(line.split()) + '\n')

clean = f'''
        #!/usr/bin/env bash

        # Lista de padrões
        patterns=(
                "bc_velocity_*.txt"
                "density"
                "heat"
                "pressure"
                "surface"
                "strain"
                "temperature"
                "time"
                "velocity"
                "viscosity"
                "scale_bcv.txt"
                "steps"
                "lithos"
                "Phi"
                "dPhi"
                "X_depletion"
                "*.bin*.txt"
                "bc*-1.txt"
                )

        # Para cada padrão, procurar e remover com segurança
        for pat in "${{patterns[@]}}"; do
            find . -maxdepth 1 -name "$pat" -exec rm -f {{}} +
        done
    '''
with open('clean.sh', 'w') as f:
    for line in clean.split('\n'):
        line = line.strip()
        if len(line):
            f.write(' '.join(line.split()) + '\n')

#zip input files
filename = 'inputs_'+path[-1]+'.zip'
files_list = ' infos*.txt interfaces.txt param.txt input*_0.txt run*.sh vel*.txt scale_bcv.txt *.png precipitation.txt climate.txt zipper.sh clean.sh ' + os.path.basename(svg_file)
os.system('zip '+filename+files_list)
