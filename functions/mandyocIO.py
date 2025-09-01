"""
Libraries to handle input and output from Mandyoc code.
"""
import glob
import os
import gc
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.transforms import Bbox
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.mlab as mlab
import matplotlib.colors


SEG = 365. * 24. * 60. * 60.

PARAMETERS = {
    "compositional_factor": "C",
    "density": "rho",
    "radiogenic_heat": "H",
    "pre-exponential_scale_factor": "A",
    "power_law_exponent": "n",
    "activation_energy": "Q",
    "activation_volume": "V"
}

TEMPERATURE_HEADER = "T1\nT2\nT3\nT4"

OUTPUTS = {
    "density": "density",
    "depletion_factor":"X_depletion",
    "incremental_melt":"dPhi",
    "melt":"Phi",
    # "X_depletion":"X_depletion",
    # "dPhi":"dPhi",
    # "Phi":"Phi",
    "radiogenic_heat": "heat",
    "pressure": "pressure",
    "strain": "strain",
    "strain_rate": "strain_rate",
    "temperature": "temperature",
    "viscosity": "viscosity",
    "surface": "surface",
    "velocity": "velocity",
}

OUTPUT_TIME = "time_"

PARAMETERS_FNAME = "param.txt"

# Define which datasets are scalars measured on the nodes of the grid, e.g.
# surface and velocity are not scalars.
SCALARS = tuple(OUTPUTS.keys())[:10]

def make_coordinates(region, shape):
    """
    Create grid coordinates for 2D and 3D models

    Parameters
    ----------
    region : tuple or list
        List containing the boundaries of the region of the grid. If the grid 
        is 2D, the boundaries should be passed in the following order:
        ``x_min``, ``x_max``,``z_min``, ``z_max``.
        If the grid is 3D, the boundaries should be passed in the following 
        order:
        ``x_min``, ``x_max``, ``y_min``, ``y_max``, ``z_min``, ``z_max``.
    shape : tuple
        Total number of grid nodes along each direction.
        If the grid is 2D, the tuple must be: ``n_x``, ``n_z``.
        If the grid is 3D, the tuple must be: ``n_x``, ``n_y``, ``n_z``.

    Returns
    -------
    coordinates : :class:`xarray.DataArrayCoordinates`
        Coordinates located on a regular grid.
    """
    # Sanity checks
    _check_region(region)
    _check_shape(region, shape)
    # Build coordinates according to
    if len(shape) == 2:
        nx, nz = shape[:]
        x_min, x_max, z_min, z_max = region[:]
        x = np.linspace(x_min, x_max, nx)
        z = np.linspace(z_min, z_max, nz)
        dims = ("x", "z")
        coords = {"x": x, "z": z}
    else:
        nx, ny, nz = shape[:]
        x_min, x_max, y_min, y_max, z_min, z_max = region[:]
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)
        dims = ("x", "y", "z")
        coords = {"x": x, "y": y, "z": z}
    da = xr.DataArray(np.zeros(shape), coords=coords, dims=dims)
    return da.coords

def make_interface(coordinates, values=[0.0], direction='x'):
    """
    Create an array to represent a 2D or 3D interface.
    
    If a single values is given, creates a horizontal interface with that 
    value as depth. If a list of points is given, creates the interface by 
    linear iterpolation.

    Parameters
    ----------
    coordinates : :class:`xarray.DataArrayCoordinates`
        Coordinates located on a regular grid that will be used to create the 
        interface. Must be in meters and can be either 2D or 3D. If they are 
        in 2D, the interface will be a curve, and if the coordinates are 3D, 
        the interface will be a surface.
    values : float (optional), None or list
        Value that will fill the initialized array. If None, the array will be
        filled with ``numpy.nan``s. If a list of vertices is given, it 
        interpolates them.
    direction : string (optional)
        Direction of the subduction. If working in 3D it can be either *"x"* or *"y"*.
        When working in 2D, it must be *"x"*.

    Returns
    -------
    arr : :class:`xarray.DataArray`
        Array containing the interface.
    """
    # Get shape of coordinates
    shape = _get_shape(coordinates)
    
    cond_1 = not isinstance(values, list)
    cond_2 = (isinstance(values, list)) and (len(values)==1)
    cond_3 = (isinstance(values, list)) and (len(values)>1)
    
    if (cond_1) or (cond_2):
        islist = False
    elif (cond_3):
        islist = True
    else:
        raise ValueError("Unrecognized values for making interface.")
        
    if (islist==False):
        if values is None:
            values = np.nan
        # Remove the shape on z
        shape = shape[:-1]
        arr = xr.DataArray(
            values * np.ones(shape),
            coords = [coordinates[i] for i in coordinates if i != "z"],
        )
    elif (islist==True):
        h_min, h_max = coordinates[direction].min(), coordinates[direction].max()
        values = np.array(values)
        # _check_boundary_vertices(values, h_min, h_max)
        interface = np.interp(coordinates[direction], values[:, 0], values[:, 1])
        arr = xr.DataArray(interface, coords=[coordinates[direction]], dims=direction)
        if len(coordinates.dims) == 3:
            if direction == "x":
                missing_dim = "y"
            elif direction == "y":
                missing_dim == "x"
            arr = arr.expand_dims({missing_dim: coordinates[missing_dim].size})
            arr.coords[missing_dim] = coordinates[missing_dim]
            arr = arr.transpose("x", "y")
    return arr

def merge_interfaces(interfaces):
    """
    Merge a dictionary of interfaces into a single xarray.Dataset

    Parameters
    ----------
    interfaces : dict
        Dictionary containing a collection of interfaces.

    Returns
    -------
    ds : :class:`xarray.Dataset`
        Dataset containing the interfaces.
    """
    ds = None
    for name, interface in interfaces.items():
        if ds:
            ds[name] = interface
        else:
            ds = interfaces[name].to_dataset(name=name)
    return ds

def save_interfaces(interfaces, parameters, path, strain_softening, fname='interfaces.txt'):
    """
    Save the interfaces and the rheological parameters as an ASCII file.

    Parameters
    ----------
    interfaces : :class:`xarray.Dataset`
        Dataset with the interfaces depth.
    parameters : dict
        Dictionary with the parameters values for each lithological unit.
        The necessary parameters are:
            - ``compositional factor``,
            - ``density``,
            - ``radiogenic heat``,
            - ``pre-exponential scale factor``,
            - ``power law exponent``,
            - ``activation energy``,
            - ``activation volume``,
            - ``weakening seed``
            - ``cohesion max``
            - ``cohesion min``
            - ``friction angle min``
            - ``friction angle max``
    path : str
        Path to save the file.
    fname : str (optional)
        Name to save the interface file. Default ``interface.txt``
    """
    # Check if givens parameters are consistent
    _check_necessary_parameters(parameters, interfaces, strain_softening)

    # Generate the header with the layers parameters
    header = []
    for parameter in parameters:
        header.append(
            PARAMETERS[parameter]
            + " "
            + " ".join(list(str(i) for i in parameters[parameter]))
        )
    header = "\n".join(header)
    dimension = len(interfaces.dims)
    expected_dims = "x"
    interfaces = interfaces.transpose(*expected_dims)
    # Stack and ravel the interfaces from the dataset
    # We will use order "F" on numpy.ravel in order to make the first index to change
    # faster than the rest
    stacked_interfaces = np.hstack(
        list(interfaces[i].values.ravel(order="F")[:, np.newaxis] for i in interfaces)
    )
    # Save the interface and the layers parameters
    np.savetxt(
        os.path.join(path, fname),
        stacked_interfaces,
        fmt="%f",
        header=header,
        comments="",
    )
    
def make_grid(coordinates, value=0):
    """
    Create an empty grid for a set of coordinates.

    Parameters
    ----------
    coordinates : :class:`xarray.DataArrayCoordinates`
        Coordinates located on a regular grid where the temperature distribution will be
        created. Must be in meters and can be either 2D or 3D.
    value : float (optional) or None
        Value that will fill the initialized array. If None, the array will be filled
        with ``numpy.nan``s. Default to 0.
    """
    # Get shape of coordinates
    shape = _get_shape(coordinates)
    if value is None:
        value = np.nan
    return xr.DataArray(value * np.ones(shape), coords=coordinates)

def save_temperature(temperatures, path, fname="input_temperature_0.txt"):
    """
    Save the temperature grid as an ASCII file.

    The temperatures grid values are saved on a single column, following each axis
    in increasing order, with the ``x`` indexes changing faster that the ``z``.

    Parameters
    ----------
    temperatures : :class:`xarray.DataArray`
        Array containing a temperature distribution. Can be either 2D or 3D.
    path : str
        Path to save the temperature file.
    fname : str (optional)
       Filename of the output ASCII file. Deault to ``input_temperature_0.txt``.
    """
    expected_dims = ("x", "z")
    # Check if temperature dims are the right ones
    invalid_dims = [dim for dim in temperatures.dims if dim not in expected_dims]
    if invalid_dims:
        raise ValueError(
            "Invalid temperature dimensions '{}': ".format(invalid_dims)
            + "must be '{}' for a 2D temperature grid.".format(expected_dims)
        )
    # Change order of temperature dimensions to ("x", "z") to ensure
    # right order of elements when the array is ravelled
    temperatures = temperatures.transpose(*expected_dims)
    # Ravel and save temperatures
    # We will use order "F" on numpy.ravel in order to make the first index to change
    # faster than the rest
    # Will add a custom header required by MANDYOC
    np.savetxt(
        os.path.join(path, fname), temperatures.values.ravel(order="F"), header=TEMPERATURE_HEADER
    )
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
        
def read_mandyoc_output(model_path, parameters_file=PARAMETERS_FNAME, datasets=tuple(OUTPUTS.keys()), skip=1, steps_slice=None, save_big_dataset=False):
    """
    Read the files  generate by Mandyoc code
    Parameters
    ----------
    model_path : str
        Path to the folder where the Mandyoc files are located.
    parameters_file : str (optional)
        Name of the parameters file. It must be located inside the ``path``
        directory.
        Default to ``"param.txt"``.
    datasets : tuple (optional)
        Tuple containing the datasets that wants to be loaded.
        The available datasets are:
            - ``temperature``
            - ``density"``
            - ``radiogenic_heat``
            - ``strain``
            - ``strain_rate``
            - ``pressure``
            - ``viscosity``
            - ``velocity``
            - ``surface``
        By default, every dataset will be read.
    skip: int
        Reads files every <skip> value to save mamemory.
    steps_slice : tuple
        Slice of steps to generate the step array. If it is None, it is taken
        from the folder where the Mandyoc files are located.
    save_big_dataset : bool
        Save all datasets in a single dataset. Recomended to small models
    filetype : str
        Files format to be read. Default to ``"ascii"``.
    Returns
    -------
    dataset :  :class:`xarray.Dataset`
        Dataset containing data generated by Mandyoc code.
    """
    # Read parameters
    parameters = _read_parameters(os.path.join(model_path, parameters_file))
    # Build coordinates
    shape = parameters["shape"]
    aux_coords = make_coordinates(region=parameters["region"], shape=shape)
    coordinates = np.array(aux_coords["x"]), np.array(aux_coords["z"])
    # Get array of times and steps
    steps, times = _read_times(model_path, parameters["print_step"], parameters["step_max"], steps_slice)
    steps = steps[::skip]
    times = times[::skip]
    end = np.size(times)
    # Create the coordinates dictionary containing the coordinates of the nodes
    # and the time and step arrays. Then create data_vars dictionary containing
    # the desired scalars datasets.
    coords = {"time": times, "step": ("time", steps)}
    dims = ("time", "x", "z")
    profile_dims = ("time", "x")
    coords["x"], coords["z"] = coordinates[:]

    print(f"Starting...")
    datasets_aux = []
    for scalar in SCALARS:
        if scalar in datasets:
            datasets_aux.append(scalar)
            scalars = _read_scalars(model_path, shape, steps, quantity=scalar)
            data_aux = {scalar: (dims, scalars)}
            xr.Dataset(data_aux, coords=coords, attrs=parameters).to_netcdf(f"{model_path}/_output_{scalar}.nc")
            print(f"{scalar.capitalize()} files saved.")
            del scalars
            del data_aux
            gc.collect()
    
    
    # Read surface if needed
    if "surface" in datasets:
        datasets_aux.append("surface")
        surface = _read_surface(model_path, shape[0], steps)
        data_aux = {"surface": (profile_dims, surface)}
        xr.Dataset(data_aux, coords=coords, attrs=parameters).to_netcdf(f"{model_path}/_output_surface.nc")
        print(f"Surface files saved.")
        del surface
        del data_aux
        gc.collect()

    # Read velocity if needed
    if "velocity" in datasets:
        datasets_aux.append("velocity")
        velocities = _read_velocity(model_path, shape, steps)
        data_aux = {}
        data_aux["velocity_x"] = (dims, velocities[0])
        data_aux["velocity_z"] = (dims, velocities[1])
        xr.Dataset(data_aux, coords=coords, attrs=parameters).to_netcdf(f"{model_path}/_output_velocity.nc")
        print(f"Velocity files saved.")
        del velocities
        del data_aux
        gc.collect()

    print(f"All files read and saved.")
    # return xr.Dataset(data_vars, coords=coords, attrs=parameters)  
    
#     empty_dataset = True
#     for item in datasets_aux:
#         dataset_aux = xr.open_dataset(f"{model_path}/_output_{item}.nc")
#         if (empty_dataset == True):
#             dataset = dataset_aux
#             empty_dataset = False
#         else:
#             dataset = dataset.merge(dataset_aux)
#         del dataset_aux
#     gc.collect()

#     dataset.to_netcdf(f"{model_path}/data.nc", format="NETCDF3_64BIT")
    # del dataset
    # gc.collect()
    
    return datasets_aux

def read_datasets(model_path, datasets, save_big_dataset=False):
    empty_dataset = True
    for item in datasets:
        dataset_aux = xr.open_dataset(f"{model_path}/_output_{item}.nc")
        if (empty_dataset == True):
            dataset = dataset_aux
            empty_dataset = False
        else:
            dataset = dataset.merge(dataset_aux)
        del dataset_aux
    gc.collect()

    if (save_big_dataset):
        print(f'Saving dataset with all Mandyoc data')
        dataset.to_netcdf(f"{model_path}/data.nc", format="NETCDF3_64BIT")
        print(f"Big dataset file saved.")
        # del dataset
        # gc.collect()
    
    return dataset

def diffuse_field(field, cond_air, kappa, dx, dz, t_max=1.0E6, fac=100):
    """
    Calculates the diffusion of a 2D field using finite difference.
    ----------
    field : numpy.ndarray
        2D field that will be diffused.
    conda_air : numpy.ndarray
        2D box where value will be constant.
    kappa : float
        Thermal diffusivity coefficient.
    dx: float
        Spacing in the x (horizontal) direction.
    dz: float
        Spacing in the z (vertical) direction.
    t_max: float
        Maximum diffusion time in years.
    fac: int
        Number of time steps to diffuse the field.
    Returns
    -------
    field :  numpy.ndarray
        2D array containing the diffused field.
    """
    dx_aux = np.min([np.abs(dx),np.abs(dz)])
    dt = np.min([dx_aux**2./(2.*kappa), t_max/fac])
    
    CTx = kappa * dt * SEG / (dx**2)
    CTz = kappa * dt * SEG / (dz**2)
    
    t = 0.0
    while (t<=t_max):
        auxX = field[2:,1:-1] + field[:-2,1:-1] - 2 * field[1:-1,1:-1]
        auxZ = field[1:-1,2:] + field[1:-1,:-2] - 2 * field[1:-1,1:-1]
        field[1:-1,1:-1] = field[1:-1,1:-1] + (CTx * auxX) + (CTz * auxZ)
        # boundary conditions
        field[:,cond_air] = 0.0
        field[0,:] = field[1,:]
        field[-1,:] = field[-2,:]
        # time increment
        t += dt
    return field

def build_parameter_file(**kwargs):
    """
    Creates a parameters dicitionary containing all the parameters that will be
    necessary fot the simulation.

    Parameters
    ----------
    **kwargs : dict
        Arguments used in the paramenter file.

    Returns
    -------
    params : dict
        Complete parameter file dictionary.
    """
    defaults = {
        'nx' : None,
        'nz' : None,
        'lx' : None,
        'lz' : None, 

        'aux00': '# Simulation options',
        'solver' : 'direct',
        'denok' : 1.0e-10,
        'rtol' : 1.0e-7,
        'RK4' : 'Euler',
        'Xi_min' : 1.0e-7,
        'random_initial_strain' : 0.0,
        'pressure_const' : -1.0,
        'initial_dynamic_range' : True,
        'periodic_boundary' : False,
        'high_kappa_in_asthenosphere' : False,
        'basal_heat' : -1.0,

        'aux01': '# Particles options',
        'particles_per_element' : 81,
        'particles_per_element_x' : 0,
        'particles_per_element_z' : 0,
        'particles_perturb_factor' : 0.7,

        'aux02': '# Surface processes',
        'sp_surface_tracking' : False,
        'sea_level' : 0.0,
        'sp_surface_processes' : False,
        'sp_dt' : 0,
        'a2l' : True,
        'sp_mode' : 1,
        'free_surface_stab' : True,
        'theta_FSSA' : 0.5,
        'sticky_blanket_air' : False,
        'precipitation_profile_from_ascii' : False,
        'climate_change_from_ascii' : False,

        'aux03': '# Time constrains',
        'step_max' : 7000,
        'time_max' : 10.0e6,
        'dt_max' : 10.0e3,
        'step_print' : 10,
        'sub_division_time_step' : 1.0,
        'initial_print_step' : 0,
        'initial_print_max_time' : 1.0e6,

        'aux04': '# Viscosity',
        'viscosity_reference' : None,
        'viscosity_max' : None,
        'viscosity_min' : None,
        'viscosity_per_element' : 'constant',
        'viscosity_mean_method' : 'arithmetic',
        'viscosity_dependence' : 'pressure',

        'aux05': '# External ASCII inputs/outputs',
        'interfaces_from_ascii' : True,
        'n_interfaces' : None,
        'temperature_from_ascii' : True,
        'velocity_from_ascii' : False,
        'variable_bcv' : False,
        'multi_velocity' : False,
        'binary_output' : False,
        'print_step_files' : True,

        'aux06': '# Physical parameters',
        'temperature_difference' : None,
        'thermal_expansion_coefficient' : None,
        'thermal_diffusivity_coefficient' : None,
        'gravity_acceleration' : None,
        'density_mantle' : 3300.,
        'heat_capacity' : None,
        'adiabatic_component' : None,
        'radiogenic_component' : None,

        'aux07': '# Strain softening',
        'non_linear_method' : 'on',
        'plasticity' : 'on',
        'weakening_min' : 0.05,
        'weakening_max' : 1.05,

        'aux08': '# Velocity boundary conditions',
        'top_normal_velocity' : 'fixed',
        'top_tangential_velocity' : 'free',
        'bot_normal_velocity' : 'fixed',
        'bot_tangential_velocity' : 'free',
        'left_normal_velocity' : 'fixed',
        'left_tangential_velocity' : 'free',
        'right_normal_velocity' : 'fixed ',
        'right_tangential_velocity' : 'free',

        'aux09': '# Temperature boundary conditions',
        'top_temperature' : 'fixed',
        'bot_temperature' : 'fixed',
        'left_temperature' : 'fixed',
        'right_temperature' : 'free',
        'rheology_model' : 19,
        'T_initial' : 0,
    }

    params = {}
    for key, value in defaults.items():
        param = kwargs.get(key, value)
        if param is None:
            raise ValueError(f"The parameter '{key}' is mandatory.")
        params[key] = str(param)
    return params

def save_parameter_file(params, run_dir):
    """
    Saves the parameter dictionary into a file called param.txt

    Parameters
    ----------

    params : dict
        Dictionary containing the parameters of the param.txt file.
    """
    # Create the parameter file
    with open(os.path.join(run_dir,"param.txt"), "w") as f:
        for key, value in params.items():
            if key[:3] == "aux":
                f.write(f"\n{value}\n")
            else:
                f.write('{:<32} = {}\n'.format(key, value))

def _read_scalars(path, shape, steps, quantity):
    """
    Read Mandyoc scalar data
    Read ``temperature``, ``density``, ``radiogenic_heat``, ``viscosity``,
    ``strain``, ``strain_rate`` and ``pressure``.
    Parameters
    ----------
    path : str
        Path to the folder where the Mandyoc files are located.
    shape: tuple
        Shape of the expected grid.
    steps : array
        Array containing the saved steps.
    quantity : str
        Type of scalar data to be read.
    Returns
    -------
    data: np.array
        Array containing the Mandyoc scalar data.
    """
    print(f"Reading {quantity} files...", end=" ")
    data = []
    for step in steps:
        filename = "{}_{}".format(OUTPUTS[quantity], step)
        data_step = np.loadtxt(
            os.path.join(path, filename + ".txt"),
            unpack=True,
            comments="P",
            skiprows=2,
        )
        # Convert very small numbers to zero
        data_step[np.abs(data_step) < 1.0e-200] = 0
        # Reshape data_step
        data_step = data_step.reshape(shape, order="F")
        # Append data_step to data
        data.append(data_step)
    data = np.array(data)
    print(f"{quantity.capitalize()} files read.", end=" ")
    return data

def _read_velocity(path, shape, steps):
    """
    Read velocity data generated by Mandyoc code
    Parameters
    ----------
    path : str
        Path to the folder where the Mandyoc output files are located.
    shape: tuple
        Shape of the expected grid.
    steps : array
        Array containing the saved steps.
    Returns
    -------
    data: tuple of arrays
        Tuple containing the components of the velocity vector.
    """
    print(f"Reading velocity files...", end=" ")
    # Determine the dimension of the velocity data
    dimension = len(shape)
    velocity_x, velocity_z = [], []
    for step in steps:
        filename = "{}_{}".format(OUTPUTS["velocity"], step)
        velocity = np.loadtxt(
            os.path.join(path, filename + ".txt"), comments="P", skiprows=2
        )
        # Convert very small numbers to zero
        velocity[np.abs(velocity) < 1.0e-200] = 0
        # Separate velocity into their three components
        velocity_x.append(velocity[0::dimension].reshape(shape, order="F"))
        velocity_z.append(velocity[1::dimension].reshape(shape, order="F"))
    # Transform the velocity_* lists to arrays
    velocity_x = np.array(velocity_x)
    velocity_z = np.array(velocity_z)
    print(f"Velocity files read.", end=" ")
    return (velocity_x, velocity_z)

def _read_surface(path, size, steps):
    """
    Read surface data generated by the Mandyoc code
    Parameters
    ----------
    path : str
        Path to the folder where the Mandyoc output files are located.
    size : int
        Size of the surface profile.
    steps : array
        Array containing the saved steps.
    Returns
    -------
    data : np.array
        Array containing the Mandyoc profile data.
    """
    print(f"Reading surface files...", end=" ")
    data = []
    for step in steps:
        filename = "sp_surface_global_{}".format(step)
        data_step = np.loadtxt(
            os.path.join(path, filename + ".txt"),
            unpack=True,
            comments="P",
            skiprows=2,
        )
        # Convert very small numbers to zero
        data_step[np.abs(data_step) < 1.0e-200] = 0
        # Reshape data_step
        # data_step = data_step.reshape(shape, order="F")
        # Append data_step to data
        data.append(data_step)
    data = np.array(data)
    print(f"Surface files read.", end=" ")
    return data
            
def _read_parameters(parameters_file):
    """
    Read parameters file
    .. warning :
        The parameters file contains the length of the region along each axe.
        While creating the region, we are assuming that the z axe points upwards
        and therefore all values beneath the surface are negative, and the x
        and y axes are all positive within the region.
    Parameters
    ----------
    parameters_file : str
        Path to the location of the parameters file.
    Returns
    -------
    parameters : dict
        Dictionary containing the parameters of Mandyoc files.
    """
    parameters = {}
    with open(parameters_file, "r") as params_file:
        for line in params_file:
            # Skip blank lines
            if not line.strip():
                continue
            if line[0] == "#":
                continue
            # Remove comments lines
            line = line.split("#")[0].split()
            var_name, var_value = line[0], line[2]
            parameters[var_name.strip()] = var_value.strip()
        # Add shape
        parameters["shape"] = (int(parameters["nx"]), int(parameters["nz"]))
        # Add dimension
        parameters["dimension"] = len(parameters["shape"])
        # Add region
        parameters["region"] = (
            0,
            float(parameters["lx"]),
            -float(parameters["lz"]),
            0,
        )
        parameters["step_max"] = int(parameters["step_max"])
        parameters["time_max"] = float(parameters["time_max"])
        parameters["print_step"] = int(parameters["step_print"])
        # Add units
        parameters["coords_units"] = "m"
        parameters["times_units"] = "Ma"
        parameters["temperature_units"] = "C"
        parameters["density_units"] = "kg/m^3"
        parameters["heat_units"] = "W/m^3"
        parameters["viscosity_units"] = "Pa s"
        parameters["strain_rate_units"] = "s^(-1)"
        parameters["pressure_units"] = "Pa"
    return parameters

def _read_times(path, print_step, max_steps, steps_slice):
    """
    Read the time files generated by Mandyoc code
    Parameters
    ----------
    path : str
        Path to the folder where the Mandyoc files are located.
    print_step : int
        Only steps multiple of ``print_step`` are saved by Mandyoc.
    max_steps : int
        Maximum number of steps. Mandyoc could break computation before the
        ``max_steps`` are run if the maximum time is reached. This quantity only
        bounds the number of time files.
    steps_slice : tuple
        Slice of steps (min_steps_slice, max_steps_slice). If it is None,
        min_step_slice = 0 and max_steps_slice = max_steps.
    Returns
    -------
    steps : numpy array
        Array containing the saved steps.
    times : numpy array
        Array containing the time of each step in Ma.
    """
    steps, times = [], []
    # Define the mininun and maximun step
    if steps_slice is not None:
        min_steps_slice, max_steps_slice = steps_slice[:]
    else:
        min_steps_slice, max_steps_slice = 0, max_steps
    for step in range(min_steps_slice, max_steps_slice + print_step, print_step):
        filename = os.path.join(path, "{}{}.txt".format(OUTPUT_TIME, step))
        if not os.path.isfile(filename):
            break
        time = np.loadtxt(filename, unpack=True, delimiter=":", usecols=(1))
        if time.shape == ():
            times.append(time)
        else:
            time = time[0]
            times.append(time)
        steps.append(step)

    # Transforms lists to arrays
    times = 1e-6 * np.array(times)  # convert time units into Ma
    steps = np.array(steps, dtype=int)
    return steps, times

def _check_necessary_parameters(parameters, interfaces, strain_softening):
    """
    Check if there all parameters are given (not checking number).
    """
    if (strain_softening):
        PARAMETERS['weakening_seed'] = 'weakening_seed'
        PARAMETERS['cohesion_min'] = 'cohesion_min'
        PARAMETERS['cohesion_max'] = 'cohesion_max'
        PARAMETERS['friction_angle_min'] = 'friction_angle_min'
        PARAMETERS['friction_angle_max'] = 'friction_angle_max'
    else:
        PARAMETERS.pop('weakening_seed', None)
        PARAMETERS.pop('cohesion_min', None)
        PARAMETERS.pop('cohesion_max', None)
        PARAMETERS.pop('friction_angle_min', None)
        PARAMETERS.pop('friction_angle_max', None)

    for parameter in PARAMETERS:
        if parameter not in parameters:
            raise ValueError(
                "Parameter '{}' missing. ".format(parameter)
                + "All the following parameters must be included:"
                + "\n    "
                + "\n    ".join([str(i) for i in PARAMETERS.keys()])
            )
            
    """
    Check if the number of parameters is correct for each lithological unit.
    """
    sizes = list(len(i) for i in list(parameters.values()))
    if not np.allclose(sizes[0], sizes):
        raise ValueError(
            "Missing parameters for the lithological units. "
            + "Check if each lithological unit has all the parameters."
        )  
    """
    Check if the number of parameters is equal to the number of lithological units.
    """
    size = len(list(parameters.values())[0])
    if not np.allclose(size, len(interfaces) + 1):
        raise ValueError(
            "Invalid number of parameters ({}) for given number of lithological units ({}). ".format(
                size, len(interfaces)
            )
            + "The number of lithological units must be the number of interfaces plus one."
        )
    """
    Check if the interfaces do not cross each other.
    """
    inames = tuple(i for i in interfaces)
    for i in range(len(inames) - 1):
        if not (interfaces[inames[i + 1]] >= interfaces[inames[i]]).values.all():
            raise ValueError(
                "Interfaces are in the wrong order or crossing each other. "
                + "Check interfaces ({}) and ({}). ".format(
                    inames[i], inames[i + 1]
                )
            )

def _check_region(region):
    """
    Sanity checks for region
    """
    if len(region) == 4:
        x_min, x_max, z_min, z_max = region
    elif len(region) == 6:
        x_min, x_max, y_min, y_max, z_min, z_max = region
        if y_min >= y_max:
            raise ValueError(
                "Invalid region domain '{}' (x_min, x_max, z_min, z_max). ".format(region)
                + "Must have y_min =< y_max. "
            )
    else:
        raise ValueError(
            "Invalid number of region domain limits '{}'. ".format(region)
            + "Only 4 or 6 values allowed for 2D and 3D dimensions, respectively."
        )
    if x_min >= x_max:
        raise ValueError(
            "Invalid region '{}' (x_min, x_max, z_min, z_max). ".format(region)
            + "Must have x_min =< x_max. "
        )
    if z_min >= z_max:
        raise ValueError(
            "Invalid region '{}' (x_min, x_max, z_min, z_max). ".format(region)
            + "Must have z_min =< z_max. "
        )

def _check_shape(region, shape):
    """
    Check shape lenght and if the region matches it
    """
    if len(shape) not in (2, 3):
        raise ValueError(
            "Invalid shape '{}'. ".format(shape) + "Shape must have 2 or 3 elements."
        )
    if len(shape) != len(region) // 2:
        raise ValueError(
            "Invalid region '{}' for shape '{}'. ".format(region, shape)
            + "Region must have twice the elements of shape."
        )
        
def _get_shape(coordinates):
    """
    Return the shape of ``coordinates``.

    Parameters
    ----------
    coordinates : :class:`xarray.DataArrayCoordinates`
        Coordinates located on a regular grid.

    Return
    ------
    shape : tuple
        Tuple containing the shape of the coordinates
    """
    return tuple(coordinates[i].size for i in coordinates.dims)

def _check_boundary_vertices(values, h_min, h_max):
    """
    Check if the boundary vertices match the boundary coordinates.
    """
    h = values[:, 0]
    if not np.allclose(h_min, h.min()) or not np.allclose(h_max, h.max()):
        raise ValueError(
            "Invalid vertices for creating the interfaces: {}. ".format(values)
            + "Remember to include boundary nodes that matches the coordinates "
            + "boundaries '{}.'".format((h_min, h_max))
        )

def read_particle_path(path, position, unit_number=np.nan, ncores=np.nan):
    """
    Follow a particle through time.
    
    Parameters
    ----------
    path : str
        Path to read the data.
    position : tuple
        (x, z) position in meters of the particle to be followed. 
        The closest particle will be used.
    unit_number : int
        Lithology number of the layer.
    ncores: int
        Number of cores used during the simulation, necessary to read the files properly.
    
    Return
    ------
    particle_path : array (legth N)
        Position of the particle in each time step.
    """
    
    # check ncores
    if (np.isnan(ncores)):
        aux = glob.glob(f"{path}/step_0_*.txt")
        ncores = np.size(aux)
    
    # read first step
    first_x, first_z, first_ID, first_lithology, first_strain = _read_step(path, "step_0_", ncores)
    
    # order closest points to <position> at the first step
    pos_x, pos_z = position
    dist = np.sqrt(((first_x - pos_x)**2) + ((first_z - pos_z)**2))
    clst = np.argsort(dist)
    
    # read last step
    parameters = _read_parameters(os.path.join(path, PARAMETERS_FNAME))
    nsteps = np.size(glob.glob(f"{path}/time_*.txt"))
    last_step_number = int(parameters["print_step"]*(nsteps-1))
    last_x, last_z, last_ID, last_lithology, last_strain = _read_step(path, f"step_{last_step_number}_", ncores)
    
    # loop through closest poinst while the closest point is not in the last step
    print(f'Finding closest point to x: {pos_x} [m] and z: {pos_z} [m]...')
    cont = 0
    point_in_sim = False
    point_in_lit = False
    while (point_in_sim == False):
        clst_ID = first_ID[clst[cont]]
        # check if closest point is within the desired lithology
        if (np.isnan(unit_number)):
            point_in_lit = True # lithology number does not matter
        else:
            if int(first_lithology[clst[cont]]) == unit_number:
                point_in_lit = True
            else:
                point_in_lit = False # line not necessary (this is for sanity)
                
        # check if closest point is in the last step or find another closer one
        if (clst_ID in last_ID) and (point_in_lit == True):
            print(f'Found point with ID: {first_ID[clst[cont]]}, x: {first_x[clst[cont]]} [m], z: {first_z[clst[cont]]} [m]')
            point_in_sim = True
            closest_ID = clst_ID                
        else:
            print(f'Found point with ID: {first_ID[clst[cont]]}, x: {first_x[clst[cont]]}, z: {first_z[clst[cont]]}')
            print(f'Point DOES NOT persist through the simulation. Finding another one...')
            cont += 1

    # read all steps storing the point position
    print("Reading step files...", end=" ")
    x, z = [], []
    for i in range(0, last_step_number, parameters["print_step"]):
        current_x, current_z, current_ID, current_lithology, current_strain = _read_step(path, f"step_{i}_", ncores)
        arg = np.where(current_ID == closest_ID)
        x = np.append(x, current_x[arg])
        z = np.append(z, current_z[arg])
    print("Step files read.")
        
    return x, z, closest_ID

def _read_step(path, filename, ncores):
    """
    Read a step file.
    
    Parameters
    ----------
    path : str
        Path to read the data.
    filename : str
        Auxiliary file name.
    ncores : int
        Number of cores the simulation used.
        
    Return
    ------
    data_x : array (Length N)
        Array containing the position x of the particle.
    data_z : array (Length N)
        Array containing the position z of the particle.
    data_ID : array (Length N)
        Array containing the ID of the particle.
    data_lithology : array (Length N)
        Array containing the number of the particle lithology of the particle.
    data_strain : array (Length N)
        Array containing the strain of the particle.
    """
    data_x, data_z, data_ID, data_lithology, data_strain = [], [], [], [], []
    for i in range(ncores):
        try:
            aux_x, aux_z, aux_ID, aux_lithology, aux_strain = np.loadtxt(os.path.join(path, f"{filename}{str(i)}.txt"), unpack=True, comments="P")
        except:
            print('didnt read')
            continue
        data_x = np.append(data_x, aux_x)
        data_z = np.append(data_z, aux_z)
        data_ID = np.append(data_ID, aux_ID)
        data_lithology = np.append(data_lithology, aux_lithology)
        data_strain = np.append(data_strain, aux_strain)
    return np.asarray(data_x), np.asarray(data_z), np.asarray(data_ID), np.asarray(data_lithology), np.asarray(data_strain)


def _extract_interface(z, Z, Nx, field_datai, value_to_search):
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

    field_data: array_like (Nz, Nx)
        Mandyoc field from mandyoc (e.g.: density, temperature, etc).

    value_to_search: float
        Value of density to be searched in field_data array.

    Returns
    -------
    mapped_interface: array_like
        Interface extracted from Rhoi field
    '''

    mapped_interface = []

    for j in np.arange(Nx):

        interp_function = interp1d(z, field_datai[:,j]) #return a "function" of interpolation to apply in other array
        
        interface_inverted = interp_function(Z)[::-1] #inverted array: from top to bottom

        idx = np.argmax(interface_inverted > value_to_search) #first occurrence of rho in the inverted array
        idx_corrected = len(interface_inverted) - idx #correcting the index to the original array

        depth = Z[idx_corrected]

        mapped_interface = np.append(mapped_interface, depth)

    return mapped_interface

def _log_fmt(x, pos):
    return "{:.0f}".format(np.log10(x))

def change_dataset(properties, datasets):
    '''
    Create new_datasets based on the properties that will be plotted
    
    Parameters
    ----------
    properties: list of strings
        Properties to plot.

    datasets: list of strings
        List of saved properties.

    Returns
    -------
    new_datasets: list of strings
        New list of properties that will be read.
    '''
    
    new_datasets = []
    for prop in properties:
        if (prop in datasets) and (prop not in new_datasets):
            new_datasets.append(prop)
        if (prop == "lithology") and ("strain" not in new_datasets):
            new_datasets.append("strain")
        if (prop == "temperature_anomaly") and ("temperature" not in new_datasets):
            new_datasets.append("temperature")
        if(prop == 'melt'):
            new_datasets.append('melt')
        if(prop == 'incremental_melt'):
            new_datasets.append('incremental_melt')
        if (prop == "lithology" or prop == 'temperature_anomlay') and ("density" not in new_datasets):
            new_datasets.append("density")
            
    return new_datasets

def _calc_melt_dry(To,Po):

    P=np.asarray(Po)/1.0E6 # Pa -> MPa
    T=np.asarray(To)+273.15 #oC -> Kelvin

    Tsol = 1394 + 0.132899*P - 0.000005104*P**2
    cond = P>10000.0
    Tsol[cond] = 2212 + 0.030819*(P[cond] - 10000.0)

    Tliq = 2073 + 0.114*P

    X = P*0

    cond=(T>Tliq) #melt
    X[cond]=1.0

    cond=(T<Tliq)&(T>Tsol) #partial melt
    X[cond] = ((T[cond]-Tsol[cond])/(Tliq[cond]-Tsol[cond]))

    return(X)

def _calc_melt_wet(To,Po):
    P=np.asarray(Po)/1.0E6 # Pa -> MPa
    T=np.asarray(To)+273.15 #oC -> Kelvin

    Tsol = 1240 + 49800/(P + 323)
    cond = P>2400.0
    Tsol[cond] = 1266 - 0.0118*P[cond] + 0.0000035*P[cond]**2

    Tliq = 2073 + 0.114*P

    X = P*0

    cond=(T>Tliq)
    X[cond]=1.0

    cond=(T<Tliq)&(T>Tsol)
    X[cond] = ((T[cond]-Tsol[cond])/(Tliq[cond]-Tsol[cond]))

    return(X)

def measure_margins_width(dataset, xl_begin=600, xl_end=800, xr_begin=800, xr_end=1000):
    '''
    Measure the width of the rifted margins in a given region of the model.

    Parameters
    ----------

    fpath: str
        Path to the scenario directory.

    step: int
        Step of Mandyoc output files.

    Nx: int
        Number of points in x direction.

    Nz: int
        Number of points in z direction.

    Lx: float
        Length of the model in x direction.

    Lz: float
        Length of the model in z direction.

    xl_begin: float
        Initial x position to measure the left margin.

    xl_end: float
        Final x position to measure the left margin.

    xr_begin: float
        Initial x position to measure the right margin.

    xr_end: float
        Final x position to measure the right margin.

    Returns
    -------
    marginl_begin: float
        Initial position of the left margin.

    marginl_end: float
        Final position of the left margin.

    marginl_wdt: float
        Width of the left margin.

    marginr_begin: float
        Initial position of the right margin.

    marginr_end: float
        Final position of the right margin.

    marginr_wdt: float
        Width of the right margin.
    '''
    Rhoi = dataset.density.values.T

    Nx = int(dataset.nx)
    Nz = int(dataset.nz)
    Lx = float(dataset.lx)
    Lz = float(dataset.lz)

    x = np.linspace(0, Lx/1000.0, Nx)
    z = np.linspace(-Lz/1000.0, 0, Nz)
    Z = np.linspace(-Lz/1000.0, 0, 8001) #zi

    h_air = 40.0
    topography_interface = _extract_interface(z, Z, Nx, Rhoi, 200.) + h_air
    lower_interface = _extract_interface(z, Z, Nx, Rhoi, 2900.) + h_air
    
    crustal_thickness = np.abs(lower_interface - topography_interface)
    
    #left side
    cond_xl = (x >= xl_begin) & (x <= xl_end)

    # topol_region = topography_interface[cond_xl]
    # idx_max_topol = np.where(topol_region == np.max(topol_region))[0][0] #the first index of the maximum topography is more distal
    # marginl_begin = x[cond_xl][idx_max_topol]

    marginl_begin = xl_begin

    # left_margin_region = lower_interface[cond_xl]
    left_margin_region = crustal_thickness[cond_xl]

    # idx_min_lithol = np.where(lithol_region == np.max(lithol_region))[0][0]
    # idx_min_lithol = np.where(left_margin_region >= -5.0)[0][0]
    idx_min_lithol = np.where(left_margin_region <=5.0)[0][0]
    marginl_end = x[cond_xl][idx_min_lithol]

    marginl_wdt = np.abs(marginl_end - marginl_begin)

    #right side
    cond_xr = (x >= xr_begin) & (x <= xr_end)

    # topor_region = topography_interface[cond_xr]
    # idx_max_topor = np.where(topor_region == np.max(topor_region))[0][-1] #the last index of the maximum topography is more distal
    # marginr_begin = x[cond_xr][idx_max_topor]
    marginr_begin = xr_end

    # right_margin_region = lower_interface[cond_xr]
    right_margin_region = crustal_thickness[cond_xr]
    # idx_min_lithor = np.where(lithor_region == np.max(lithor_region))[0][-1]
    # idx_min_lithor = np.where(right_margin_region >= -5.0)[0][-1]
    idx_min_lithor = np.where(right_margin_region <=5.0)[0][-1]
    marginr_end = x[cond_xr][idx_min_lithor]

    marginr_wdt = np.abs(marginr_end - marginr_begin)

    return marginl_begin, marginl_end, marginl_wdt, marginr_begin, marginr_end, marginr_wdt

def measure_crustal_thickness(dataset, Nx, Nz, Lx, Lz, x_begin=600.0, x_end=900.0, rho_topo=200., rho_lower_crust=2850., topography_from_density=True):
    '''
    Measure the crustal thickness in a given region of the model.

    Parameters
    ----------
    dataset: xArray Dataset
        dataset with mandyoc data. Assure that it has the density data.
        
    Nx: int
        Number of points in x direction.

    Nz: int
        Number of points in z direction.

    Lx: float
        Length of the model in x direction.

    Lz: float
        Length of the model in z direction.

    topography: float
        Topography value to extract the interface.
        
    x_begin: float
        Initial x position to measure the crustal thickness.

    x_end: float
        Final x position to measure the crustal thickness.

    rho_topo: int
        Density of the topography interface.

    rho_lower_crust: int
        Density of the lower crust interface.

    topography_from_density: bool
        If True, the topography is extracted from the density field. Otherwise, the topography is extracted from the sp_surface_global_*.txt file.

    Returns
    -------

    position_max_thickness: float
        Position of the maximum crustal thickness in the selected region.

    topo_max_thickness: float
        Topography value in the maximum crustal thickness in the selected region.

    crustal_thickness: float
        Maximum crustal thickness in the selected region [km]

    lower_crust_interface: array_like
        Lower crust interface [km] corrected from air layer

    topography_interface: array_like
        Topography interface [km] corrected from air layer

    total_crustal_thickness: array_like
        Crustal thickness [km] corrected from air layer

    '''
    h_air = 40.0 #km
    x_aux = np.linspace(0, Lx/1000.0, Nx)
    z_aux = np.linspace(-Lz/1000.0, 0, Nz)
    Z = np.linspace(-Lz/1000.0, 0, 8001) #zi
    
    Rhoi = dataset.density #read_density(fpath, step, Nx, Nz)
    
    if(topography_from_density == True):
        topography_interface = _extract_interface(z_aux, Z, Nx, Rhoi, rho_topo) + h_air
    else:
        condx = (x_aux >= 100) & (x_aux <= 400)
        z_mean = np.mean(dataset.surface[condx])/1.0e3 + h_air
        topography_interface = dataset.surface/1.0e3 + h_air + np.abs(z_mean) #km + air layer correction

    lower_crust_interface = _extract_interface(z_aux, Z, Nx, Rhoi, rho_lower_crust) + h_air

    total_crustal_thickness = np.abs(lower_crust_interface - topography_interface)
    
    #selecting the region to measure the crustal thickness
    cond_x = (x_aux >= x_begin) & (x_aux <= x_end)

    x_region = x_aux[cond_x]  
    topo_region = topography_interface[cond_x]
    crustal_thickness_region = total_crustal_thickness[cond_x]

    #index of the maximum crustal thickness
    idx = np.where(crustal_thickness_region == np.max(crustal_thickness_region))[0][0]

    #information of the maximum crustal thickness
    position_max_thickness = x_region[idx]
    topo_max_thickness = topo_region[idx]
    crustal_thickness = crustal_thickness_region[idx]
    

    return position_max_thickness, topo_max_thickness, crustal_thickness, lower_crust_interface, topography_interface, total_crustal_thickness

def plot_rift_domains(ax, xnecking_left, xnecking_right, xhyper_left, xhyper_right, xexhumed_left, xexhumed_right, xoceanic, z_bar,
                      color_necking = 'xkcd:dark brown',
                      color_hyperextended = 'xkcd:gray',
                      color_exhumed_mantle = 'xkcd:bright orange',
                      color_oceanic = 'xkcd:cobalt blue',
                      lw=5):
    #Necking domain
    ax.plot(xnecking_left, z_bar, '-', color=color_necking, lw=lw, zorder=90) #left
    ax.plot(xnecking_right, z_bar, '-', color=color_necking, lw=lw, zorder=90) #right

    #Hyperextended domain
    ax.plot(xhyper_left, z_bar, color=color_hyperextended, lw=lw, zorder=90) #left
    ax.plot(xhyper_right, z_bar, color=color_hyperextended, lw=lw, zorder=90) #right

    #Exhumed mantle domain
    ax.plot(xexhumed_left, z_bar, color=color_exhumed_mantle, lw=lw, zorder=90) #left
    ax.plot(xexhumed_right, z_bar, color=color_exhumed_mantle, lw=lw, zorder=90) #right

    #Proto-oceanic/oceanic domain
    ax.plot(xoceanic, z_bar, color=color_oceanic, lw=lw, zorder=90) #left

def plot_tracked_particles_depth_coded(trackdataset, ax, i, plot_other_particles=False, size_other_particles=5):
    """
    Plot tracked particles in the subplot ax

    Parameters
    ----------
    trackdataset : xarray.Dataset
        Dataset containing the tracked particles
    ax : matplotlib.axes._subplots.AxesSubplot
        Subplot to plot the tracked particles
    i : int
        Current time step
    """
    x_track = trackdataset.xtrack.values[::-1]
    z_track = trackdataset.ztrack.values[::-1]
    P = trackdataset.ptrack.values[::-1]
    T = trackdataset.ttrack.values[::-1]
    time = trackdataset.time.values[::-1]
    steps = trackdataset.step.values[::-1]
    n = int(trackdataset.ntracked.values)
    nTotal = np.size(x_track)
    steps = nTotal//n

    x_track = np.reshape(x_track,(steps,n))
    z_track = np.reshape(z_track,(steps,n))
    P = np.reshape(P,(steps,n))
    T = np.reshape(T,(steps,n))
    particles_layers = trackdataset.particles_layers.values[::-1] #code of the tracked layers

    asthenosphere_code = 0
    mantle_lithosphere1_code = 1
    seed_code = 2
    mantle_lithosphere2_code = 3
    lower_crust_code = 4
    upper_crust_code = 5

    T_initial = T[0]
    P_initial = P[0]

    if(asthenosphere_code in particles_layers):
        cond_ast = particles_layers == asthenosphere_code
        particles_ast = particles_layers[cond_ast]

        cond_ast2plot = P_initial <= 4000 #only plot asthenosphere particles with initial pressure less than 4000 MPa
        plot_asthenosphere_particles = True

def plot_tracked_particles_depth_coded(trackdataset, ax, i, hcrust=35.0e3, markersize=4,
                                       plot_lower_crust_particles=False, plot_mantle_lithosphere_particles=True, plot_asthenosphere_particles=True,
                                       color_lower_crust='xkcd:brown',
                                       color_mlit_upper='xkcd:cerulean blue', color_mlit_intermediate='xkcd:scarlet', color_mlit_lower='xkcd:dark green', particles_alpha=0.7):
    """
    Plot tracked particles in the subplot ax

    Parameters
    ----------
    trackdataset : xarray.Dataset
        Dataset containing the tracked particles
    ax : matplotlib.axes._subplots.AxesSubplot
        Subplot to plot the tracked particles
    i : int
        Index of the current step
    hcrust : float
        Crustal thickness in m
    markersize : float
        Size of the markers
    plot_lower_crust_particles : bool
        Whether to plot lower crust particles
    plot_mantle_lithosphere_particles : bool
        Whether to plot mantle lithosphere particles
    plot_asthenosphere_particles : bool
        Whether to plot asthenosphere particles
    color_lower_crust : str
        Color of the lower crust particles
    color_mlit_upper : str
        Color of the upper mantle lithosphere particles
    color_mlit_intermediate : str
        Color of the intermediate mantle lithosphere particles
    color_mlit_lower : str
        Color of the lower mantle lithosphere particles
    particles_alpha : float
        Alpha transparency of the particles
    """
    x_track = trackdataset.xtrack.values[::-1]
    z_track = trackdataset.ztrack.values[::-1]
    P = trackdataset.ptrack.values[::-1]
    T = trackdataset.ttrack.values[::-1]
    time = trackdataset.time.values[::-1]
    steps = trackdataset.step.values[::-1]
    n = int(trackdataset.ntracked.values)
    nTotal = np.size(x_track)
    steps = nTotal//n

    x_track = np.reshape(x_track,(steps,n))
    z_track = np.reshape(z_track,(steps,n))
    P = np.reshape(P,(steps,n))
    T = np.reshape(T,(steps,n))
    particles_layers = trackdataset.particles_layers.values[::-1] #code of the tracked layers

    asthenosphere_code = 0
    mantle_lithosphere1_code = 1
    seed_code = 2
    mantle_lithosphere2_code = 3
    lower_crust_code = 4
    upper_crust_code = 5

    T_initial = T[0]
    P_initial = P[0]
    z_initial = z_track[0]
    x_initial = x_track[0]
    
    # color_lower_crust='xkcd:brown'
    # color_mlit_upper = 'xkcd:cerulean blue'
    # color_mlit_intermediate = 'xkcd:scarlet'
    # color_mlit_lower = 'xkcd:violet'

    h_air = 40.0e3 #m
    h_lithosphere = 120.0e3 #m
    #thickness of mantle lithosphere sections
    thickness_upper = 30.0e3
    thickness_intermediate = 30.0e3
    thickness_lower = h_lithosphere - (hcrust + thickness_upper + thickness_intermediate)

    zb_crust_and_air = -1.0 * (hcrust + h_air) #m z depth of the bottom of the crust considering air layer
    zb_upper = zb_crust_and_air - thickness_upper #m. The values of z_initial are negative, so we need to subtract
    zb_intermediate = zb_upper - thickness_intermediate #m. The values of z_initial are negative, so we need to subtract
    zb_lower = -1.0 * (h_lithosphere + h_air) #m z depth of the bottom of the lithosphere considering air layer

    if(asthenosphere_code in particles_layers):
        cond_ast = particles_layers == asthenosphere_code
        particles_ast = particles_layers[cond_ast]

        cond_ast2plot = P_initial <= 4000 #only plot asthenosphere particles with initial pressure less than 4000 MPa

        cond_upper_2plot_ast = (P_initial <= 4000) & (z_initial < zb_crust_and_air) & (z_initial >= zb_upper) #condition for upper particles - the values of z_initial are negative, so the logical condition is inverted
        cond_intermediate_2plot_ast = (P_initial <= 4000) & (z_initial < zb_upper) & (z_initial >= zb_intermediate) #condition for intermediate particles
        cond_lower_2plot_ast = (P_initial <= 4000) & (z_initial < zb_intermediate) & (z_initial >= zb_lower) #condition for lower particles

    else:
        plot_asthenosphere_particles = False
        cond_ast2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if((mantle_lithosphere1_code in particles_layers) | (mantle_lithosphere2_code in particles_layers)):
        cond_mlit = (particles_layers == mantle_lithosphere1_code) | (particles_layers == mantle_lithosphere2_code)
        particles_mlit = particles_layers[cond_mlit]

        # x_initial_mlit = x_initial[cond_mlit] #initial x position of lithospheric mantle particles
        # z_initial_mlit = z_initial[cond_mlit] #initial depth of lithospheric mantle particles

        cond_upper_2plot_mlit = (z_initial < zb_crust_and_air) & (z_initial >= zb_upper) #condition for upper particles - the values of z_initial are negative, so the logical condition is inverted
        cond_intermediate_2plot_mlit = (z_initial < zb_upper) & (z_initial >= zb_intermediate) #condition for intermediate particles
        cond_lower_2plot_mlit = (z_initial < zb_intermediate) & (z_initial >= zb_lower) #condition for lower particles
        
    else:
        plot_mantle_lithosphere_particles = False
        cond_mantle_lithosphere2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if(lower_crust_code in particles_layers):
        cond_crust = particles_layers == lower_crust_code
        particles_crust = particles_layers[cond_crust]

        T_initial_lower_crust = T_initial[cond_crust] #initial temperature of crustal particles
        T_initial_lower_crust_sorted = np.sort(T_initial_lower_crust)

        Ti_lower_crust_max = np.max(T_initial_lower_crust_sorted)
        mid_index = len(T_initial_lower_crust_sorted)//2
        Ti_lower_crust_mid = T_initial_lower_crust_sorted[mid_index]
        Ti_lower_crust_min = np.min(T_initial_lower_crust_sorted)

        cond_lower_crust2plot = (T_initial == Ti_lower_crust_min) | (T_initial == Ti_lower_crust_mid) | (T_initial == Ti_lower_crust_max)

    else:
        plot_lower_crust_particles = False
        cond_lower_crust2plot = np.arange(0, n, 1) == np.arange(0,n,1) + 1


    for particle, particle_layer in zip(range(n), particles_layers):
        #Plot particles in prop subplot

        if((plot_lower_crust_particles == True) & (particle_layer == lower_crust_code)): #crustal particles
            if(cond_lower_crust2plot[particle] == True):
                ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air/1.0e3, '.', color=color_lower_crust, markersize=markersize, zorder=61, alpha=particles_alpha)

        if((plot_mantle_lithosphere_particles == True) & ((particle_layer == mantle_lithosphere1_code) | (particle_layer == mantle_lithosphere2_code))): #lithospheric mantle particles
            # if(cond_mantle_lithosphere2plot[particle]==True):
                # print(f"Particle: {particle}, Layer: {particle_layer}, T_initial: {T_initial[particle]}")
            if(cond_upper_2plot_mlit[particle] == True):
                ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air/1.0e3, '.', color=color_mlit_upper, markersize=markersize, zorder=60, alpha=particles_alpha)
            elif(cond_intermediate_2plot_mlit[particle] == True):
                ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air/1.0e3, '.', color=color_mlit_intermediate, markersize=markersize, zorder=60, alpha=particles_alpha)
            elif(cond_lower_2plot_mlit[particle] == True):
                ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air/1.0e3, '.', color=color_mlit_lower, markersize=markersize, zorder=60, alpha=particles_alpha)
            # else:
            #     if(plot_other_particles == True):
            #         ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air, '.', color=color_other_particles, markersize=size_other_particles, zorder=60)

        if((plot_asthenosphere_particles == True) & (particle_layer == asthenosphere_code)):
            if(cond_upper_2plot_ast[particle] == True):
                ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air/1.0e3, '.', color=color_mlit_upper, markersize=markersize, zorder=60, alpha=particles_alpha)
            elif(cond_intermediate_2plot_ast[particle] == True):
                ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air/1.0e3, '.', color=color_mlit_intermediate, markersize=markersize, zorder=60, alpha=particles_alpha)
            elif(cond_lower_2plot_ast[particle] == True):
                ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air/1.0e3, '.', color=color_mlit_lower, markersize=markersize, zorder=60, alpha=particles_alpha)

def plot_ptt_paths_depth_coded(trackdataset, ax, instants=[], hcrust=35.0e3, plot_lower_crust_particles=False, plot_mantle_lithosphere_particles=True, plot_asthenosphere_particles=True, color_lower_crust='xkcd:brown', color_mlit_upper='xkcd:cerulean blue', color_mlit_intermediate='xkcd:scarlet', color_mlit_lower='xkcd:dark green'):
    """
    Plot PTt path of tracked particles in the subplot ax

    Parameters
    ----------
    trackdataset : xarray.Dataset
        Dataset containing the tracked particles
    ax : matplotlib.axes._subplots.AxesSubplot
        Subplot to plot the PTt path
    instants : list
        List of instants to plot the PTt path
    """
    x_track = trackdataset.xtrack.values[::-1]
    z_track = trackdataset.ztrack.values[::-1]
    P = trackdataset.ptrack.values[::-1]
    T = trackdataset.ttrack.values[::-1]
    time = trackdataset.time.values[::-1]
    steps = trackdataset.step.values[::-1]
    n = int(trackdataset.ntracked.values)
    nTotal = np.size(x_track)
    steps = nTotal//n

    x_track = np.reshape(x_track,(steps,n))
    z_track = np.reshape(z_track,(steps,n))
    P = np.reshape(P,(steps,n))
    T = np.reshape(T,(steps,n))
    particles_layers = trackdataset.particles_layers.values[::-1] #code of the tracked layers
    
    z_initial = z_track[0]
    x_initial = x_track[0]
    
    # color_lower_crust='xkcd:brown'
    # color_mlit_upper = 'xkcd:cerulean blue'
    # color_mlit_intermediate = 'xkcd:scarlet'
    # color_mlit_lower = 'xkcd:violet'
    markersize = 8

    h_air = 40.0e3 #m
    h_lithosphere = 120.0e3 #m
    #thickness of mantle lithosphere sections
    thickness_upper = 30.0e3
    thickness_intermediate = 30.0e3
    thickness_lower = h_lithosphere - (hcrust + thickness_upper + thickness_intermediate)

    zb_crust_and_air = -1.0 * (hcrust + h_air) #m z depth of the bottom of the crust considering air layer
    zb_upper = zb_crust_and_air - thickness_upper #m. The values of z_initial are negative, so we need to subtract
    zb_intermediate = zb_upper - thickness_intermediate #m. The values of z_initial are negative, so we need to subtract
    zb_lower = -1.0 * (h_lithosphere + h_air) #m z depth of the bottom of the lithosphere considering air layer

    asthenosphere_code = 0
    mantle_lithosphere1_code = 1
    seed_code = 2
    mantle_lithosphere2_code = 3
    lower_crust_code = 4
    upper_crust_code = 5
    T_initial = T[0]
    P_initial = P[0]

    if(asthenosphere_code in particles_layers):
        cond_ast = particles_layers == asthenosphere_code
        particles_ast = particles_layers[cond_ast]

        cond_ast2plot = P_initial <= 4000 #only plot asthenosphere particles with initial pressure less than 4000 MPa

        cond_upper_2plot_ast = (P_initial <= 4000) & (z_initial < zb_crust_and_air) & (z_initial >= zb_upper) #condition for upper particles - the values of z_initial are negative, so the logical condition is inverted
        cond_intermediate_2plot_ast = (P_initial <= 4000) & (z_initial < zb_upper) & (z_initial >= zb_intermediate) #condition for intermediate particles
        cond_lower_2plot_ast = (P_initial <= 4000) & (z_initial < zb_intermediate) & (z_initial >= zb_lower) #condition for lower particles

    else:
        plot_asthenosphere_particles = False
        cond_ast2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if((mantle_lithosphere1_code in particles_layers) | (mantle_lithosphere2_code in particles_layers)):
        cond_mlit = (particles_layers == mantle_lithosphere1_code) | (particles_layers == mantle_lithosphere2_code)
        particles_mlit = particles_layers[cond_mlit]

        # x_initial_mlit = x_initial[cond_mlit] #initial x position of lithospheric mantle particles
        # z_initial_mlit = z_initial[cond_mlit] #initial depth of lithospheric mantle particles

        cond_upper_2plot_mlit = (z_initial < zb_crust_and_air) & (z_initial >= zb_upper) #condition for upper particles - the values of z_initial are negative, so the logical condition is inverted
        cond_intermediate_2plot_mlit = (z_initial < zb_upper) & (z_initial >= zb_intermediate) #condition for intermediate particles
        cond_lower_2plot_mlit = (z_initial < zb_intermediate) & (z_initial >= zb_lower) #condition for lower particles

    else:
        plot_mantle_lithosphere_particles = False
        cond_mantle_lithosphere2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if(lower_crust_code in particles_layers):
        cond_crust = particles_layers == lower_crust_code
        particles_crust = particles_layers[cond_crust]

        T_initial_lower_crust = T_initial[cond_crust] #initial temperature of crustal particles
        T_initial_lower_crust_sorted = np.sort(T_initial_lower_crust)

        Ti_lower_crust_max = np.max(T_initial_lower_crust_sorted)
        mid_index = len(T_initial_lower_crust_sorted)//2
        Ti_lower_crust_mid = T_initial_lower_crust_sorted[mid_index]
        Ti_lower_crust_min = np.min(T_initial_lower_crust_sorted)

        cond_lower_crust2plot = (T_initial == Ti_lower_crust_min) | (T_initial == Ti_lower_crust_mid) | (T_initial == Ti_lower_crust_max)

    else:
        plot_lower_crust_particles = False
        cond_lower_crust2plot = np.arange(0, n, 1) == np.arange(0,n,1) + 1

    linewidth = 0.35
    markersize = 6
    color_crust='xkcd:brown'
    alpha = 0.7

    for particle, particle_layer in zip(range(n), particles_layers):
        #Plot particles in prop subplot

        if((plot_lower_crust_particles == True) & (particle_layer == lower_crust_code)):
            if(cond_lower_crust2plot[particle] == True):
                ax.plot(T[::, particle], P[::, particle], '-', color=color_crust, linewidth=linewidth, alpha=1.0, zorder=60) #PTt path
                
                if(len(instants)>0):
                    for instant in instants:
                        idx = find_nearest(time, instant)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color=color_lower_crust, markersize=markersize, zorder=60)
                else: #plotting points at each 5 Myr
                    for j in np.arange(0, time[-1], 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color=color_lower_crust, markersize=2, zorder=60)


        if((plot_mantle_lithosphere_particles == True) & ((particle_layer == mantle_lithosphere1_code) | (particle_layer == mantle_lithosphere2_code))): #lithospheric mantle particles
            if(cond_upper_2plot_mlit[particle] == True):
                ax.plot(T[::, particle], P[::, particle], '-', color=color_mlit_upper, linewidth=linewidth, alpha=alpha, zorder=61) #PTt path
                if(len(instants)>0):
                    for instant in instants:
                        idx = find_nearest(time, instant)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color=color_mlit_upper, markersize=markersize, zorder=60)
                else: #plotting points at each 5 Myr
                    for j in np.arange(0, time[-1], 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=markersize-2, zorder=60)
            elif(cond_intermediate_2plot_mlit[particle] == True):
                ax.plot(T[::, particle], P[::, particle], '-', color=color_mlit_intermediate, linewidth=linewidth, alpha=alpha, zorder=61)
                if(len(instants)>0):
                    for instant in instants:
                        idx = find_nearest(time, instant)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color=color_mlit_intermediate, markersize=markersize, zorder=60)
                else: #plotting points at each 5 Myr
                    for j in np.arange(0, time[-1], 5): 
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=markersize-2, zorder=60)
            elif(cond_lower_2plot_mlit[particle] == True):
                ax.plot(T[::, particle], P[::, particle], '-', color=color_mlit_lower, linewidth=linewidth, alpha=alpha, zorder=61)
                if(len(instants)>0):
                    for instant in instants:
                        idx = find_nearest(time, instant)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color=color_mlit_lower, markersize=markersize, zorder=60)
                else: #plotting points at each 5 Myr
                    for j in np.arange(0, time[-1], 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=markersize-2, zorder=60)

        if((plot_asthenosphere_particles == True) & (particle_layer == asthenosphere_code)):
            if(cond_upper_2plot_ast[particle] == True):
                ax.plot(T[::, particle], P[::, particle], '-', color=color_mlit_upper, linewidth=linewidth-1, alpha=alpha, zorder=61)
                if(len(instants)>0):
                    for instant in instants:
                        idx = find_nearest(time, instant)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color=color_mlit_upper, markersize=4, zorder=60)
                else: #plotting points at each 5 Myr
                    for j in np.arange(0, time[-1], 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=0.7, zorder=60)
            elif(cond_intermediate_2plot_ast[particle] == True):
                ax.plot(T[::, particle], P[::, particle], '-', color=color_mlit_intermediate, linewidth=linewidth-1, alpha=alpha, zorder=61)
                if(len(instants)>0):
                    for instant in instants:
                        idx = find_nearest(time, instant)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color=color_mlit_intermediate, markersize=4, zorder=60)
                else: #plotting points at each 5 Myr
                    for j in np.arange(0, time[-1], 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=0.7, zorder=60)
            elif(cond_lower_2plot_ast[particle] == True):
                ax.plot(T[::, particle], P[::, particle], '-', color=color_mlit_lower, linewidth=linewidth-1, alpha=alpha, zorder=61)
                if(len(instants)>0):    
                    for instant in instants:
                        idx = find_nearest(time, instant)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color=color_mlit_lower, markersize=4, zorder=60)
                else: #plotting points at each 5 Myr
                    for j in np.arange(0, time[-1], 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=0.7, zorder=60)

def plot_ptt_paths_depth_coded_frame(trackdataset, ax, i, current_time, hcrust=35.0e3, markersize=6, linewidth=0.35,
                                     plot_lower_crust_particles=False, plot_mantle_lithosphere_particles=True, plot_asthenosphere_particles=True,
                                     color_lower_crust='xkcd:brown',
                                     color_mlit_upper='xkcd:cerulean blue', color_mlit_intermediate='xkcd:scarlet', color_mlit_lower='xkcd:dark green',
                                     plot_steps=False,
                                     alpha=0.7):
    """
    Plot PTt path of tracked particles in the subplot ax

    Parameters
    ----------
    trackdataset : xarray.Dataset
        Dataset containing the tracked particles
    ax : matplotlib.axes._subplots.AxesSubplot
        Subplot to plot the PTt path
    i : int
        Current step index to plot the PTt path
    current_time : float
        Current time to plot the PTt path
    plot_lower_crust_particles : bool
        Whether to plot lower crust particles
    plot_mantle_lithosphere_particles : bool
        Whether to plot mantle lithosphere particles
    plot_asthenosphere_particles : bool
        Whether to plot asthenosphere particles
    color_lower_crust : str
        Color for lower crust particles
    color_mlit_upper : str
        Color for upper mantle lithosphere particles
    color_mlit_intermediate : str
        Color for intermediate mantle lithosphere particles
    color_mlit_lower : str
        Color for lower mantle lithosphere particles
    plot_steps : bool
        Whether to plot steps as black dots along the PTt path
    linewidth : float
        Width of the lines to plot
    markersize : float
        Size of the markers to plot
    alpha : float
        Transparency of the lines to plot
    """
    x_track = trackdataset.xtrack.values[::-1]
    z_track = trackdataset.ztrack.values[::-1]
    P = trackdataset.ptrack.values[::-1]
    T = trackdataset.ttrack.values[::-1]
    time = trackdataset.time.values[::-1]
    steps = trackdataset.step.values[::-1]
    n = int(trackdataset.ntracked.values)
    nTotal = np.size(x_track)
    steps = nTotal//n

    x_track = np.reshape(x_track,(steps,n))
    z_track = np.reshape(z_track,(steps,n))
    P = np.reshape(P,(steps,n))
    T = np.reshape(T,(steps,n))
    particles_layers = trackdataset.particles_layers.values[::-1] #code of the tracked layers
    
    z_initial = z_track[0]
    x_initial = x_track[0]
    
    # color_lower_crust='xkcd:brown'
    # color_mlit_upper = 'xkcd:cerulean blue'
    # color_mlit_intermediate = 'xkcd:scarlet'
    # color_mlit_lower = 'xkcd:violet'

    h_air = 40.0e3 #m
    h_lithosphere = 120.0e3 #m
    #thickness of mantle lithosphere sections
    thickness_upper = 30.0e3
    thickness_intermediate = 30.0e3
    thickness_lower = h_lithosphere - (hcrust + thickness_upper + thickness_intermediate)

    zb_crust_and_air = -1.0 * (hcrust + h_air) #m z depth of the bottom of the crust considering air layer
    zb_upper = zb_crust_and_air - thickness_upper #m. The values of z_initial are negative, so we need to subtract
    zb_intermediate = zb_upper - thickness_intermediate #m. The values of z_initial are negative, so we need to subtract
    zb_lower = -1.0 * (h_lithosphere + h_air) #m z depth of the bottom of the lithosphere considering air layer

    asthenosphere_code = 0
    mantle_lithosphere1_code = 1
    seed_code = 2
    mantle_lithosphere2_code = 3
    lower_crust_code = 4
    upper_crust_code = 5
    T_initial = T[0]
    P_initial = P[0]

    if(asthenosphere_code in particles_layers):
        cond_ast = particles_layers == asthenosphere_code
        particles_ast = particles_layers[cond_ast]

        cond_ast2plot = P_initial <= 4000 #only plot asthenosphere particles with initial pressure less than 4000 MPa

        cond_upper_2plot_ast = (P_initial <= 4000) & (z_initial < zb_crust_and_air) & (z_initial >= zb_upper) #condition for upper particles - the values of z_initial are negative, so the logical condition is inverted
        cond_intermediate_2plot_ast = (P_initial <= 4000) & (z_initial < zb_upper) & (z_initial >= zb_intermediate) #condition for intermediate particles
        cond_lower_2plot_ast = (P_initial <= 4000) & (z_initial < zb_intermediate) & (z_initial >= zb_lower) #condition for lower particles

    else:
        plot_asthenosphere_particles = False
        cond_ast2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if((mantle_lithosphere1_code in particles_layers) | (mantle_lithosphere2_code in particles_layers)):
        cond_mlit = (particles_layers == mantle_lithosphere1_code) | (particles_layers == mantle_lithosphere2_code)
        particles_mlit = particles_layers[cond_mlit]

        # x_initial_mlit = x_initial[cond_mlit] #initial x position of lithospheric mantle particles
        # z_initial_mlit = z_initial[cond_mlit] #initial depth of lithospheric mantle particles

        cond_upper_2plot_mlit = (z_initial < zb_crust_and_air) & (z_initial >= zb_upper) #condition for upper particles - the values of z_initial are negative, so the logical condition is inverted
        cond_intermediate_2plot_mlit = (z_initial < zb_upper) & (z_initial >= zb_intermediate) #condition for intermediate particles
        cond_lower_2plot_mlit = (z_initial < zb_intermediate) & (z_initial >= zb_lower) #condition for lower particles

    else:
        plot_mantle_lithosphere_particles = False
        cond_mantle_lithosphere2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if(lower_crust_code in particles_layers):
        cond_crust = particles_layers == lower_crust_code
        particles_crust = particles_layers[cond_crust]

        T_initial_lower_crust = T_initial[cond_crust] #initial temperature of crustal particles
        T_initial_lower_crust_sorted = np.sort(T_initial_lower_crust)

        Ti_lower_crust_max = np.max(T_initial_lower_crust_sorted)
        mid_index = len(T_initial_lower_crust_sorted)//2
        Ti_lower_crust_mid = T_initial_lower_crust_sorted[mid_index]
        Ti_lower_crust_min = np.min(T_initial_lower_crust_sorted)

        cond_lower_crust2plot = (T_initial == Ti_lower_crust_min) | (T_initial == Ti_lower_crust_mid) | (T_initial == Ti_lower_crust_max)

    else:
        plot_lower_crust_particles = False
        cond_lower_crust2plot = np.arange(0, n, 1) == np.arange(0,n,1) + 1


    for particle, particle_layer in zip(range(n), particles_layers):
        #Plot particles in prop subplot

        if((plot_lower_crust_particles == True) & (particle_layer == lower_crust_code)):
            if(cond_lower_crust2plot[particle] == True):
                ax.plot(T[:i, particle], P[:i, particle]/1.0e3, '-', color=color_lower_crust, linewidth=linewidth, alpha=alpha, zorder=60) #PTt path
                ax.plot(T[i, particle], P[i, particle]/1.0e3, '.', color=color_lower_crust, markersize=markersize) #current step
                #plotting points at each 5 Myr
                if(plot_steps==True):
                    for j in np.arange(0, current_time, 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=2, zorder=60)

        if((plot_mantle_lithosphere_particles == True) & ((particle_layer == mantle_lithosphere1_code) | (particle_layer == mantle_lithosphere2_code))): #lithospheric mantle particles
            if(cond_upper_2plot_mlit[particle] == True):
                ax.plot(T[:i, particle], P[:i, particle]/1.0e3, '-', color=color_mlit_upper, linewidth=linewidth, alpha=alpha, zorder=60) #PTt path
                ax.plot(T[i, particle], P[i, particle]/1.0e3, '.', color=color_mlit_upper, markersize=markersize, zorder=61) #current step
                #plotting points at each 5 Myr
                if(plot_steps==True):
                    for j in np.arange(0, current_time, 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=2, zorder=60)
            if(cond_intermediate_2plot_mlit[particle] == True):
                ax.plot(T[:i, particle], P[:i, particle]/1.0e3, '-', color=color_mlit_intermediate, linewidth=linewidth, alpha=alpha, zorder=60)
                ax.plot(T[i, particle], P[i, particle]/1.0e3, '.', color=color_mlit_intermediate, markersize=markersize, zorder=61) #current step
                #plotting points at each 5 Myr
                if(plot_steps==True):
                    for j in np.arange(0, current_time, 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle]/1.0e3, '.', color='xkcd:black', markersize=2, zorder=60)
            if(cond_lower_2plot_mlit[particle] == True):
                ax.plot(T[:i, particle], P[:i, particle]/1.0e3, '-', color=color_mlit_lower, linewidth=linewidth, alpha=alpha, zorder=60)
                ax.plot(T[i, particle], P[i, particle]/1.0e3, '.', color=color_mlit_lower, markersize=markersize, zorder=61) #current step
                #plotting points at each 5 Myr
                if(plot_steps==True):
                    for j in np.arange(0, current_time, 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle]/1.0e3, '.', color='xkcd:black', markersize=2, zorder=60)

        if((plot_asthenosphere_particles == True) & (particle_layer == asthenosphere_code)):
            if(cond_upper_2plot_ast[particle] == True):
                ax.plot(T[:i, particle], P[:i, particle]/1.0e3, '-', color=color_mlit_upper, linewidth=linewidth-1, alpha=alpha, zorder=60)
                ax.plot(T[i, particle], P[i, particle]/1.0e3, '.', color=color_mlit_upper, markersize=4, zorder=61) #current step
                #plotting points at each 5 Myr
                if(plot_steps==True):
                    for j in np.arange(0, current_time, 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=2, zorder=60)
            if(cond_intermediate_2plot_ast[particle] == True):
                ax.plot(T[:i, particle], P[:i, particle]/1.0e3, '-', color=color_mlit_intermediate, linewidth=linewidth-1, alpha=alpha, zorder=60)
                ax.plot(T[i, particle], P[i, particle]/1.0e3, '.', color=color_mlit_intermediate, markersize=4, zorder=61) #current step
                #plotting points at each 5 Myr
                if(plot_steps==True):
                    for j in np.arange(0, current_time, 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=2, zorder=60)
            if(cond_lower_2plot_ast[particle] == True):
                ax.plot(T[:i, particle], P[:i, particle]/1.0e3, '-', color=color_mlit_lower, linewidth=linewidth-1, alpha=alpha, zorder=60)
                ax.plot(T[i, particle], P[i, particle]/1.0e3, '.', color=color_mlit_lower, markersize=4, zorder=61) #current step
                #plotting points at each 5 Myr
                if(plot_steps==True):
                    for j in np.arange(0, current_time, 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle]/1.0e3, '.', color='xkcd:black', markersize=2, zorder=60)

def plot_tracked_particles(trackdataset, ax, i, plot_other_particles=False, color_other_particles='xkcd:black', size_other_particles=5):
    """
    Plot tracked particles in the subplot ax

    Parameters
    ----------
    trackdataset : xarray.Dataset
        Dataset containing the tracked particles
    ax : matplotlib.axes._subplots.AxesSubplot
        Subplot to plot the tracked particles
    i : int
        Current time step
    """
    x_track = trackdataset.xtrack.values[::-1]
    z_track = trackdataset.ztrack.values[::-1]
    P = trackdataset.ptrack.values[::-1]
    T = trackdataset.ttrack.values[::-1]
    time = trackdataset.time.values[::-1]
    steps = trackdataset.step.values[::-1]
    n = int(trackdataset.ntracked.values)
    nTotal = np.size(x_track)
    steps = nTotal//n

    x_track = np.reshape(x_track,(steps,n))
    z_track = np.reshape(z_track,(steps,n))
    P = np.reshape(P,(steps,n))
    T = np.reshape(T,(steps,n))
    particles_layers = trackdataset.particles_layers.values[::-1] #code of the tracked layers

    asthenosphere_code = 0
    mantle_lithosphere1_code = 1
    seed_code = 2
    mantle_lithosphere2_code = 3
    lower_crust_code = 4
    upper_crust_code = 5

    T_initial = T[0]
    P_initial = P[0]

    if(asthenosphere_code in particles_layers):
        cond_ast = particles_layers == asthenosphere_code
        particles_ast = particles_layers[cond_ast]

        cond_ast2plot = P_initial <= 4000 #only plot asthenosphere particles with initial pressure less than 4000 MPa
        plot_asthenosphere_particles = True
    else:
        plot_asthenosphere_particles = False
        cond_ast2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if((mantle_lithosphere1_code in particles_layers) | (mantle_lithosphere2_code in particles_layers)):
        cond_mlit = (particles_layers == mantle_lithosphere1_code) | (particles_layers == mantle_lithosphere2_code)
        particles_mlit = particles_layers[cond_mlit]

        T_initial_mlit = T_initial[cond_mlit] #initial temperature of lithospheric mantle particles
        T_initial_mlit_sorted = np.sort(T_initial_mlit)

        Ti_mlit_max = np.max(T_initial_mlit_sorted)
        mid_index = len(T_initial_mlit_sorted)//2
        Ti_mlit_mid = T_initial_mlit_sorted[mid_index]
        Ti_mlit_min = np.min(T_initial_mlit_sorted)

        cond_mantle_lithosphere2plot = (T_initial == Ti_mlit_min) | (T_initial == Ti_mlit_mid) | (T_initial == Ti_mlit_max)

        plot_mantle_lithosphere_particles = True

        dict_mlit_markers = {Ti_mlit_max: '*',
                            Ti_mlit_mid: '^',
                            Ti_mlit_min: 'D'}

        dict_mlit_colors = {Ti_mlit_min: 'xkcd:cerulean blue',
                            Ti_mlit_mid: 'xkcd:scarlet',
                            Ti_mlit_max: 'xkcd:dark green'}
    else:
        plot_mantle_lithosphere_particles = False
        cond_mantle_lithosphere2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if(lower_crust_code in particles_layers):
        cond_crust = particles_layers == lower_crust_code
        particles_crust = particles_layers[cond_crust]

        T_initial_lower_crust = T_initial[cond_crust] #initial temperature of crustal particles
        T_initial_lower_crust_sorted = np.sort(T_initial_lower_crust)

        Ti_lower_crust_max = np.max(T_initial_lower_crust_sorted)
        mid_index = len(T_initial_lower_crust_sorted)//2
        Ti_lower_crust_mid = T_initial_lower_crust_sorted[mid_index]
        Ti_lower_crust_min = np.min(T_initial_lower_crust_sorted)

        cond_lower_crust2plot = (T_initial == Ti_lower_crust_min) | (T_initial == Ti_lower_crust_mid) | (T_initial == Ti_lower_crust_max)
        plot_lower_crust_particles = True
    else:
        plot_lower_crust_particles = False
        cond_lower_crust2plot = np.arange(0, n, 1) == np.arange(0,n,1) + 1

    # linewidth = 0.85
    markersize = 8
    color_lower_crust='xkcd:brown'
    h_air = 40.0

    for particle, particle_layer in zip(range(n), particles_layers):
        #Plot particles in prop subplot

        if((plot_lower_crust_particles == True) & (particle_layer == lower_crust_code)): #crustal particles
            # print(particle_layer)
            if(cond_lower_crust2plot[particle] == True):
                ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air, '.', color=color_lower_crust, markersize=markersize-2, zorder=61)


        if((plot_mantle_lithosphere_particles == True) & ((particle_layer == mantle_lithosphere1_code) | (particle_layer == mantle_lithosphere2_code))): #lithospheric mantle particles
            if(cond_mantle_lithosphere2plot[particle]==True):
                # print(f"Particle: {particle}, Layer: {particle_layer}, T_initial: {T_initial[particle]}")
                ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air,
                            dict_mlit_markers[T_initial[particle]],
                            color=dict_mlit_colors[T_initial[particle]],
                            markersize=markersize-2, zorder=61)
            else:
                if(plot_other_particles == True):
                    ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air, '.', color=color_other_particles, markersize=size_other_particles, zorder=60)

        if((plot_asthenosphere_particles == True) & (particle_layer == asthenosphere_code)):
            if(cond_ast2plot[particle] == True):
                ax.plot(x_track[i, particle]/1.0e3, z_track[i, particle]/1.0e3+h_air, '.', color='xkcd:violet', markersize=markersize-6, zorder=60)

def plot_ptt_paths_three_particles(trackdataset, ax, i, current_time, plot_other_particles=True, color_other_particles='xkcd:black', size_other_particles=0.7):
    """
    Plot PTt path of tracked particles in the subplot ax

    Parameters
    ----------
    trackdataset : xarray.Dataset
        Dataset containing the tracked particles
    ax : matplotlib.axes._subplots.AxesSubplot
        Subplot to plot the PTt path
    instants : list
        List of instants to plot the PTt path
    """
    x_track = trackdataset.xtrack.values[::-1]
    z_track = trackdataset.ztrack.values[::-1]
    P = trackdataset.ptrack.values[::-1]
    T = trackdataset.ttrack.values[::-1]
    time = trackdataset.time.values[::-1]
    steps = trackdataset.step.values[::-1]
    n = int(trackdataset.ntracked.values)
    nTotal = np.size(x_track)
    steps = nTotal//n

    x_track = np.reshape(x_track,(steps,n))
    z_track = np.reshape(z_track,(steps,n))
    P = np.reshape(P,(steps,n))
    T = np.reshape(T,(steps,n))
    particles_layers = trackdataset.particles_layers.values[::-1] #code of the tracked layers

    asthenosphere_code = 0
    mantle_lithosphere1_code = 1
    seed_code = 2
    mantle_lithosphere2_code = 3
    lower_crust_code = 4
    upper_crust_code = 5
    T_initial = T[0]
    P_initial = P[0]

    if(asthenosphere_code in particles_layers):
        cond_ast = particles_layers == asthenosphere_code
        particles_ast = particles_layers[cond_ast]

        cond_ast2plot = P_initial <= 4000 #only plot asthenosphere particles with initial pressure less than 4000 MPa
        plot_asthenosphere_particles = True
    else:
        plot_asthenosphere_particles = False
        cond_ast2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if((mantle_lithosphere1_code in particles_layers) | (mantle_lithosphere2_code in particles_layers)):
        cond_mlit = (particles_layers == mantle_lithosphere1_code) | (particles_layers == mantle_lithosphere2_code)
        particles_mlit = particles_layers[cond_mlit]

        T_initial_mlit = T_initial[cond_mlit] #initial temperature of lithospheric mantle particles
        T_initial_mlit_sorted = np.sort(T_initial_mlit)

        Ti_mlit_max = np.max(T_initial_mlit_sorted)
        mid_index = len(T_initial_mlit_sorted)//2
        Ti_mlit_mid = T_initial_mlit_sorted[mid_index]
        Ti_mlit_min = np.min(T_initial_mlit_sorted)

        cond_mantle_lithosphere2plot = (T_initial == Ti_mlit_min) | (T_initial == Ti_mlit_mid) | (T_initial == Ti_mlit_max)

        plot_mantle_lithosphere_particles = True

        dict_mantle_lithosphere_markers = {Ti_mlit_max: '*',
                            Ti_mlit_mid: '^',
                            Ti_mlit_min: 'D'}

        dict_mantle_lithosphere_colors = {Ti_mlit_min: 'xkcd:cerulean blue',
                            Ti_mlit_mid: 'xkcd:scarlet',
                            Ti_mlit_max: 'xkcd:dark green'}
    else:
        plot_mantle_lithosphere_particles = False
        cond_mantle_lithosphere2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if(lower_crust_code in particles_layers):
        cond_crust = particles_layers == lower_crust_code
        particles_crust = particles_layers[cond_crust]

        T_initial_crust = T_initial[cond_crust] #initial temperature of crustal particles
        T_initial_crust_sorted = np.sort(T_initial_crust)

        Ti_crust_max = np.max(T_initial_crust_sorted)
        mid_index = len(T_initial_crust_sorted)//2
        Ti_crust_mid = T_initial_crust_sorted[mid_index]
        Ti_crust_min = np.min(T_initial_crust_sorted)

        cond_crust2plot = (T_initial == Ti_crust_min) | (T_initial == Ti_crust_mid) | (T_initial == Ti_crust_max)
        plot_lower_crust_particles = True
    else:
        plot_lower_crust_particles = False
        cond_crust2plot = np.arange(0, n, 1) == np.arange(0,n,1) + 1

    linewidth = 0.85
    markersize = 8
    color_lower_crust='xkcd:brown'
    ast_color = 'xkcd:violet'

    for particle, particle_layer in zip(range(n), particles_layers):
        #Plot particles in prop subplot

        if((plot_lower_crust_particles == True) & (particle_layer == lower_crust_code)):
            if(cond_crust2plot[particle] == True):
                ax.plot(T[:i, particle], P[:i, particle], '-', color=color_lower_crust, linewidth=linewidth, alpha=1.0, zorder=60) #PTt path
                #plotting points at each 5 Myr
                for j in np.arange(0, current_time, 5):
                    idx = find_nearest(time, j)
                    ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=2, zorder=60)

        if((plot_mantle_lithosphere_particles == True) & ((particle_layer == mantle_lithosphere1_code) | (particle_layer == mantle_lithosphere2_code))): #lithospheric mantle particles

                    if(cond_mantle_lithosphere2plot[particle]==True):
                        # print(f"Particle: {particle}, Layer: {particle_layer}, T_initial: {T_initial[particle]}"
                        ax.plot(T[i, particle], P[i, particle],
                                    dict_mantle_lithosphere_markers[T_initial[particle]],
                                    color=dict_mantle_lithosphere_colors[T_initial[particle]], markersize=10, zorder=61) #current PTt point

                        ax.plot(T[:i, particle], P[:i, particle], '-', color=dict_mantle_lithosphere_colors[T_initial[particle]], linewidth=linewidth, alpha=0.8, zorder=61) #PTt path
                        #plotting points at each 5 Myr
                        
                        for j in np.arange(0, current_time, 5):
                            idx = find_nearest(time, j)
                            ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=2, zorder=60)
                    else:
                        if(plot_other_particles == True): #plotting the other lithospheric mantle particles instead only 3
                            ax.plot(T[i, particle], P[i, particle],
                                        '.', color='xkcd:black',
                                        markersize=int(markersize/2), zorder=60) #current PTt point
                            ax.plot(T[:i, particle], P[:i, particle],
                                        '-', color='xkcd:black',
                                        linewidth=0.1, alpha = 1.0, zorder=60) #PTt path
                                #plotting points at each 5 Myr
                            for j in np.arange(0, current_time, 5):
                                idx = find_nearest(time, j)
                                ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=0.7, zorder=60)

        if((plot_asthenosphere_particles == True) & (particle_layer == asthenosphere_code)):
                    if(cond_ast2plot[particle] == True):
                        ax.plot(T[i, particle], P[i, particle], '.', color=ast_color, markersize=int(markersize/2), zorder=60)
                        ax.plot(T[:i, particle], P[:i, particle], '-', color=ast_color, linewidth=linewidth-1, alpha=1.0, zorder=60) #PTt path
                        #plotting points at each 5 Myr
                        for j in np.arange(0, current_time, 5):
                            idx = find_nearest(time, j)
                            ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:black', markersize=0.5, zorder=59)

def plot_ptt_paths(trackdataset, ax, instants=[], plot_other_particles=True, color_other_particles='xkcd:black', size_other_particles=0.7):
    """
    Plot PTt path of tracked particles in the subplot ax

    Parameters
    ----------
    trackdataset : xarray.Dataset
        Dataset containing the tracked particles
    ax : matplotlib.axes._subplots.AxesSubplot
        Subplot to plot the PTt path
    instants : list
        List of instants to plot the PTt path
    """
    x_track = trackdataset.xtrack.values[::-1]
    z_track = trackdataset.ztrack.values[::-1]
    P = trackdataset.ptrack.values[::-1]
    T = trackdataset.ttrack.values[::-1]
    time = trackdataset.time.values[::-1]
    steps = trackdataset.step.values[::-1]
    n = int(trackdataset.ntracked.values)
    nTotal = np.size(x_track)
    steps = nTotal//n

    x_track = np.reshape(x_track,(steps,n))
    z_track = np.reshape(z_track,(steps,n))
    P = np.reshape(P,(steps,n))
    T = np.reshape(T,(steps,n))
    particles_layers = trackdataset.particles_layers.values[::-1] #code of the tracked layers

    asthenosphere_code = 0
    mantle_lithosphere1_code = 1
    seed_code = 2
    mantle_lithosphere2_code = 3
    lower_crust_code = 4
    upper_crust_code = 5
    T_initial = T[0]
    P_initial = P[0]

    if(asthenosphere_code in particles_layers):
        cond_ast = particles_layers == asthenosphere_code
        particles_ast = particles_layers[cond_ast]

        cond_ast2plot = P_initial <= 4000 #only plot asthenosphere particles with initial pressure less than 4000 MPa
        plot_asthenosphere_particles = True
    else:
        plot_asthenosphere_particles = False
        cond_ast2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if((mantle_lithosphere1_code in particles_layers) | (mantle_lithosphere2_code in particles_layers)):
        cond_mlit = (particles_layers == mantle_lithosphere1_code) | (particles_layers == mantle_lithosphere2_code)
        particles_mlit = particles_layers[cond_mlit]

        T_initial_mlit = T_initial[cond_mlit] #initial temperature of lithospheric mantle particles
        T_initial_mlit_sorted = np.sort(T_initial_mlit)

        Ti_mlit_max = np.max(T_initial_mlit_sorted)
        mid_index = len(T_initial_mlit_sorted)//2
        Ti_mlit_mid = T_initial_mlit_sorted[mid_index]
        Ti_mlit_min = np.min(T_initial_mlit_sorted)

        cond_mantle_lithosphere2plot = (T_initial == Ti_mlit_min) | (T_initial == Ti_mlit_mid) | (T_initial == Ti_mlit_max)

        plot_mantle_lithosphere_particles = True

        dict_mlit_markers = {Ti_mlit_max: '*',
                            Ti_mlit_mid: '^',
                            Ti_mlit_min: 'D'}

        dict_mlit_colors = {Ti_mlit_min: 'xkcd:cerulean blue',
                            Ti_mlit_mid: 'xkcd:scarlet',
                            Ti_mlit_max: 'xkcd:dark green'}
    else:
        plot_mantle_lithosphere_particles = False
        cond_mantle_lithosphere2plot = np.arange(0, n, 1) == np.arange(0, n, 1) + 1

    if(lower_crust_code in particles_layers):
        cond_crust = particles_layers == lower_crust_code
        particles_crust = particles_layers[cond_crust]

        T_initial_crust = T_initial[cond_crust] #initial temperature of crustal particles
        T_initial_crust_sorted = np.sort(T_initial_crust)

        Ti_crust_max = np.max(T_initial_crust_sorted)
        mid_index = len(T_initial_crust_sorted)//2
        Ti_crust_mid = T_initial_crust_sorted[mid_index]
        Ti_crust_min = np.min(T_initial_crust_sorted)

        cond_crust2plot = (T_initial == Ti_crust_min) | (T_initial == Ti_crust_mid) | (T_initial == Ti_crust_max)
        plot_lower_crust_particles = True
    else:
        plot_lower_crust_particles = False
        cond_crust2plot = np.arange(0, n, 1) == np.arange(0,n,1) + 1

    linewidth = 0.85
    markersize = 8
    color_crust='xkcd:brown'

    for particle, particle_layer in zip(range(n), particles_layers):
        #Plot particles in prop subplot

        if((plot_lower_crust_particles == True) & (particle_layer == lower_crust_code)):
            if(cond_crust2plot[particle] == True):
                ax.plot(T[::, particle], P[::, particle], '-', color=color_crust, linewidth=linewidth, alpha=1.0, zorder=60) #PTt path
                
                if(len(instants)>0):
                    for instant in instants:
                        idx = find_nearest(time, instant)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color=color_crust, markersize=markersize, zorder=60)
                else: #plotting points at each 5 Myr
                    for j in np.arange(0, time[-1], 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color=color_crust, markersize=2, zorder=60)


        if((plot_mantle_lithosphere_particles == True) & ((particle_layer == mantle_lithosphere1_code) | (particle_layer == mantle_lithosphere2_code))): #lithospheric mantle particles
            if(cond_mantle_lithosphere2plot[particle]==True):
                ax.plot(T[::, particle], P[::, particle], '-', color=dict_mlit_colors[T_initial[particle]], linewidth=linewidth, alpha=0.8, zorder=61) #PTt path
                
                if(len(instants)>0):
                    for instant in instants:
                        idx = find_nearest(time, instant)
                        ax.plot(T[idx, particle], P[idx, particle], dict_mlit_markers[T_initial[particle]], color=dict_mlit_colors[T_initial[particle]], markersize=markersize, zorder=60)
                else: #plotting points at each 5 Myr
                    for j in np.arange(0, time[-1], 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], dict_mlit_markers[T_initial[particle]], color=dict_mlit_colors[T_initial[particle]], markersize=markersize, zorder=60)
            else:
                if(plot_other_particles == True):
                    ax.plot(T[::, particle], P[::, particle], '-', color=color_other_particles, linewidth=0.1, alpha=1.0, zorder=60)
                    for j in np.arange(0, time[-1], 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color=color_other_particles, markersize=size_other_particles, zorder=59)

        if((plot_asthenosphere_particles == True) & (particle_layer == asthenosphere_code)):
            if(cond_ast2plot[particle] == True):
                ax.plot(T[::, particle], P[::, particle], '-', color='xkcd:violet', linewidth=linewidth-1, alpha=0.8, zorder=61)
                if(len(instants)>0):
                    for instant in instants:
                        idx = find_nearest(time, instant)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:violet', markersize=4, zorder=60)
                else: #plotting points at each 5 Myr
                    for j in np.arange(0, time[-1], 5):
                        idx = find_nearest(time, j)
                        ax.plot(T[idx, particle], P[idx, particle], '.', color='xkcd:violet', markersize=0.7, zorder=60)

def plot_property(dataset, prop, xlims, ylims, model_path,
                fig,
                ax,
                print_time=True,
                correction_factor=0.0,
                plot_isotherms=True, isotherms=[400, 600, 800, 1000, 1300],
                topo_from_density=True,
                plot_particles=False,
                particle_size=0.2,
                particle_marker="o",
                plot_colorbar=True,
                bbox_to_anchor=(0.85,#horizontal position respective to parent_bbox or "loc" position
                                0.3,# vertical position
                                0.12,# width
                                0.35),
                ncores=20,
                step_plot=1,
                plot_melt=False,
                color_incremental_melt = 'xkcd:bright pink',
                color_depleted_mantle='xkcd:bright purple',):
    '''
    Plot data from mandyoc according to a given property and domain limits.

    Parameters
    ----------
    dataset: :class:`xarray.Dataset`
        Dataset containing mandyoc data for a single time step.
    
    prop: str
        Property from mandyoc.

    xlims: list
        List with the limits of x axis
        
    ylims: list
        List with the limits of y axis

    model_path: str
        Path to model

    output_path: str
        Path to save outputs
        
    save_frames: bool
        True to save frame by frames
        False to do not save the frames
    '''
    
    props_label = {'density':              r'$\mathrm{[kg/m^3]}$',
                   'radiogenic_heat':       'log(W/kg)',
                   'lithology':            r'log$(\epsilon_{II})$',
                   'pressure':              'P [GPa]',
                   'strain':               r'Accumulated strain [$\varepsilon$]',
                   'strain_rate':          r'log($\dot{\varepsilon}$)',
#                    'strain_rate':          r'$\dot{\varepsilon}$',
                   'temperature':          r'$^{\circ}\mathrm{[C]}$',
                   'temperature_anomaly':  r'Temperature anomaly $^{\circ}\mathrm{[C]}$',
                   'topography':            'Topography [km]',
                   'viscosity':             'log(Pa.s)'
                   }
    
    props_cmap = {'density': 'viridis',
                  'radiogenic_heat': 'inferno',
                  'lithology': 'viridis',
                  'pressure': 'viridis',
#                   'strain': 'viridis', #Default. Comment this line and uncomment one of the options bellow
#                   'strain': 'cividis',
#                   'strain': 'Greys',
                  'strain': 'inferno',
#                   'strain': 'magma',
                  'strain_rate': 'viridis',
                  'temperature': 'viridis',
                  'temperature_anomaly': 'RdBu_r',
                  'topography': '',
                  'viscosity': 'viridis'
                   }

    #limits of colorbars
    vals_minmax = {'density':             [0.0, 3378.],
                   'radiogenic_heat':     [1.0E-13, 1.0E-9],
                   'lithology':           [None, None],
                   'pressure':            [-1.0E-3, 1.0],
                   'strain':              [None, None],
                   'strain_rate':         [1.0E-18, 1.0E-14],
                #    'strain_rate':         [1.0E-20, 1.0E-15],
#                    'strain_rate':         [np.log10(1.0E-19), np.log10(1.0E-14)],
                   'temperature':         [0, 1600],
                   'temperature_anomaly': [-150, 150],
                   'surface':             [-6, 6],
                   'viscosity':           [1.0E18, 1.0E25],
#                    'viscosity':           [np.log10(1.0E18), np.log10(1.0E25)]
                  }

    Nx = int(dataset.nx)
    Nz = int(dataset.nz)
    Lx = float(dataset.lx)
    Lz = float(dataset.lz)
    instant = np.round(float(dataset.time), 2)
    
    xi = np.linspace(0, Lx/1000, Nx)
    zi = np.linspace(-Lz/1000+40, 0+40, Nz) #km, +40 to compensate the air layer above sea level
    xx, zz = np.meshgrid(xi, zi)
    
    #creating Canvas
    label_size=12
    plt.rc('xtick', labelsize = label_size)
    plt.rc('ytick', labelsize = label_size)
    
    #plot Time in Myr
    # ax.text(0.68, 1.035, ' {:01} Myr'.format(instant), bbox=dict(facecolor='white', edgecolor='white', alpha=0.0), fontsize = 14, zorder=52, transform=ax.transAxes)
    if(print_time == True):
        ax.text(0.78, 1.035, ' {:01} Myr'.format(instant), bbox=dict(facecolor='white', edgecolor='white', alpha=0.0), fontsize = 14, zorder=52, transform=ax.transAxes)
    
    val_minmax = vals_minmax[prop]
    
    if(plot_isotherms == True and prop != 'surface'): #Plot isotherms
        Temperi = dataset.temperature.T
        
        isot_colors = []
        for isotherm in isotherms:
            isot_colors.append('red')
            
        cs = ax.contour(xx, zz, Temperi, 100, levels=isotherms, colors=isot_colors)
        
        # if(instant == instants[0]):
        #     fmt = {}
        #     for level, isot in zip(cs.levels, isotherms):
        #         fmt[level] = str(level) + r'$^{\circ}$C'

        #     ax.clabel(cs, cs.levels, fmt=fmt, inline=True, use_clabeltext=True)

    if(plot_melt == True and prop != 'surface'):

        #TRYING TO PLOT MELT
        # melt = xr.open_dataset(f'{model_path}/_output_melt.nc')
        melt = dataset.melt.T #depleted mantle

        levels_contourf = np.arange(0.02, 0.5, 0.02)
        for i in levels_contourf:
            alpha = 0.9/levels_contourf[-1]*i + 0.1

            cs0 = ax.contourf(xx, zz, melt, levels=[i, i+0.02], colors='xkcd:black', alpha=alpha, zorder=30)


        levels_melt = [0.1, 0.2] #np.arange(0.2, 0.7, 0.4)#[0.2, 0.6]
        cs = ax.contour(xx,
                    zz,
                    melt,
                    levels = levels_melt,
                    colors='xkcd:blue',
                    # cmap = 'inferno',
                    zorder=30,
                    )
        
        
        # incremental_melt = xr.open_dataset(f'{model_path}/_output_incremental_melt.nc')
        incremental_melt = dataset.incremental_melt.T
        scale=1.0e4
        
        levels_incremental_melt = [5.0e-10, 1.0e-6]

        ax.contourf(xx,
                    zz,
                    incremental_melt/scale,#, 'dashdot', 'dashed'],
                    levels = levels_incremental_melt,
                    colors='xkcd:bright pink',#['xkcd:bright pink', 'xkcd:pink'],
                    # colors=colors,
                    alpha=0.4,
                    zorder=30)
        
        ax.contour(xx,
                    zz,
                    incremental_melt/scale,
                    linestyles=['solid'],#, 'dashdot', 'dashed'],
                    levels = [5.0E-10,1.0E-8],
                    colors='xkcd:bright pink',#['xkcd:bright pink', 'xkcd:pink'],
                    # colors=colors,
                    alpha=1.0,
                    zorder=30)
        
    #dealing with special data
    if(prop == 'lithology'):
        data = dataset['strain']
        
    elif(prop == 'temperature_anomaly'):
        #removing horizontal mean temperature
        A = dataset['temperature']
        B = A.T #shape: (Nz, Nx)
        C = np.mean(B, axis=1) #shape: 151==Nz
        D = (B.T - C) #B.T (Nx,Nz) para conseguir subtrair C
        data = D
        
    elif(prop == 'surface'):
        # print('Dealing with data')
        # topo_from_density = True
        # topo_from_density = False
        
        if(topo_from_density == True):
            Rhoi = dataset.density.T
            interfaces=[2900, 3365]
            ##Extract layer topography
            z = np.linspace(Lz/1000.0, 0, Nz)
            Z = np.linspace(Lz/1000.0, 0, 8001) #zi
            x = np.linspace(Lx/1000.0, 0, Nx)

            topo_interface = _extract_interface(z, Z, Nx, Rhoi, 300.) #200 kg/m3 = air/crust interface
            
            condx = (xi >= 100) & (xi <= 600)
            z_mean = np.mean(topo_interface[condx])
            
            topo_interface -= np.abs(z_mean)
            topo_interface = -1.0*topo_interface

            data = topo_interface
        else:
            condx = (xi >= 100) & (xi <= 600)
            z_mean = np.mean(dataset.surface[condx])/1.0e3 + 40.0

            data = dataset.surface/1.0e3 + 40.0 + np.abs(z_mean) #km + air layer correction
            
            
    elif(prop == 'pressure'):
        data = dataset[prop]/1.0E9 #GPa
        
    else:
        data = dataset[prop] 
        
    if(prop == 'strain_rate' or prop == 'radiogenic_heat' or prop == 'strain_rate' or prop == 'viscosity'): #properties that need a lognorm colorbar
        im = ax.imshow(data.T,
                       cmap = props_cmap[prop],
                       origin='lower',
                       extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40),
                       norm = LogNorm(vmin=val_minmax[0], vmax=val_minmax[1]),
#                        vmin=val_minmax[0], vmax=val_minmax[1],
                       aspect = 'auto')
        
        #creating colorbar
        if(plot_colorbar):
            axins1 = inset_axes(ax,
                                loc='lower right',
                                width="100%",  # respective to parent_bbox width
                                height="100%",  # respective to parent_bbox width
                                bbox_to_anchor=(0.7,#horizontal position respective to parent_bbox or "loc" position
                                                0.3,# vertical position
                                                0.25,# width
                                                0.05),# height
                                bbox_transform=ax.transAxes
                                )

            clb = fig.colorbar(im,
                            cax=axins1,
                            # ticks=[-20, -18, -16, -14],#ticks,
                            orientation='horizontal',
                            fraction=0.09,
                            pad=0.2,
                            format=_log_fmt)

            clb.set_label(props_label[prop], fontsize=12)
            clb.ax.tick_params(labelsize=12)
            clb.minorticks_off()
    
    elif (prop == 'density' or prop == 'pressure' or prop == 'temperature' or prop == 'temperature_anomaly'): #properties that need a regular colorbar
        im = ax.imshow(data.T,
                       cmap = props_cmap[prop],
                       origin='lower',
                       extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40),
                       vmin=val_minmax[0], vmax=val_minmax[1],
                       aspect = 'auto')
        
        if(plot_colorbar):
            axins1 = inset_axes(ax,
                                loc='lower right',
                                width="100%",  # respective to parent_bbox width
                                height="100%",  # respective to parent_bbox width
                                bbox_to_anchor=(0.05,#horizontal position respective to parent_bbox or "loc" position
                                                0.3,# vertical position
                                                0.4,# width
                                                0.05),# height
                                bbox_transform=ax.transAxes
                                )
            
    #         ticks = np.linspace(val_minmax[0], val_minmax[1], 6, endpoint=True)

            #precision of colorbar ticks
            if(prop == 'pressure'): 
                fmt = '%.2f'
            else:
                fmt = '%.0f'
                
            clb = fig.colorbar(im,
                            cax=axins1,
    #                            ticks=ticks,
                            orientation='horizontal',
                            fraction=0.08,
                            pad=0.2,
                            format=fmt)

            clb.set_label(props_label[prop], fontsize=12)
            clb.ax.tick_params(labelsize=12)
            clb.minorticks_off()
        
    elif(prop == 'strain'):
        im = ax.imshow(data.T,
                       cmap = props_cmap[prop],
                       origin='lower',
                       extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40),
                       vmin = float(data.min()),
                       vmax = float(data.max()),
                       aspect = 'auto')
        
    elif(prop == 'surface'):
        ax.plot(dataset.x/1.0e3, data, alpha = 1, linewidth = 2.0, color = "blueviolet")
        
    elif(prop == 'lithology'): #shaded lithology plot
        cr = 255.
        color_uc = (228. / cr, 156. / cr, 124. / cr)
        color_lc = (240. / cr, 209. / cr, 188. / cr)
        color_lit = (155. / cr, 194. / cr, 155. / cr)
        color_ast = (207. / cr, 226. / cr, 205. / cr)

        # cr = 255.
        # color_sed = (241./cr,184./cr,68./cr)
        # color_dec = (137./cr,81./cr,151./cr)
        # color_uc = (228./cr,156./cr,124./cr)
        # color_lc = (240./cr,209./cr,188./cr)
        # color_lit = (155./cr,194./cr,155./cr)
        # color_mlit_uc = (180. / cr, 194. / cr, 162. / cr)
        # color_mlit_lc = (155. / cr, 194. / cr, 155. / cr)
        # color_ast = (207./cr,226./cr,205./cr)
        
        Rhoi = dataset.density.T
        # interfaces=[2900, 3365]
        
        # ##Extract layer topography
        # z = np.linspace(Lz/1000.0, 0, Nz)
        # Z = np.linspace(Lz/1000.0, 0, 8001) #zi
        # x = np.linspace(Lx/1000.0, 0, Nx)
            
        # topo_interface = _extract_interface(z, Z, Nx, Rhoi, 300.) #200 kg/m3 = air/crust interface
        # condx = (xi >= 100) & (xi <= 400)
        # z_mean = np.mean(topo_interface[condx])
        # topo_interface -= np.abs(z_mean)
        # topo_interface = -1.0*topo_interface
        
        #Density field
        ax.contourf(xx,
                    zz+correction_factor,
                    Rhoi,
                    levels = [200., 2750, 2900, 3365, 3900],
                    colors = [color_uc, color_lc, color_lit, color_ast],
                    # levels=[200.,2350,2450,2750,2900,3325,3355,3365,3378],
                    # colors=[color_sed,color_dec,color_uc,color_lc,color_lit,color_mlit_uc,color_mlit_lc,color_ast],
                    )
        #Strain shaded areas
        im=ax.imshow(data.T,
                     cmap = 'Greys',
                     origin = 'lower',
                     extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40+correction_factor, 0 + 40+correction_factor),
                     # extent = (xlims[0], xlims[1], ylims[0], ylims[1]),
                     zorder = 50,
                     alpha = 0.2, vmin=-0.5,
                     vmax = 0.7,
                     aspect = 'auto')
        #legend box
        if(plot_colorbar == True):
            bv1 = inset_axes(ax,
                            loc='lower right',
                            width="100%",  # respective to parent_bbox width
                            height="100%",  # respective to parent_bbox width
                            bbox_to_anchor=bbox_to_anchor,
                            bbox_transform=ax.transAxes
                            )
            
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
            bv1.contourf(
                xxA,
                yyA,
                A,
                levels=[air_threshold, 2750, 2900, 3365, 3900],
                colors=[color_uc, color_lc, color_lit, color_ast],
                extent=[-0.5, 0.9, 0, 1.5]
            )

            bv1.imshow(
                xxA[::-1, :],
                extent=[-0.5, 0.9, 0, 1.5],
                zorder=100,
                alpha=0.2,
                cmap=plt.get_cmap("Greys"),
                vmin=-0.5,
                vmax=0.9,
                aspect='auto'
            )

            bv1.set_yticklabels([])
            bv1.set_xlabel(r"log$(\varepsilon_{II})$", size=10)
            bv1.tick_params(axis='x', which='major', labelsize=10)
            bv1.set_xticks([-0.5, 0, 0.5])
            bv1.set_yticks([])
            bv1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
    
    if(plot_particles == True):
        if(prop != 'surface'):
            # ncores = 20
            data_x, data_z, data_ID, data_lithology, data_strain = _read_step(model_path, f"step_{int(dataset.step)}_", ncores)
            # ax.scatter(data_x/1000, data_z/1000, 2, c='xkcd:black', marker='.', zorder=30)

            cond_litho = data_lithology > 1
            cond_mb = data_lithology == 1
            cond_ast = data_lithology == 0

            # if(prop=='lithology'):
            #     color_litho = 'xkcd:bright pink'
            #     color_ast = 'xkcd:black'
            # else:
            #     color_litho = 'xkcd:bright green'
            #     color_ast = 'xkcd:black'

            color_litho = 'xkcd:black'
            # color_mb = 'xkcd:neon green'
            color_mb = 'xkcd:black'
            color_ast = 'xkcd:bright pink'
            color_mb = 'xkcd:black'
            # color_mb = 'xkcd:neon green'

            ax.plot(data_x[cond_litho][::step_plot]/1000, data_z[cond_litho][::step_plot]/1000+40, particle_marker, color=color_litho, markersize=particle_size, alpha=1.0, zorder=30)
            ax.plot(data_x[cond_mb][::step_plot]/1000, data_z[cond_mb][::step_plot]/1000+40, particle_marker, color=color_mb, markersize=particle_size, alpha=1.0, zorder=30)
            ax.plot(data_x[cond_ast][::step_plot]/1000, data_z[cond_ast][::step_plot]/1000+40, particle_marker, color=color_ast, markersize=particle_size, alpha=1.0, zorder=30)
            # ax.plot(data_x[cond_ast][::step_plot*4]/1000, data_z[cond_ast][::step_plot*4]/1000+40, particle_marker, color=color_ast, markersize=particle_size-0.95, alpha=1.0, zorder=30)
    
    #Fill above topography with white color
    if(prop != 'surface'):
        # topo_from_density = True
        # topo_from_density = False
        if(topo_from_density==True):
            Rhoi = dataset.density.T
            # interfaces=[2900, 3365]
            # ##Extract layer topography
            z = np.linspace(Lz/1000.0, 0, Nz)
            Z = np.linspace(Lz/1000.0, 0, 8001) #zi
            x = np.linspace(Lx/1000.0, 0, Nx)

            topo_interface = _extract_interface(z, Z, Nx, Rhoi, 200.) #200 kg/m3 = air/crust interface
            # condx = (xi >= 100) & (xi <= 400)
            # z_mean = np.mean(topo_interface[condx])
            # topo_interface -= np.abs(z_mean)
            topo_interface = -1.0*topo_interface + 40.0 - correction_factor
        else:
            topo_interface = dataset.surface/1.0e3 + 40.0

        # topo_interface = dataset.surface/1.0e3 + 40.0
        xaux = xx[0]
        condaux = (xaux>=xlims[0]) & (xaux<=xlims[1])
        xaux = xaux[condaux]
        xaux[0] += 2
        xaux[-1] -= 2
        ax.fill_between(xaux, topo_interface[condaux], ylims[-1]-0.8, color='white', alpha=1.0, zorder=51)

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        # ax.set_xlabel("Distance (km)", fontsize = label_size)
        # ax.set_ylabel("Depth (km)", fontsize = label_size)
        
    else:
        ax.grid('-k', alpha=0.7)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        # ax.set_xlabel("Distance (km)", fontsize = label_size)
        # ax.set_ylabel("Topography (km)", fontsize = label_size)

def single_plot(dataset, prop, xlims, ylims, model_path, output_path,
                save_frames=True,
                plot_isotherms=True,
                topo_from_density=True,
                plot_particles=False,
                particle_size=0.2,
                particle_marker="o",
                plot_colorbar=True,
                ncores=20,
                step_plot=1,
                isotherms=[400, 600, 800, 1000, 1300],
                # isotherms=[800, 1300],
                plot_melt=False, melt_method='dry'):
    '''
    Plot and save data from mandyoc according to a given property and domain limits.

    Parameters
    ----------
    dataset: :class:`xarray.Dataset`
        Dataset containing mandyoc data for a single time step.
    
    prop: str
        Property from mandyoc.

    xlims: list
        List with the limits of x axis
        
    ylims: list
        List with the limits of y axis

    model_path: str
        Path to model

    output_path: str
        Path to save outputs
        
    save_frames: bool
        True to save frame by frames
        False to do not save the frames
    '''
    
    props_label = {'density':              r'$\mathrm{[kg/m^3]}$',
                   'radiogenic_heat':       'log(W/kg)',
                   'lithology':            r'log$(\epsilon_{II})$',
                   'pressure':              'P [GPa]',
                   'strain':               r'Accumulated strain [$\varepsilon$]',
                   'strain_rate':          r'log($\dot{\varepsilon}$)',
#                    'strain_rate':          r'$\dot{\varepsilon}$',
                   'temperature':          r'$^{\circ}\mathrm{[C]}$',
                   'temperature_anomaly':  r'Temperature anomaly $^{\circ}\mathrm{[C]}$',
                   'topography':            'Topography [km]',
                   'viscosity':             'log(Pa.s)'
                   }
    
    props_cmap = {'density': 'viridis',
                  'radiogenic_heat': 'inferno',
                  'lithology': 'viridis',
                  'pressure': 'viridis',
#                   'strain': 'viridis', #Default. Comment this line and uncomment one of the options bellow
#                   'strain': 'cividis',
#                   'strain': 'Greys',
                  'strain': 'inferno',
#                   'strain': 'magma',
                  'strain_rate': 'viridis',
                  'temperature': 'viridis',
                  'temperature_anomaly': 'RdBu_r',
                  'topography': '',
                  'viscosity': 'viridis'
                   }

    #limits of colorbars
    vals_minmax = {'density':             [0.0, 3378.],
                   'radiogenic_heat':     [1.0E-13, 1.0E-9],
                   'lithology':           [None, None],
                   'pressure':            [-1.0E-3, 1.0],
                   'strain':              [None, None],
                   'strain_rate':         [1.0E-19, 1.0E-14],
#                    'strain_rate':         [np.log10(1.0E-19), np.log10(1.0E-14)],
                   'temperature':         [0, 1600],
                   'temperature_anomaly': [-150, 150],
                   'surface':             [-6, 6],
                   'viscosity':           [1.0E18, 1.0E25],
#                    'viscosity':           [np.log10(1.0E18), np.log10(1.0E25)]
                  }

    model_name = model_path.split('/')[-1] #os.path.split(model_path)[0].split('/')[-1]

    Nx = int(dataset.nx)
    Nz = int(dataset.nz)
    Lx = float(dataset.lx)
    Lz = float(dataset.lz)
    instant = np.round(float(dataset.time), 2)
    
    xi = np.linspace(0, Lx/1000, Nx)
    zi = np.linspace(-Lz/1000+40, 0+40, Nz) #km, +40 to compensate the air layer above sea level
    xx, zz = np.meshgrid(xi, zi)
    h_air = 40.0 #km
    #creating Canvas
    plt.close()
    label_size=12
    plt.rc('xtick', labelsize = label_size)
    plt.rc('ytick', labelsize = label_size)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12*(Lz/Lx)), constrained_layout = True)
    # fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout = True)
    #plot Time in Myr
    ax.text(0.85, 1.05, ' {:01} Myr'.format(instant), fontsize = 18, zorder=52, transform=ax.transAxes)
    
    val_minmax = vals_minmax[prop]
    
    if(plot_isotherms == True and prop != 'surface'): #Plot isotherms
        Temperi = dataset.temperature.T
        
        isot_colors = []
        for isotherm in isotherms:
            isot_colors.append('red')
            
        cs = ax.contour(xx, zz, Temperi, 100, levels=isotherms, colors=isot_colors)
        
        # if(instant == instants[0]):
        #     fmt = {}
        #     for level, isot in zip(cs.levels, isotherms):
        #         fmt[level] = str(level) + r'$^{\circ}$C'

        #     ax.clabel(cs, cs.levels, fmt=fmt, inline=True, use_clabeltext=True)

    if(plot_melt == True and prop != 'surface'):

        #TRYING TO PLOT MELT
        # melt = xr.open_dataset(f'{model_path}/_output_melt.nc')
        melt = dataset.melt.T #depleted mantle

        levels_contourf = np.arange(0.02, 0.5, 0.02)
        for i in levels_contourf:
            alpha = 0.9/levels_contourf[-1]*i + 0.1

            cs0 = ax.contourf(xx, zz, melt, levels=[i, i+0.02], colors='xkcd:black', alpha=alpha, zorder=30)


        levels_melt = [0.1, 0.2] #np.arange(0.2, 0.7, 0.4)#[0.2, 0.6]
        cs = ax.contour(xx,
                    zz,
                    melt,
                    levels = levels_melt,
                    colors='xkcd:blue',
                    # cmap = 'inferno',
                    zorder=30,
                    )
        
        
        # incremental_melt = xr.open_dataset(f'{model_path}/_output_incremental_melt.nc')
        incremental_melt = dataset.incremental_melt.T
        scale=1.0e4
        
        levels_incremental_melt = [5.0e-10, 1.0e-6]

        ax.contourf(xx,
                    zz,
                    incremental_melt/scale,#, 'dashdot', 'dashed'],
                    levels = levels_incremental_melt,
                    colors='xkcd:bright pink',#['xkcd:bright pink', 'xkcd:pink'],
                    # colors=colors,
                    alpha=0.4,
                    zorder=30)
        
        ax.contour(xx,
                    zz,
                    incremental_melt/scale,
                    linestyles=['solid'],#, 'dashdot', 'dashed'],
                    levels = [5.0E-10,1.0E-8],
                    colors='xkcd:bright pink',#['xkcd:bright pink', 'xkcd:pink'],
                    # colors=colors,
                    alpha=1.0,
                    zorder=30)
        
    #dealing with special data
    if(prop == 'lithology'):
        data = dataset['strain']
        
    elif(prop == 'temperature_anomaly'):
        #removing horizontal mean temperature
        A = dataset['temperature']
        B = A.T #shape: (Nz, Nx)
        C = np.mean(B, axis=1) #shape: 151==Nz
        D = (B.T - C) #B.T (Nx,Nz) para conseguir subtrair C
        data = D
        
    elif(prop == 'surface'):
        # print('Dealing with data')
        # topo_from_density = True
        # topo_from_density = False
        
        if(topo_from_density == True):
            Rhoi = dataset.density.T
            interfaces=[2900, 3365]
            ##Extract layer topography
            z = np.linspace(Lz/1000.0, 0, Nz)
            Z = np.linspace(Lz/1000.0, 0, 8001) #zi
            x = np.linspace(Lx/1000.0, 0, Nx)

            topo_interface = _extract_interface(z, Z, Nx, Rhoi, 200.) #200 kg/m3 = air/crust interface
            
            condx = (xi >= 100) & (xi <= 400)
            z_mean = np.mean(topo_interface[condx])
            
            topo_interface -= np.abs(z_mean)
            topo_interface = -1.0*topo_interface

            data = topo_interface
        else:
            # print('entrei')
            condx = (xi >= 100) & (xi <= 600)
            z_mean = np.mean(dataset.surface[condx])/1.0e3 + 40.0

            data = dataset.surface/1.0e3 + 40.0 - np.abs(z_mean) #km + air layer correction
            
            
    elif(prop == 'pressure'):
        data = dataset[prop]/1.0E9 #GPa
        
    else:
        data = dataset[prop] 
        
    if(prop == 'strain_rate' or prop == 'radiogenic_heat' or prop == 'strain_rate' or prop == 'viscosity'): #properties that need a lognorm colorbar
        im = ax.imshow(data.T,
                       cmap = props_cmap[prop],
                       origin='lower',
                       extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40),
                       norm = LogNorm(vmin=val_minmax[0], vmax=val_minmax[1]),
#                        vmin=val_minmax[0], vmax=val_minmax[1],
                       aspect = 'auto')
        
        #creating colorbar
        if(plot_colorbar):
            axins1 = inset_axes(ax,
                                loc='lower right',
                                width="100%",  # respective to parent_bbox width
                                height="100%",  # respective to parent_bbox width
                                bbox_to_anchor=(0.7,#horizontal position respective to parent_bbox or "loc" position
                                                0.3,# vertical position
                                                0.25,# width
                                                0.05),# height
                                bbox_transform=ax.transAxes
                                )

            clb = fig.colorbar(im,
                            cax=axins1,
    #                            ticks=ticks,
                            orientation='horizontal',
                            fraction=0.08,
                            pad=0.2,
                            format=_log_fmt)

            clb.set_label(props_label[prop], fontsize=12)
            clb.ax.tick_params(labelsize=12)
            clb.minorticks_off()
    
    elif (prop == 'density' or prop == 'pressure' or prop == 'temperature' or prop == 'temperature_anomaly'): #properties that need a regular colorbar
        im = ax.imshow(data.T,
                       cmap = props_cmap[prop],
                       origin='lower',
                       extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40),
                       vmin=val_minmax[0], vmax=val_minmax[1],
                       aspect = 'auto')
        
        if(plot_colorbar):
            axins1 = inset_axes(ax,
                                loc='lower right',
                                width="100%",  # respective to parent_bbox width
                                height="100%",  # respective to parent_bbox width
                                bbox_to_anchor=(0.7,#horizontal position respective to parent_bbox or "loc" position
                                                0.3,# vertical position
                                                0.25,# width
                                                0.05),# height
                                bbox_transform=ax.transAxes
                                )
            
    #         ticks = np.linspace(val_minmax[0], val_minmax[1], 6, endpoint=True)

            #precision of colorbar ticks
            if(prop == 'pressure'): 
                fmt = '%.2f'
            else:
                fmt = '%.0f'
                
            clb = fig.colorbar(im,
                            cax=axins1,
    #                            ticks=ticks,
                            orientation='horizontal',
                            fraction=0.08,
                            pad=0.2,
                            format=fmt)

            clb.set_label(props_label[prop], fontsize=12)
            clb.ax.tick_params(labelsize=12)
            clb.minorticks_off()
        
    elif(prop == 'strain'):

        im = ax.imshow(data.T,
                       cmap = props_cmap[prop],
                       origin='lower',
                       extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40, 0 + 40),
                       # norm = LogNorm(vmin=data.min(), vmax=data.max()),
                       # norm = LogNorm(vmin=1.0e18, vmax=1.0e25),
                       vmin = float(data.min()),
                       vmax = float(data.max()),
                       aspect = 'auto')
        
    elif(prop == 'surface'):
        ax.plot(dataset.x/1.0e3, data, alpha = 1, linewidth = 2.0, color = "blueviolet")
        
    elif(prop == 'lithology'): #shaded lithology plot
        cr = 255.
        color_uc = (228. / cr, 156. / cr, 124. / cr)
        color_lc = (240. / cr, 209. / cr, 188. / cr)
        color_lit = (155. / cr, 194. / cr, 155. / cr)
        color_ast = (207. / cr, 226. / cr, 205. / cr)
        
        Rhoi = dataset.density.T
        interfaces=[2900, 3365]
        
        ##Extract layer topography
        z = np.linspace(Lz/1000.0, 0, Nz)
        Z = np.linspace(Lz/1000.0, 0, 8001) #zi
        x = np.linspace(Lx/1000.0, 0, Nx)
            
        topo_interface = _extract_interface(z, Z, Nx, Rhoi, 200.) #200 kg/m3 = air/crust interface
        condx = (xi >= 100) & (xi <= 600)
        z_mean = np.mean(topo_interface[condx])
        topo_interface -= np.abs(z_mean)
        topo_interface = -1.0*topo_interface
        
        ax.contourf(xx,
                    zz,
                    Rhoi,
                    levels = [300., 2750, 2900, 3365, 3900],
                    colors = [color_uc, color_lc, color_lit, color_ast])
        
        im=ax.imshow(data.T,
                     cmap = 'Greys',
                     origin = 'lower',
                     extent = (0, Lx / 1.0E3, -Lz / 1.0E3 + 40,0 + 40),
                     zorder = 50,
                     alpha = 0.2, vmin=-0.5,
                     vmax = 0.7,
                     aspect = 'auto')
        #legend box
        bv1 = inset_axes(ax,
                        loc='lower right',
                        width="100%",  # respective to parent_bbox width
                        height="100%",  # respective to parent_bbox width
                        bbox_to_anchor=(0.9,#horizontal position respective to parent_bbox or "loc" position
                                        0.29,# vertical position
                                        0.065,# width
                                        0.25),# height
                        bbox_transform=ax.transAxes
                        )
        
        A = np.zeros((100, 10))

        A[:25, :] = 2700
        A[25:50, :] = 2800
        A[50:75, :] = 3300
        A[75:100, :] = 3400

        A = A[::-1, :]

        xA = np.linspace(-0.5, 0.9, 10)
        yA = np.linspace(0, 1.5, 100)

        xxA, yyA = np.meshgrid(xA, yA)
        air_threshold = 300
        bv1.contourf(
            xxA,
            yyA,
            A,
            levels=[air_threshold, 2750, 2900, 3365, 3900],
            colors=[color_uc, color_lc, color_lit, color_ast],
            extent=[-0.5, 0.9, 0, 1.5]
        )

        bv1.imshow(
            xxA[::-1, :],
            extent=[-0.5, 0.9, 0, 1.5],
            zorder=100,
            alpha=0.2,
            cmap=plt.get_cmap("Greys"),
            vmin=-0.5,
            vmax=0.9,
            aspect='auto'
        )

        bv1.set_yticklabels([])
        bv1.set_xlabel(r"log$(\varepsilon_{II})$", size=10)
        bv1.tick_params(axis='x', which='major', labelsize=10)
        bv1.set_xticks([-0.5, 0, 0.5])
        bv1.set_yticks([])
        bv1.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            
    
    if(plot_particles == True):
        if(prop != 'surface'):
            # ncores = 12
            # data_x, data_z, data_ID, data_lithology, data_strain = _read_step(model_path, f"step_{int(dataset.step)}_", ncores)
            # ax.scatter(data_x/1000, data_z/1000, 2, c='xkcd:black', marker='.', zorder=30)

            data_x, data_z, data_ID, data_lithology, data_strain = [], [], [], [], []
            for i in range(ncores):
                # aux_x, aux_z, aux_ID, aux_lithology, aux_strain = np.loadtxt(f"step_{int(dataset.step)}_{str(i)}.txt", unpack=True, comments="P")
                try:
                    aux_x, aux_z, aux_ID, aux_lithology, aux_strain = np.loadtxt(f"step_{int(dataset.step)}_{str(i)}.txt", unpack=True, comments="P")
                except:
                    filepath = f"step_{int(dataset.step)}_{str(i)}.txt"
                    print(f"didnt read file {filepath}\n")
                    continue
                data_x = np.append(data_x, aux_x)
                data_z = np.append(data_z, aux_z)
                data_ID = np.append(data_ID, aux_ID)
                data_lithology = np.append(data_lithology, aux_lithology)
                data_strain = np.append(data_strain, aux_strain)

            cond_litho = (data_lithology >= 1) & (data_lithology <= 8)
            # cond_mb = data_lithology == 1
            cond_ast = data_lithology == 0

            # if(prop=='lithology'):
            #     color_litho = 'xkcd:bright pink'
            #     color_ast = 'xkcd:black'
            # else:
            #     color_litho = 'xkcd:bright green'
            #     color_ast = 'xkcd:black'

            color_litho = 'xkcd:black'
            color_mb = 'xkcd:neon green'
            # color_mb = 'xkcd:black'
            color_ast = 'xkcd:bright pink'

            ax.plot(data_x[cond_litho][::step_plot]/1000, data_z[cond_litho][::step_plot]/1000+40, particle_marker, color=color_litho, markersize=particle_size, alpha=1.0, zorder=30)

            # ax.plot(data_x[cond_mb][::step_plot]/1000, data_z[cond_mb][::step_plot]/1000+40, particle_marker, color=color_mb, markersize=particle_size, alpha=1.0, zorder=30)
            # ax.plot(data_x[cond_mb][::step_plot]/1000, data_z[cond_mb][::step_plot]/1000+40, particle_marker, color=color_mb, markersize=particle_size*10, alpha=1.0, zorder=30)

            ax.plot(data_x[cond_ast][::step_plot]/1000, data_z[cond_ast][::step_plot]/1000+40, particle_marker, color=color_ast, markersize=particle_size, alpha=1.0, zorder=30)
            
        # else:
        #     print('Error: You cannot print particles in the Surface plot!')
        #     return()
    
    #Filling above topographic surface
    if(prop != 'surface'):
        # topo_from_density = True
        # topo_from_density = False
        if(topo_from_density==True):
            Rhoi = dataset.density.T
            # interfaces=[2900, 3365]
            # ##Extract layer topography
            z = np.linspace(Lz/1000.0, 0, Nz)
            Z = np.linspace(Lz/1000.0, 0, 8001) #zi
            x = np.linspace(Lx/1000.0, 0, Nx)

            topo_interface = _extract_interface(z, Z, Nx, Rhoi, 200.) #200 kg/m3 = air/crust interface
            # condx = (xi >= 100) & (xi <= 600)
            # z_mean = np.mean(topo_interface[condx])
            # topo_interface -= np.abs(z_mean)
            topo_interface = -1.0*topo_interface + 40.0
        else:
            topo_interface = dataset.surface/1.0e3 + 40.0
            
        xaux = xx[0]
        condaux = (xaux>xlims[0]) & (xaux<xlims[1])
        xaux = xaux[condaux]

        xaux[0] += 2
        xaux[-1] -= 2

        ax.fill_between(xaux, topo_interface[condaux], ylims[-1]-0.8, color='white', alpha=1.0, zorder=51)
        
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel("Distance (km)", fontsize = label_size)
        ax.set_ylabel("Depth (km)", fontsize = label_size)
        
    else:
        ax.grid('-k', alpha=0.7)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_xlabel("Distance (km)", fontsize = label_size)
        ax.set_ylabel("Topography (km)", fontsize = label_size)
        
    if (save_frames == True):
        fig_name = f"{output_path}/{model_name}_{prop}"

        if(plot_melt==True):
                # fig_name = f"{output_path}/{model_name}_{prop}_MeltFrac_{melt_method}_{str(int(dataset.step)).zfill(6)}.png"
                fig_name = f"{fig_name}_MeltFrac"
        
        if(plot_particles == True):
                fig_name = f"{fig_name}_particles"
        
        fig_name = f"{fig_name}_{str(int(dataset.step)).zfill(6)}.png"
        # fig_name = f"{fig_name}_onlymb_{str(int(dataset.step)).zfill(6)}.png"


        plt.savefig(fig_name, dpi=400)
        
    plt.close('all')

    del fig
    del ax
    del dataset
    del data
    gc.collect()
        