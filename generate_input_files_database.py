"""
This example simulates the evolution of divergent margins, taking into account the plastic rheology and the sin-rift geodynamics
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
import xarray as xr

path = os.getcwd().split('/')
machine_path = '/'+path[1]+'/'+path[2]

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


label_size=18
plt.rc('xtick', labelsize=label_size)
plt.rc('ytick', labelsize=label_size)

scenario_infos = ['SCENARIO INFOS:']
scenario_infos.append(' ')
scenario_infos.append('Name: ' + path[-1])

#Setting the kind of tectonic scenario
# scenario_kind = 'rifting'
# scenario_kind = 'stab'
# scenario_kind = 'accordion'
# scenario_kind = 'accordion_lit_hetero'
# scenario_kind = 'accordion_keel'
scenario_kind = 'stab_keel'
# scenario_kind = 'quiescence'


experiemnts = {'rifting': 'Rifting experiment',
               'stab': 'LAB (1300 oC) stability',
               'accordion': 'Accordion',
               'accordion_lit_hetero': 'Accordion with central weak lithospheric mantle heterogeneity',
               'accordion_keel': 'Accordion with central cratonic keel',
               'stab_keel': 'LAB (1300 oC) stability with central cratonic keel',
               'quiescence': 'Quiescence - central mobile belt',
               }

ncores=20

print('Scenario kind: ' + experiemnts[scenario_kind])
scenario_infos.append('Scenario kind: ' + experiemnts[scenario_kind])

print('N cores: '+str(ncores))
scenario_infos.append('N cores for aguia: '+str(ncores))

#Main parameters used to construct param .txt that changes accordind to
#tectonic regime

if(scenario_kind == 'rifting'):
    #Rheological and Thermal parameters
    # Clc = 1.0
    Clc = 10.0

    Cseed = 0.1
    
    DeltaT = 0
    # DeltaT = 290 # oC
    Hast = 7.38e-12 #Turccote book #original is 0.0
    preset = True
    # preset = False

    selection_in_preset = True
    # selection_in_preset = False

    # keel_center = True
    keel_center = False

    # extra_fragil = True
    extra_fragil = False

    # mean_litho = True
    mean_litho = False

    # high_kappa_in_asthenosphere = True
    high_kappa_in_asthenosphere = False #default

    # scenario = '/Doutorado/cenarios/mandyoc/stable/stable_PT200_rheol19_c1250_C1/'

    # scenario = '/Doutorado/cenarios/mandyoc/stable/lit80km/stable_PT200_rheol19_c1250_C1_HprodAst/'
    # scenario = '/Doutorado/cenarios/mandyoc/stable/lit80km/stable_PT290_rheol19_c1250_C1_HprodAst/'
    # scenario = '/Doutorado/cenarios/mandyoc/stable/lit80km/stable_PT350_rheol19_c1250_C1_HprodAst/'
    # scenario = '/Doutorado/cenarios/mandyoc/stable/lit80km/stable_PT400_rheol19_c1250_C1_HprodAst/'
    
    scenario = '/Doutorado/cenarios/mandyoc/stable/lit150km/stable_DT200_rheol19_c1250_C1_HprodAst_Hlit150km/'
    # scenario = '/Doutorado/cenarios/mandyoc/stable/lit150km/stable_DT290_rheol19_c1250_C1_HprodAst_Hlit150km/'
    # scenario = '/Doutorado/cenarios/mandyoc/stable/lit150km/stable_DT350_rheol19_c1250_C1_HprodAst_Hlit150km/'

    # scenario = '/Doutorado/cenarios/mandyoc/keel/stable_DT200_keel_HprodAst/'

    #Convergence criteria
    denok                            = 1.0e-15
    particles_per_element            = 100

    #Surface constrains
    sp_surface_tracking              = True
    sp_surface_processes             = False
    # sp_surface_processes             = True
    
    #time constrains 
    time_max                         = 40.0e6
    # time_max                         = 200.0e6
    step_print                       = 100

    #External inputs: bc velocity, velocity field, precipitation and
    #climate change
    # variable_bcv                     = True
    variable_bcv                     = False
    velocity_from_ascii              = True
    

    if(sp_surface_processes == True):
        precipitation_profile_from_ascii = True #False
        climate_change_from_ascii        = True #False
    else:
        precipitation_profile_from_ascii = False
        climate_change_from_ascii        = False

    # 
    #step files
    print_step_files                 = True

    #velocity bc
    top_normal_velocity                 = 'fixed'         # ok
    top_tangential_velocity             = 'free '         # ok
    bot_normal_velocity                 = 'fixed'         # ok
    bot_tangential_velocity             = 'free '         # ok
    left_normal_velocity                = 'fixed'         # ok
    left_tangential_velocity            = 'free '         # ok
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
    left_temperature                    = 'fixed'          # ok
    right_temperature                   = 'fixed'          # ok

    ##################################################################
    # total model horizontal extent (m)
    Lx = 1000 * 1.0e3
    # total model vertical extent (m)
    Lz = 300 * 1.0e3 #400 * 1.0e3
    # Lz = 440 * 1.0e3
    # number of points in horizontal direction
    # Nx = 501 #
    # Nx = 801
    Nx = 1001
    # number of points in vertical direction
    # Nz = 151  #
    # Nz = 111 #Belon
    # Nz = 351
    Nz = 301

    # thickness of sticky air layer (m)
    thickness_sa = 40 * 1.0e3
    # thickness of upper crust (m)
    thickness_upper_crust = 20 * 1.0e3
    # thickness of lower crust (m)
    thickness_lower_crust = 15 * 1.0e3
    # total thickness of lithosphere (m)
    # thickness_litho = 80 * 1.0e3
    thickness_litho = 150 * 1.0e3
    # seed depth bellow base of lower crust (m)
    seed_depth = 3 * 1.0e3 #9 * 1.0e3 #original

    print('Important scale factors (C):')
    print('C lower crust: '+str(Clc))

    scenario_infos.append(' ')
    scenario_infos.append('Important scale factors (C):')
    scenario_infos.append('C lower crust: '+str(Clc))

    scenario_infos.append(' ')
    print('Preset of initial temperature field: ' + str(preset))
    scenario_infos.append('Preset of initial temperature field: '+str(preset))
    
    if(preset == True):
        print('Selection in preset: ' + str(selection_in_preset))
        scenario_infos.append('Selection in preset: ' + str(selection_in_preset))

        print('Use horizontal mean of temperature from preset in lithosphere: ' + str(mean_litho))
        scenario_infos.append('Use horizontal mean of temperature from preset in lithosphere: ' + str(mean_litho))

    print('High kappa in asthenosphere: ' + str(high_kappa_in_asthenosphere))
    scenario_infos.append('High kappa in asthenosphere: ' + str(high_kappa_in_asthenosphere))

    print('Seed extra fragil: ' + str(extra_fragil))
    scenario_infos.append('Seed extra fragil: ' + str(extra_fragil))
    
    print('Surface process: '+str(sp_surface_processes))
    scenario_infos.append('Surface process: '+str(sp_surface_processes))

    print('Velocity field: '+str(velocity_from_ascii))
    scenario_infos.append('Velocity field: '+str(velocity_from_ascii))

    print('Variable velocity field: '+str(variable_bcv))
    scenario_infos.append('Variable velocity field: '+str(variable_bcv))

    print('Climate change: '+str(climate_change_from_ascii))
    scenario_infos.append('Climate change: '+str(climate_change_from_ascii))

elif(scenario_kind == 'stab'):
    #Rheological and Thermal parameters
    Clc = 10.0
    Cseed = 0.1

    
    print('Important scale factors (C):')
    print('C lower crust: '+str(Clc))
    scenario_infos.append(' ')
    scenario_infos.append('Important scale factors (C):')
    scenario_infos.append('C lower crust: '+str(Clc))

    # DeltaT = 0
    DeltaT = 200 #oC incrase in mantle potential temperature
    # DeltaT = 290
    # DeltaT = 350
    # DeltaT = 500
    # DeltaT = 600
    # DeltaT = 700
    # DeltaT = 800

    Hast = 7.38e-12 #Turccote book #original is 0.0

    preset = False
    # keel_center = True
    keel_center = False
    extra_fragil = False

    # high_kappa_in_asthenosphere = True
    high_kappa_in_asthenosphere = False #default
    
    #Convergence criteria
    # denok                            = 1.0e-11
    denok                            = 1.0e-15

    particles_per_element            = 100
    
    #Surface constrains
    sp_surface_tracking              = True
    sp_surface_processes             = False
    
    #time constrains  
    time_max                         = 1.0e9 
    step_print                       = 1000
    
    #External inputs: bc velocity, velocity field, precipitation and
    #climate change
    variable_bcv                     = False
    velocity_from_ascii              = False
    
    
    if(sp_surface_processes == True):
        precipitation_profile_from_ascii = True #False
        climate_change_from_ascii        = True #False
    else:
        precipitation_profile_from_ascii = False
        climate_change_from_ascii        = False

    
    #step files
    print_step_files                 = True

    #velocity bc
    top_normal_velocity                 = 'fixed'         # ok
    top_tangential_velocity             = 'free '         # ok
    bot_normal_velocity                 = 'fixed'         # ok
    bot_tangential_velocity             = 'free '         # ok
    left_normal_velocity                = 'fixed'         # ok
    left_tangential_velocity            = 'free '         # ok
    right_normal_velocity               = 'fixed'         # ok
    right_tangential_velocity           = 'free'         # ok

    # periodic_boundary = True
    periodic_boundary = False
    
    if(periodic_boundary == True):
        left_normal_velocity                = 'free'         # ok
        left_tangential_velocity            = 'free'         # ok
        right_normal_velocity               = 'free'         # ok
        right_tangential_velocity           = 'free'         # ok

    #temperature bc
    top_temperature                     = 'fixed'         # ok
    bot_temperature                     = 'fixed'         # ok
    left_temperature                    = 'free'          # ok
    right_temperature                   = 'free'          # ok

    ##################################################################
    # total model horizontal extent (m)
    Lx = 1600 * 1.0e3
    # total model vertical extent (m)
    Lz = 700 * 1.0e3 #400 * 1.0e3
    # number of points in horizontal direction
    Nx = 161 #401 # #801 #1601
    # number of points in vertical direction
    Nz = 71 #176 #71 #176 #351 #71 #301 #401

    # thickness of sticky air layer (m)
    thickness_sa = 40 * 1.0e3
    # thickness of upper crust (m)
    thickness_upper_crust = 20 * 1.0e3
    # thickness of lower crust (m)
    thickness_lower_crust = 15 * 1.0e3
    # total thickness of lithosphere (m)
    # thickness_litho = 80 * 1.0e3
    thickness_litho = 150 * 1.0e3
    # seed depth bellow base of lower crust (m)
    seed_depth = 3 * 1.0e3 #9 * 1.0e3 #original

    scenario_infos.append(' ')
    print('Preset of initial temperature field: ' + str(preset))
    scenario_infos.append('Preset of initial temperature field: '+str(preset))
    print('High kappa in asthenosphere: ' + str(high_kappa_in_asthenosphere))
    scenario_infos.append('High kappa in asthenosphere: ' + str(high_kappa_in_asthenosphere))
    print('Surface process: '+str(sp_surface_processes))
    scenario_infos.append('Surface process: '+str(sp_surface_processes))
    print('Velocity field: '+str(velocity_from_ascii))
    scenario_infos.append('Velocity field: '+str(velocity_from_ascii))
    print('Climate change: '+str(climate_change_from_ascii))
    print('Variable velocity field: '+str(variable_bcv))
    scenario_infos.append('Variable velocity field: '+str(variable_bcv))
    scenario_infos.append('Climate change: '+str(climate_change_from_ascii))
    print('Periodic Boundary: '+str(periodic_boundary))
    scenario_infos.append('Periodic Boundary: '+str(periodic_boundary))

elif(scenario_kind == 'accordion'):
    #Rheological and Thermal parameters
    # Clc = 1.0
    Clc = 10.0
    # Clc = 40.0

    Cseed = 0.1

    DeltaT = 0
    # DeltaT = 290 # oC
    Hast = 7.38e-12 #Turccote book #original is 0.0
    
    preset = True
    # preset = False

    # keel_center = True
    keel_center = False

    selection_in_preset = False
    # selection_in_preset = True

    # extra_fragil = True
    extra_fragil = False

    # mean_litho = True
    mean_litho = False

    # high_kappa_in_asthenosphere = True
    high_kappa_in_asthenosphere = False #default

    # scenario = 'stable/stable_PT200_rheol19_c1250_C1/'

    # scenario = 'stable/lit80km/stable_PT200_rheol19_c1250_C1_HprodAst'
    # scenario = 'stable/lit80km/stable_PT290_rheol19_c1250_C1_HprodAst'
    # scenario = 'stable/lit80km/stable_PT350_rheol19_c1250_C1_HprodAst'
    # scenario = 'stable/lit80km/stable_PT400_rheol19_c1250_C1_HprodAst'
    
    # scenario = 'stable/lit150km/stable_DT200_rheol19_c1250_C1_HprodAst_Hlit150km'
    # scenario = 'stable/lit150km/stable_DT290_rheol19_c1250_C1_HprodAst_Hlit150km'
    scenario = 'stable/lit150km/stable_DT350_rheol19_c1250_C1_HprodAst_Hlit150km'

    # scenario = 'keel/stable_DT200_keel_HprodAst'

    #Convergence criteria
    denok                            = 1.0e-15
    particles_per_element            = 100

    #Surface constrains
    sp_surface_tracking              = True

    # sp_surface_processes             = True
    sp_surface_processes             = False

    #time constrains 
    time_max                         = 120.0e6
    # time_max                         = 200.0e6

    step_print                       = 100

    #External inputs: bc velocity, velocity field, precipitation and
    #climate change
    variable_bcv                     = True
    velocity_from_ascii              = True

    if(sp_surface_processes == True):
        precipitation_profile_from_ascii = True #False
        climate_change_from_ascii        = True #False
    else:
        precipitation_profile_from_ascii = False
        climate_change_from_ascii        = False

    # 
    #step files
    print_step_files                 = True

    #velocity bc
    top_normal_velocity                 = 'fixed'         # ok
    top_tangential_velocity             = 'free '         # ok
    bot_normal_velocity                 = 'fixed'         # ok
    bot_tangential_velocity             = 'free '         # ok
    left_normal_velocity                = 'fixed'         # ok
    left_tangential_velocity            = 'free '         # ok
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
    left_temperature                    = 'fixed'          # ok
    right_temperature                   = 'fixed'          # ok

    ##################################################################
    bellon = False
    # bellon = True

    if(bellon == False):
        # total model horizontal extent (m)
        Lx = 1600 * 1.0e3
        # total model vertical extent (m)
        Lz = 700 * 1.0e3 #400 * 1.0e3
        # Lz = 440 * 1.0e3
        # number of points in horizontal direction
        Nx = 401 #
        # Nx = 801
        # number of points in vertical direction
        Nz = 176  #
        # Nz = 351

        # thickness of sticky air layer (m)
        thickness_sa = 40 * 1.0e3
        # thickness of upper crust (m)
        thickness_upper_crust = 20 * 1.0e3
        # thickness of lower crust (m)
        thickness_lower_crust = 15 * 1.0e3
        # total thickness of lithosphere (m)
        # thickness_litho = 80 * 1.0e3
        thickness_litho = 150 * 1.0e3
        # thickness_litho = 180 * 1.0e3
        # thickness_litho = 210 * 1.0e3
        # seed depth bellow base of lower crust (m)
        seed_depth = 3 * 1.0e3 #9 * 1.0e3 #original

    else: #Bellon, Silva and Sacek
        preset = False
        time_max = 150.0e6

        # total model horizontal extent (m)
        Lx = 1600 * 1.0e3
        # total model vertical extent (m)
        Lz = 440 * 1.0e3
        # number of points in horizontal direction
        Nx = 401 
        # number of points in vertical direction
        Nz = 111

        # thickness of sticky air layer (m)
        thickness_sa = 40 * 1.0e3
        # thickness of upper crust (m)
        thickness_upper_crust = 20 * 1.0e3
        # thickness of lower crust (m)
        thickness_lower_crust = 15 * 1.0e3
        # total thickness of lithosphere (m)
        thickness_litho = 180 * 1.0e3
        # seed depth bellow base of lower crust (m)
        seed_depth = 3 * 1.0e3 #9 * 1.0e3 #original

    print('Important scale factors (C):')
    print('C lower crust: '+str(Clc))

    scenario_infos.append(' ')
    scenario_infos.append('Important scale factors (C):')
    scenario_infos.append('C lower crust: '+str(Clc))

    scenario_infos.append(' ')
    print('Preset of initial temperature field: ' + str(preset))
    scenario_infos.append('Preset of initial temperature field: '+str(preset))

    print('Seed extra fragil: ' + str(extra_fragil))
    scenario_infos.append('Seed extra fragil: ' + str(extra_fragil))

    print('Use horizontal mean of temperature from preset in lithosphere: ' + str(mean_litho))
    scenario_infos.append('Use horizontal mean of temperature from preset in lithosphere: ' + str(mean_litho))

    print('Surface process: '+str(sp_surface_processes))
    scenario_infos.append('Surface process: '+str(sp_surface_processes))

    print('Velocity field: '+str(velocity_from_ascii))
    scenario_infos.append('Velocity field: '+str(velocity_from_ascii))

    print('Variable velocity field: '+str(variable_bcv))
    scenario_infos.append('Variable velocity field: '+str(variable_bcv))

    print('Climate change: '+str(climate_change_from_ascii))
    scenario_infos.append('Climate change: '+str(climate_change_from_ascii))

elif(scenario_kind == 'accordion_lit_hetero'):
    #Rheological and Thermal parameters
    # Clc = 1.0
    Clc = 10.0
    # Clc = 40.0

    Clitl = 1.0 #lateral
    Clitc = 0.1 #central
    Cseed = 0.1

    print('Important scale factors (C):')
    print('C lower crust: '+str(Clc))
    print('C lateral lithosphere: '+str(Clitl))
    print('C central lithosphere: '+str(Clitc))
    scenario_infos.append(' ')
    scenario_infos.append('Important scale factors (C):')
    scenario_infos.append('C lower crust: '+str(Clc))
    scenario_infos.append('C lateral lithosphere: '+str(Clitl))
    scenario_infos.append('C central lithosphere: '+str(Clitc))

    DeltaT = 0
    # DeltaT = 290 # oC
    
    Hast = 7.38e-12 #Turccote book #original is 0.0
    
    # preset = True
    preset = False
    
    # keel_center = True
    keel_center = False
    
    # preset = False
    
    # extra_fragil = True
    extra_fragil = False
    
    mean_litho = True
    # mean_litho = False

    # high_kappa_in_asthenosphere = True
    high_kappa_in_asthenosphere = False #default
    
    scenario_infos.append(' ')
    print('Preset of initial temperature field: ' + str(preset))
    scenario_infos.append('Preset of initial temperature field: '+str(preset))

    print('Seed extra fragil: ' + str(extra_fragil))
    scenario_infos.append('Seed extra fragil: ' + str(extra_fragil))

    print('Use horizontal mean of temperature from preset in lithosphere: ' + str(mean_litho))
    scenario_infos.append('Use horizontal mean of temperature from preset in lithosphere: ' + str(mean_litho))

    # scenario = 'stable/stable_PT200_rheol19_c1250_C1/'

    # scenario = 'stable/lit80km/stable_PT200_rheol19_c1250_C1_HprodAst/'
    # scenario = 'stable/lit80km/stable_PT290_rheol19_c1250_C1_HprodAst/'
    scenario = 'stable/lit80km/stable_PT350_rheol19_c1250_C1_HprodAst/'
    # scenario = 'stable/lit80km/stable_PT400_rheol19_c1250_C1_HprodAst/'
    
    # scenario = 'stable/lit150km/stable_DT200_rheol19_c1250_C1_HprodAst_Hlit150km/'
    # scenario = 'stable/lit150km/stable_DT290_rheol19_c1250_C1_HprodAst_Hlit150km/'
    # scenario = 'stable/lit150km/stable_DT350_rheol19_c1250_C1_HprodAst_Hlit150km/'

    # scenario = 'keel/stable_DT200_keel_HprodAst/'

    #Convergence criteria
    denok                            = 1.0e-13
    particles_per_element            = 100

    #Surface constrains
    sp_surface_tracking              = True
    sp_surface_processes             = False 
    print('Surface process: '+str(sp_surface_processes))
    scenario_infos.append('Surface process: '+str(sp_surface_processes))

    #time constrains 
    time_max                         = 120.0e6
    # time_max                         = 200.0e6
    step_print                       = 100

    #External inputs: bc velocity, velocity field, precipitation and
    #climate change
    variable_bcv                     = True
    velocity_from_ascii              = True
    print('Velocity field: '+str(velocity_from_ascii))
    scenario_infos.append('Velocity field: '+str(velocity_from_ascii))
    print('Variable velocity field: '+str(variable_bcv))
    scenario_infos.append('Variable velocity field: '+str(variable_bcv))

    if(sp_surface_processes == True):
        precipitation_profile_from_ascii = True #False
        climate_change_from_ascii        = True #False
    else:
        precipitation_profile_from_ascii = False
        climate_change_from_ascii        = False

    print('Climate change: '+str(climate_change_from_ascii))
    scenario_infos.append('Climate change: '+str(climate_change_from_ascii))
    # 
    #step files
    print_step_files                 = True

    #velocity bc
    top_normal_velocity                 = 'fixed'         # ok
    top_tangential_velocity             = 'free '         # ok
    bot_normal_velocity                 = 'fixed'         # ok
    bot_tangential_velocity             = 'free '         # ok
    left_normal_velocity                = 'fixed'         # ok
    left_tangential_velocity            = 'free '         # ok
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
    left_temperature                    = 'fixed'          # ok
    right_temperature                   = 'fixed'          # ok

    bellon = False
    # bellon = True

    ##################################################################
    # total model horizontal extent (m)
    Lx = 1600 * 1.0e3
    # total model vertical extent (m)
    Lz = 700 * 1.0e3 #400 * 1.0e3
    # Lz = 440 * 1.0e3
    # number of points in horizontal direction
    Nx = 401 #
    # Nx = 801
    # number of points in vertical direction
    Nz = 176  #
    # Nz = 111 #Bellon
    # Nz = 351

    # thickness of sticky air layer (m)
    thickness_sa = 40 * 1.0e3
    # thickness of upper crust (m)
    thickness_upper_crust = 20 * 1.0e3
    # thickness of lower crust (m)
    thickness_lower_crust = 15 * 1.0e3
    # total thickness of lithosphere (m)
    # thickness_litho = 80 * 1.0e3
    thickness_litho = 150 * 1.0e3
    # thickness_litho = 180 * 1.0e3
    # thickness_litho = 210 * 1.0e3
    # seed depth bellow base of lower crust (m)
    seed_depth = 3 * 1.0e3 #9 * 1.0e3 #original

elif(scenario_kind == 'accordion_keel'):
    #Rheological and Thermal parameters
    # Clc = 1.0
    Clc = 10.0
    # Clc = 40.0
    Cseed = 0.1

    print('Important scale factors (C):')
    print('C lower crust: '+str(Clc))
    scenario_infos.append(' ')
    scenario_infos.append('Important scale factors (C):')
    scenario_infos.append('C lower crust: '+str(Clc))

    DeltaT = 290 # oC
    Hast = 7.38e-12 #Turccote book #original is 0.0
    preset = True
    keel_center = False
    # preset = False
    # extra_fragil = True
    extra_fragil = False
    mean_litho = True
    # mean_litho = False
    # high_kappa_in_asthenosphere = True
    high_kappa_in_asthenosphere = False #default

    scenario_infos.append(' ')
    print('Preset of initial temperature field: ' + str(preset))
    scenario_infos.append('Preset of initial temperature field: '+str(preset))

    print('Seed extra fragil: ' + str(extra_fragil))
    scenario_infos.append('Seed extra fragil: ' + str(extra_fragil))

    print('Use horizontal mean of temperature from preset in lithosphere: ' + str(mean_litho))
    scenario_infos.append('Use horizontal mean of temperature from preset in lithosphere: ' + str(mean_litho))

    # scenario = 'keel/stable_PT200_keel_HprodAst/'
    scenario = 'keel/stable_DT290_keel_HprodAst/'

    #Convergence criteria
    denok                            = 1.0e-13
    particles_per_element            = 100

    #Surface constrains
    sp_surface_tracking              = True
    sp_surface_processes             = False 
    print('Surface process: '+str(sp_surface_processes))
    scenario_infos.append('Surface process: '+str(sp_surface_processes))

    #time constrains 
    time_max                         = 120.0e6 
    step_print                       = 100

    #External inputs: bc velocity, velocity field, precipitation and
    #climate change
    variable_bcv                     = True
    velocity_from_ascii              = True
    print('Velocity field: '+str(velocity_from_ascii))
    scenario_infos.append('Velocity field: '+str(velocity_from_ascii))
    print('Variable velocity field: '+str(variable_bcv))
    scenario_infos.append('Variable velocity field: '+str(variable_bcv))

    if(sp_surface_processes == True):
        precipitation_profile_from_ascii = True #False
        climate_change_from_ascii        = True #False
    else:
        precipitation_profile_from_ascii = False
        climate_change_from_ascii        = False

    print('Climate change: '+str(climate_change_from_ascii))
    scenario_infos.append('Climate change: '+str(climate_change_from_ascii))
    # 
    #step files
    print_step_files                 = True

    #velocity bc
    top_normal_velocity                 = 'fixed'         # ok
    top_tangential_velocity             = 'free '         # ok
    bot_normal_velocity                 = 'fixed'         # ok
    bot_tangential_velocity             = 'free '         # ok
    left_normal_velocity                = 'fixed'         # ok
    left_tangential_velocity            = 'free '         # ok
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
    left_temperature                    = 'fixed'          # ok
    right_temperature                   = 'fixed'          # ok

    bellon = False
    # bellon = True

    ##################################################################
    # total model horizontal extent (m)
    Lx = 1600 * 1.0e3
    # total model vertical extent (m)
    Lz = 700 * 1.0e3 #400 * 1.0e3
    # number of points in horizontal direction
    # Nx = 401 #
    Nx = 801
    # number of points in vertical direction
    # Nz = 176  #
    Nz = 351

    # thickness of sticky air layer (m)
    thickness_sa = 40 * 1.0e3
    # thickness of upper crust (m)
    thickness_upper_crust = 20 * 1.0e3
    # thickness of lower crust (m)
    thickness_lower_crust = 15 * 1.0e3
    # total thickness of lithosphere (m)
    thickness_litho = 80 * 1.0e3
    # seed depth bellow base of lower crust (m)
    seed_depth = 3 * 1.0e3 #9 * 1.0e3 #original

elif(scenario_kind == 'stab_keel'):
    #Rheological and Thermal parameters
    Clc = 10.0
    Cseed = 0.1

    print('Important scale factors (C):')
    print('C lower crust: '+str(Clc))
    scenario_infos.append(' ')
    scenario_infos.append('Important scale factors (C):')
    scenario_infos.append('C lower crust: '+str(Clc))

    DeltaT = 200
    # DeltaT = 290
    # DeltaT = 350

    Hast = 7.38e-12 #Turccote book #original is 0.0

    preset = True
    # preset = False

    selection_in_preset = True
    # selection_in_preset = False

    # keel_center = True
    keel_center = False

    # extra_fragil = True
    extra_fragil = False

    # mean_litho = True
    mean_litho = False

    # high_kappa_in_asthenosphere = True
    high_kappa_in_asthenosphere = False #default


    # scenario = '/Doutorado/cenarios/mandyoc/stable/lit80km/stable_PT200_rheol19_c1250_C1_HprodAst/'
    # scenario = '/Doutorado/cenarios/mandyoc/stable/lit80km/stable_PT290_rheol19_c1250_C1_HprodAst/'
    scenario = '/Doutorado/cenarios/mandyoc/stable/lit80km/stable_PT350_rheol19_c1250_C1_HprodAst/'

    scenario_infos.append(' ')
    print('Preset of initial temperature field: ' + str(preset))
    scenario_infos.append('Preset of initial temperature field: '+str(preset))

    #Convergence criteria
    denok                            = 1.0e-11
    particles_per_element            = 100
    
    #Surface constrains
    sp_surface_tracking              = True
    sp_surface_processes             = False

    print('Surface process: '+str(sp_surface_processes))
    scenario_infos.append('Surface process: '+str(sp_surface_processes))

    #time constrains 
    time_max                         = 1.0e9
    step_print                       = 500    
    
    #External inputs: bc velocity, velocity field, precipitation and
    #climate change
    variable_bcv                     = False
    velocity_from_ascii              = False

    print('Velocity field: '+str(velocity_from_ascii))
    scenario_infos.append('Velocity field: '+str(velocity_from_ascii))
    print('Variable velocity field: '+str(variable_bcv))
    scenario_infos.append('Variable velocity field: '+str(variable_bcv))

    if(sp_surface_processes == True):
        precipitation_profile_from_ascii = True #False
        climate_change_from_ascii        = True #False
    else:
        precipitation_profile_from_ascii = False
        climate_change_from_ascii        = False 
    print('Climate change: '+str(climate_change_from_ascii))
    scenario_infos.append('Climate change: '+str(climate_change_from_ascii))

    #step files
    print_step_files                 = True

    #velocity bc
    top_normal_velocity                 = 'fixed'         # ok
    top_tangential_velocity             = 'free '         # ok
    bot_normal_velocity                 = 'fixed'         # ok
    bot_tangential_velocity             = 'free '         # ok
    left_normal_velocity                = 'fixed'         # ok
    left_tangential_velocity            = 'free '         # ok
    right_normal_velocity               = 'fixed'         # ok
    right_tangential_velocity           = 'free'         # ok

    # periodic_boundary = True
    periodic_boundary = False

    print('Periodic Boundary: '+str(periodic_boundary))
    scenario_infos.append('Periodic Boundary: '+str(periodic_boundary))

    if(periodic_boundary == True):
        left_normal_velocity                = 'free'         # ok
        left_tangential_velocity            = 'free '         # ok
        right_normal_velocity               = 'free'         # ok
        right_tangential_velocity           = 'free'         # ok

    #temperature bc
    top_temperature                     = 'fixed'         # ok
    bot_temperature                     = 'fixed'         # ok
    left_temperature                    = 'free'          # ok
    right_temperature                   = 'free'          # ok

    bellon = False
    # bellon = True

    ##################################################################
    # total model horizontal extent (m)
    # Lx = 1600 * 1.0e3
    Lx = 3000 * 1.0e3
    # total model vertical extent (m)
    Lz = 700 * 1.0e3 #400 * 1.0e3
    # number of points in horizontal direction
    # Nx = 161 #401 # #801 #1601
    Nx = 301
    # number of points in vertical direction
    Nz = 71 #176 #71 #176 #351 #71 #301 #401
    
    # thickness of sticky air layer (m)
    thickness_sa = 40 * 1.0e3
    # thickness of upper crust (m)
    thickness_upper_crust = 20 * 1.0e3
    # thickness of lower crust (m)
    thickness_lower_crust = 15 * 1.0e3
    # total thickness of lithosphere (m)
    thickness_litho = 80 * 1.0e3
    # seed depth bellow base of lower crust (m)
    seed_depth = 3 * 1.0e3 #9 * 1.0e3 #original

elif(scenario_kind == 'quiescence'):
    #Rheological and Thermal parameters
    Clc = 100.0
    Cseed = 0.1

    print('Important scale factors (C):')
    print('C lower crust: '+str(Clc))
    scenario_infos.append(' ')
    scenario_infos.append('Important scale factors (C):')
    scenario_infos.append('C lower crust: '+str(Clc))

    # DeltaT = 200
    DeltaT = 0#290
    Hast = 7.38e-12 #Turccote book #original is 0.0
    preset = False
    keel_center = False
    extra_fragil = False
    # high_kappa_in_asthenosphere = True
    high_kappa_in_asthenosphere = False #default
    
    scenario_infos.append(' ')
    print('Preset of initial temperature field: ' + str(preset))
    scenario_infos.append('Preset of initial temperature field: '+str(preset))

    #Convergence criteria
    denok                            = 1.0e-11
    particles_per_element            = 100
    
    #Surface constrains
    sp_surface_tracking              = True
    sp_surface_processes             = False
    print('Surface process: '+str(sp_surface_processes))
    scenario_infos.append('Surface process: '+str(sp_surface_processes))

    #time constrains 
    time_max                         = 300.0e6
    step_print                       = 250    
    
    #External inputs: bc velocity, velocity field, precipitation and
    #climate change
    variable_bcv                     = False
    velocity_from_ascii              = False
    print('Velocity field: '+str(velocity_from_ascii))
    scenario_infos.append('Velocity field: '+str(velocity_from_ascii))
    print('Variable velocity field: '+str(variable_bcv))
    scenario_infos.append('Variable velocity field: '+str(variable_bcv))

    print_step_files                 = True
    
    if(sp_surface_processes == True):
        precipitation_profile_from_ascii = True #False
        climate_change_from_ascii        = True #False
    else:
        precipitation_profile_from_ascii = False
        climate_change_from_ascii        = False 
    print('Climate change: '+str(climate_change_from_ascii))
    scenario_infos.append('Climate change: '+str(climate_change_from_ascii))

    #step files
    print_step_files                 = True

    #velocity bc
    top_normal_velocity                 = 'fixed'         # ok
    top_tangential_velocity             = 'free '         # ok
    bot_normal_velocity                 = 'fixed'         # ok
    bot_tangential_velocity             = 'free '         # ok
    left_normal_velocity                = 'fixed'         # ok
    left_tangential_velocity            = 'free '         # ok
    right_normal_velocity               = 'fixed'         # ok
    right_tangential_velocity           = 'free'         # ok

    # periodic_boundary = True
    periodic_boundary = False
    print('Periodic Boundary: '+str(periodic_boundary))
    scenario_infos.append('Periodic Boundary: '+str(periodic_boundary))

    if(periodic_boundary == True):
        left_normal_velocity                = 'free'         # ok
        left_tangential_velocity            = 'free '         # ok
        right_normal_velocity               = 'free'         # ok
        right_tangential_velocity           = 'free'         # ok

    #temperature bc
    top_temperature                     = 'fixed'         # ok
    bot_temperature                     = 'fixed'         # ok
    left_temperature                    = 'free'          # ok
    right_temperature                   = 'free'          # ok

    bellon = False
    # bellon = True
    
    ##################################################################
    # total model horizontal extent (m)
    Lx = 1600 * 1.0e3
    # total model vertical extent (m)
    Lz = 700 * 1.0e3 #400 * 1.0e3
    # number of points in horizontal direction
    Nx = 161 #401 # #801 #1601
    # number of points in vertical direction
    Nz = 71 #176 #71 #176 #351 #71 #301 #401
    
    # thickness of sticky air layer (m)
    thickness_sa = 40 * 1.0e3
    # thickness of upper crust (m)
    thickness_upper_crust = 15 * 1.0e3
    # thickness of lower crust (m)
    thickness_lower_crust = 15 * 1.0e3
    # total thickness of lithosphere (m)
    thickness_litho = 250 * 1.0e3
    # seed depth bellow base of lower crust (m)
    seed_depth = 3 * 1.0e3 #9 * 1.0e3 #original


x = np.linspace(0, Lx, Nx)
z = np.linspace(Lz, 0, Nz)
X, Z = np.meshgrid(x, z)

print('Domain parameters:')
print('Lx: ' + str(Lx*1.0e-3) + ' km')
print('Lz: ' + str(Lz*1.0e-3) + ' km')
print('Nx: ' + str(Nx))
print('Nz: ' + str(Nz))
print('Resolution dx x dz: ' + str(1.0e-3*Lx/(Nx-1)) + 'x' + str(1.0e-3*Lz/(Nz-1)) + ' km2')
print('Layers thickness:')
print('H air: ' + str(thickness_sa*1.0e-3) + ' km')
print('H upper crust: ' + str(thickness_upper_crust*1.0e-3) + ' km')
print('H lower crust: ' + str(thickness_lower_crust*1.0e-3) + ' km')
print('H mantle lithosphere: ' + str((thickness_litho - thickness_upper_crust - thickness_lower_crust)*1.0e-3) + ' km')
print('H lithosphere: ' + str(thickness_litho*1.0e-3) + ' km')

scenario_infos.append(' ')
scenario_infos.append('Domain parameters:')
scenario_infos.append('Lx: ' + str(Lx*1.0e-3) + ' km')
scenario_infos.append('Lz: ' + str(Lz*1.0e-3) + ' km')
scenario_infos.append('Nx: ' + str(Nx))
scenario_infos.append('Nz: ' + str(Nz))
scenario_infos.append('Resolution dx x dz: ' + str(int(1.0e-3*Lx/(Nx-1))) + 'x' + str(int(1.0e-3*Lz/(Nz-1))) + ' km2')

scenario_infos.append(' ')
scenario_infos.append('Layers thickness:')
scenario_infos.append('H air: ' + str(thickness_sa*1.0e-3) + ' km')
scenario_infos.append('H upper crust: ' + str(thickness_upper_crust*1.0e-3) + ' km')
scenario_infos.append('H lower crust: ' + str(thickness_lower_crust*1.0e-3) + ' km')
scenario_infos.append('H mantle lithosphere: ' + str((thickness_litho - thickness_upper_crust - thickness_lower_crust)*1.0e-3) + ' km')
scenario_infos.append('H lithosphere: ' + str(thickness_litho*1.0e-3) + ' km')
scenario_infos.append(' ')


##############################################################################
# Interfaces (bottom first)
##############################################################################


if(scenario_kind == 'accordion_lit_hetero'):
    interfaces = {
        "litho_LAB": np.ones(Nx) * (thickness_litho + thickness_sa), #lab horizontal
        "litho_HETERO": np.ones(Nx) * (thickness_litho + thickness_sa), #interface entre central e lateral -  interface 
        # "seed_base": np.ones(Nx) * (seed_depth + thickness_lower_crust + thickness_upper_crust + thickness_sa),
        # "seed_top": np.ones(Nx) * (seed_depth + thickness_lower_crust + thickness_upper_crust + thickness_sa),
        "lower_crust": np.ones(Nx) * (thickness_lower_crust + thickness_upper_crust + thickness_sa),
        "upper_crust": np.ones(Nx) * (thickness_upper_crust + thickness_sa),
        "air": np.ones(Nx) * (thickness_sa),
    }

    dx = Lx/(Nx-1)
    M_lit = thickness_litho - (thickness_upper_crust + thickness_lower_crust) #thickness of lithospheric mantle

    Wcenter = 200.0e3 #width of central portion #m
    N_Wcenter = int(Wcenter//dx) #largura em indices

    #thinning for central portion to create the strong mantle lithosphere
    thinning = thickness_litho - M_lit #m
    interfaces['litho_HETERO'][Nx//2 - N_Wcenter//2 : Nx//2 + N_Wcenter//2] = thickness_sa + thinning

    #thickening for central portion to create the weak mantle lithosphere
    # thickening = thickness_sa + M_lit#thickness_upper_crust + thickness_lower_crust + thickness_sa + M_lit #m
    # interfaces['litho_center'][Nx//2 - N_Wcenter//2 : Nx//2 + N_Wcenter//2] = thickness_sa + thinning

else:
    interfaces = {
        "litho": np.ones(Nx) * (thickness_litho + thickness_sa),
        "seed_base": np.ones(Nx) * (seed_depth + thickness_lower_crust + thickness_upper_crust + thickness_sa),
        "seed_top": np.ones(Nx) * (seed_depth + thickness_lower_crust + thickness_upper_crust + thickness_sa),
        "lower_crust": np.ones(Nx) * (thickness_lower_crust + thickness_upper_crust + thickness_sa),
        "upper_crust": np.ones(Nx) * (thickness_upper_crust + thickness_sa),
        "air": np.ones(Nx) * (thickness_sa),
    }

    # seed thickness (m)
    thickness_seed = 6 * 1.0e3
    # seed horizontal position (m)
    # x_seed = 800 * 1.0e3
    x_seed = Lx / 2.0
    # x_seed = Lx / 2.0 + 200.0e3
    # seed: number of points of horizontal extent
    n_seed = 6

    interfaces["seed_base"][
        int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)
    ] = (
        interfaces["seed_base"][
            int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)
        ]
        + thickness_seed // 2
    )
    interfaces["seed_top"][
        int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)
    ] = (
        interfaces["seed_top"][
            int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)
        ]
        - thickness_seed // 2
    )


if(scenario_kind == 'stab_keel' or scenario_kind == 'accordion_keel'):
    dx = Lx/(Nx-1)
    # Lcraton = 600.0e3 #m
    Lcraton = 1200.0e3 #m
    thickening = thickness_litho + 120.e3 #m
    Ncraton = int(Lcraton//dx) #largura em indices
    interfaces['litho'][Nx//2 - Ncraton//2 : Nx//2 + Ncraton//2] = thickness_sa + thickening

if(scenario_kind == 'accordion' or scenario_kind == 'rifting'):
    if(extra_fragil == True):
        Cseed = 0.01
        dx = Lx/(Nx-1)
        Lfragil = 12.0e3 #m
        # thinning = thickness_litho - 45.0e3
        thinning = thickness_litho - 20.0e3
        Nfragil = int(Lfragil//dx) #largura em indices
        interfaces['litho'][Nx//2 - Nfragil//2 : Nx//2 + Nfragil//2] = thickness_sa + thinning

if(scenario_kind == 'quiescence'):
    dx = Lx/(Nx-1)
    WTL = 400.0e3 #m
    thinning = thickness_litho - 150.e3 #m
    N_WTL = int(WTL//dx) #largura em indices
    interfaces['litho'][Nx//2 - N_WTL//2 : Nx//2 + N_WTL//2] = thickness_sa + thinning


Huc = 2.5e-6 / 2700.0 #9.259E-10
Hlc = 0.8e-6 / 2800.0 #2.85E-10

# Create the interface file

if(scenario_kind == 'accordion_lit_hetero'):
    with open("interfaces.txt", "w") as f:
        rheology_mlit = 'dry' #rheology of lithospheric mantle: dry olivine or wet olivine
        
        if(rheology_mlit == 'dry'):
            layer_properties = f"""
                C   1.0       {Clitl}    {Clitc}     {Clc}       1.0         1.0
                rho 3378.0    3354.0     3354.0      2800.0      2700.0      1.0
                H   {Hast}    9.0e-12    9.0e-12     {Hlc}       {Huc}       0.0
                A   1.393e-14 2.4168e-15 2.4168e-15  8.574e-28   8.574e-28   1.0e-18
                n   3.0       3.5        3.5         4.0         4.0         1.0
                Q   429.0e3   540.0e3    540.0e3     222.0e3     222.0e3     0.0
                V   15.0e-6   25.0e-6    25.0e-6     0.0         0.0         0.0
            """

        if(rheology_mlit == 'wet'):
            layer_properties = f"""
                C   1.0       {Clitl}    {Clitc}    {Clc}       1.0         1.0
                rho 3378.0    3354.0     3354.0     2800.0      2700.0      1.0
                H   {Hast}    9.0e-12    9.0e-12    {Hlc}       {Huc}       0.0
                A   1.393e-14 1.393e-14  1.393e-14  8.574e-28   8.574e-28   1.0e-18
                n   3.0       3.0        3.0        4.0         4.0         1.0
                Q   429.0e3   429.0e3    429.0e3    222.0e3     222.0e3     0.0
                V   15.0e-6   15.0e-6    15.0e-6    0.0         0.0         0.0
            """

        for line in layer_properties.split("\n"):
            line = line.strip()
            if len(line):
                f.write(" ".join(line.split()) + "\n")

        # layer interfaces
        data = -1 * np.array(tuple(interfaces.values())).T
        np.savetxt(f, data, fmt="%.1f")

else: 
    with open("interfaces.txt", "w") as f:
        rheology_mlit = 'dry' #rheology of lithospheric mantle: dry olivine or wet olivine
        
        if(rheology_mlit == 'dry'):
            layer_properties = f"""
                C   1.0       1.0        {Cseed}    1.0        {Clc}       1.0         1.0
                rho 3378.0    3354.0     3354.0     3354.0     2800.0      2700.0      1.0
                H   {Hast}    9.0e-12    9.0e-12    9.0e-12    {Hlc}       {Huc}       0.0
                A   1.393e-14 2.4168e-15 2.4168e-15 2.4168e-15 8.574e-28   8.574e-28   1.0e-18
                n   3.0       3.5        3.5        3.5        4.0         4.0         1.0
                Q   429.0e3   540.0e3    540.0e3    540.0e3    222.0e3     222.0e3     0.0
                V   15.0e-6   25.0e-6    25.0e-6    25.0e-6    0.0         0.0         0.0
            """

        if(rheology_mlit == 'wet'):
            layer_properties = f"""
                C   1.0       5.0        {Cseed}    5.0        {Clc}       1.0         1.0
                rho 3378.0    3354.0     3354.0     3354.0     2800.0      2700.0      1.0
                H   {Hast}    9.0e-12    9.0e-12    9.0e-12    {Hlc}       {Huc}       0.0
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
    print(label, "(size): ", np.size(layer))
    ax.plot(x/1.0E3, (-layer + thickness_sa)/1.0E3, label=f"{label}")

ax.set_xlim([0, Lx/1.0E3])
ax.set_ylim([Lz/1.0E3, 0])
ax.set_xlabel("x [km]", fontsize=label_size)
ax.set_ylabel("Depth [km]", fontsize=label_size)

ax.set_yticks(np.arange(-Lz / 1e3, 1 / 1e3, 50))
ax.set_xlim([0, Lx/1000])
ax.set_ylim([(-Lz + thickness_sa) / 1e3, 0 + thickness_sa / 1e3])

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
denok                               = {denok}       # default is 1.0E-4
particles_per_element               = {particles_per_element}          # default is 81
particles_perturb_factor            = 0.7           # default is 0.5 [values are between 0 and 1]
rtol                                = 1.0e-7        # the absolute size of the residual norm (relevant only for iterative methods), default is 1.0E-5
RK4                                 = Euler         # default is Euler [Euler/Runge-Kutta]
Xi_min                              = 1.0e-7       # default is 1.0E-14
random_initial_strain               = 0.2           # default is 0.0
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
sp_surface_processes                = {sp_surface_processes}         # default is False [True/False]
sp_dt                               = 1.0e5        # default is 0.0
sp_d_c                              = 1.0          # default is 0.0
plot_sediment                       = False         # default is False [True/False]
a2l                                 = True          # default is True [True/False]
free_surface_stab                   = True          # default is True [True/False]
theta_FSSA                          = 0.5           # default is 0.5 (only relevant when free_surface_stab = True)
# Time constrains
step_max                            = 400000          # Maximum time-step of the simulation
time_max                            = {time_max}  #1.0e9     # Maximum time of the simulation [years]
dt_max                              = 5.0e3      # Maximum time between steps of the simulation [years]
step_print                          = {step_print} #500            # Make file every <step_print>
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
variable_bcv                        = {variable_bcv} #False         # default is False [True/False]
temperature_from_ascii              = True         # default is False [True/False]
velocity_from_ascii                 = {velocity_from_ascii} #False      # default is False [True/False]
binary_output                       = False         # default is False [True/False]
sticky_blanket_air                  = True         # default is False [True/False]
precipitation_profile_from_ascii    = {precipitation_profile_from_ascii}         # default is False [True/False]
climate_change_from_ascii           = {climate_change_from_ascii}         # default is False [True/False]
print_step_files                    = {print_step_files}          # default is True [True/False]
checkered                           = False         # Print one element in the print_step_files (default is False [True/False])
sp_mode                             = 5             # default is 1 [0/1/2]
geoq                                = on            # ok
geoq_fac                            = 100.0           # ok
# Physical parameters
temperature_difference              = 1500.         # ok
thermal_expansion_coefficient       = 3.28e-5       # ok
thermal_diffusivity_coefficient     = 1.0e-6 #0.75e-6       #default is 1.0e-6        # ok
gravity_acceleration                = 10.0          # ok
density_mantle                      = 3300.         # ok
external_heat                       = 0.0e-12       # ok
heat_capacity                       = 1250         # ok #default is 1250
non_linear_method                   = on            # ok
adiabatic_component                 = on            # ok
radiogenic_component                = on            # ok
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
rheology_model                      = 19             # ok
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

print('Initial temperature field setup:')
scenario_infos.append('Initial temperature field setup:')

if(preset == False):
    T = 1300 * (z - thickness_sa) / (thickness_litho)  # Temperature
    # T = 1300 * (z - thickness_sa) / (130*1.0E3)  # Temperature of 1300 isotherm bellow the lithosphere

    ccapacity = 1250*1.0 #937.5=75% #J/kg/K? #DEFAULT.
    # ccapacity = 1250*0.75 #937.5=75% #J/kg/K?
    # ccapacity = 750
    print('Increase in mantle basal temperature (Ta): '+str(DeltaT)+' oC')
    scenario_infos.append('Increase in mantle basal temperature (Ta): '+str(DeltaT)+' oC')

    TP = 1262 #mantle potential temperature
    # TP = 1350
    # TP = 1400
    # TP = 1450
    print('Assumed mantle Potential Temperature: '+str(TP)+' oC')
    scenario_infos.append('Assumed mantle Potential Temperature: '+str(TP)+' oC')

    Ta = (TP / np.exp(-10 * 3.28e-5 * (z - thickness_sa) / ccapacity)) + DeltaT
    # Ta = 1262 / np.exp(-10 * 3.28e-5 * (z - thickness_sa) / ccapacity)

    T[T < 0.0] = 0.0
    cond1 = Ta<T #VICTOR
    T[T > Ta] = Ta[T > Ta] #apply the temperature of asthenosphere Ta where temperature T is greater than Ta, 

    # kappa = 0.75*1.0e-6 #thermal diffusivity
    kappa = 1.0e-6 #thermal diffusivity

    H = np.zeros_like(T)

    cond = (z >= thickness_sa) & (z < thickness_upper_crust + thickness_sa)  # upper crust
    H[cond] = Huc

    cond = (z >= thickness_upper_crust + thickness_sa) & (
        z < thickness_lower_crust + thickness_upper_crust + thickness_sa
    )  # lower crust
    H[cond] = Hlc

    Taux = np.copy(T)
    t = 0
    dt = 5000
    dt_sec = dt * 365 * 24 * 3600
    # cond = (z > thickness_sa + thickness_litho) | (T == 0)  # (T > 1300) | (T == 0) #OLD
    cond = cond1 | (T == 0)  # (T > 1300) | (T == 0) #VICTOR
    dz = Lz / (Nz - 1)

    
    while t < 500.0e6:
        T[1:-1] += (
            kappa * dt_sec * ((T[2:] + T[:-2] - 2 * T[1:-1]) / dz ** 2)
            + H[1:-1] * dt_sec / ccapacity
        )
        T[cond] = Taux[cond]
        t = t + dt
    
    T = np.ones_like(X) * T[:, None] #(Nz, Nx)

    print('shape T: ', np.shape(T))

    # Save the initial temperature file
    np.savetxt("input_temperature_0.txt", np.reshape(T, (Nx * Nz)), header="T1\nT2\nT3\nT4")

else:
    dz = Lz / (Nz - 1)

    print('Used external scenario: ', scenario)
    scenario_infos.append('Used external scenario: ' + scenario)

    from_dataset = True
    # from_dataset = False

    if(from_dataset == True):
        # print('entrei from dataset')
        # local = True
        local = False
        if(local==True):
            fpath = f"{machine_path}/{scenario}"
        else:
            # print('entrei local false')
            external_media = 'Joao_Macedo'
            if(path[1] == 'home'):
                print('entrei home')
                fpath = f"{machine_path}/{external_media}{scenario}"
                print
            elif(path[1] == 'Users'):
                fpath = f'/Volumes/{external_media}{scenario}'

            elif(path[1] == 'media'):
                fpath = f"{machine_path}/{external_media}{scenario}"
            elif(path[1] == 'Volumes'):
                fpath = f'/Volumes/{external_media}{scenario}'

        # external_media = 'Joao_Macedo'
        # fpath = f"{machine_path}/{external_media}{scenario}"

        dataset = xr.open_dataset(f"{fpath}_output_temperature.nc")
        
        Nx_aux = int(dataset.nx)
        Nz_aux = int(dataset.nz)
        Lx_aux = float(dataset.lx)
        Lz_aux = float(dataset.lz)

        x_aux = np.linspace(0, Lx_aux, Nx_aux)
        z_aux = np.linspace(Lz_aux, 0, Nz_aux)
        xx_aux, zz_aux  = np.meshgrid(x_aux, z_aux)

        time = dataset.time[-1]
        Datai = dataset.temperature[-1].values.T
    else:
        fpath = f"{machine_path}/Doutorado/cenarios/mandyoc/{scenario}/"
        Nx_aux, Nz_aux, Lx_aux, Lz_aux = read_params(fpath)

        x_aux = np.linspace(0, Lx_aux, Nx_aux)
        z_aux = np.linspace(Lz_aux, 0, Nz_aux)
        xx_aux, zz_aux  = np.meshgrid(x_aux, z_aux)

        steps = sorted(glob.glob(fpath+"time_*.txt"), key=os.path.getmtime)
        step_final = int(steps[-1].split('/')[-1][5:-4]) #step of final thermal structure
        
        time_fname = fpath + 'time_' + str(step_final) + '.txt'
        time = np.loadtxt(time_fname, usecols=2, max_rows=1)

        Datai = read_data('temperature', step_final, Nz_aux, Nx_aux, fpath) #(read final thermal structure (Nz, Nx)
    

    #Setting procedure with external temperature field. Choose between:
        ##Use the horizontal mean of temperature from final step of used scenario (horizontal_mean)
        ##or
        ##Use the original thermal state used as input interpolated on new grid Nx x Nz (interp2d)

    if(scenario_kind == 'rifting'):
        interp_method = 'horizontal_mean' #using interp1d
        # interp_method = 'interp2d'
    else:
        interp_method = 'horizontal_mean' #using interp1d
        # interp_method = 'interp2d'
    
    print('Interpolation of temperature field using: '+interp_method)
    scenario_infos.append('Interpolation of temperature field using: '+interp_method)
    

    if(interp_method == 'horizontal_mean'):

        print('Keel center: '+str(keel_center))
        scenario_infos.append('Keel center: '+str(keel_center))

        if(keel_center==True):
            xregion = (xx_aux>=700.0e3) & (xx_aux <= 900.0e3) #craton
            Data_region = Datai[xregion].reshape(Nz_aux, len(xregion[0][xregion[0]==True]))
            datai_mean = np.mean(Data_region, axis=1) #horizontal mean

        elif(selection_in_preset == True):
            xcenter = Lx_aux/2.0
            region  = (xx_aux >= xcenter - 500.0e3) & (xx_aux <= xcenter + 500.0e3) & (zz_aux >= 0.0e3) & (zz_aux <= Lz_aux)

            xregion = (xx_aux >= xcenter - 500.0e3) & (xx_aux <= xcenter + 500.0e3)
            zregion = (zz_aux >= 0.0e3) & (zz_aux <= Lz_aux)

            Nx_new = len(xregion[0][xregion[0] == True])
            Nz_new = len(zregion.T[0][zregion.T[0] == True])

            Data_region = np.asarray(Datai)[region].reshape(Nz_new, Nx_new)
            datai_mean = np.mean(Data_region, axis=1)
        elif(keel_adjust == True):
        	xcenter = Lx_aux/2.0
            region  = (xx_aux >= xcenter - Lcraton/2.0) & (xx_aux <= xcenter + Lcraton/2.0) & (zz_aux >= 0.0e3) & (zz_aux <= Lz_aux)

            datai_mean = calc_mean_temperaure_region(Datai, Nz_aux, xx_aux, 0, Lx_aux)

            Tk_mean = np.copy(datai_mean)
            cond_mlit = (z_aux <= thickening+thickness_sa) & (z_aux >= thickness_sa + thickness_upper_crust + thickness_lower_crust)
			         
			T1 = datai_mean[cond_mlit][0] #bottom
			T0 = datai_mean[cond_mlit][-1] #top
			z1 = z[cond_mlit][0]
			z0  = z[cond_mlit][-1]

			Tk_mean[cond_mlit] = ((T1 - T0) / (z1 - z0)) * (z[cond_mlit] - z0) + T0

			fk = interp1d(z_aux, Tk_mean)
			Tk_mean_interp = fk(z)
			Tk_mean_interp[Tk_mean_interp <= 1.0e-7] = 0.0 #dealing with <=0 values inherited from interpolation
       	 	Tk_mean_interp[zcond] = 0.0

       	 	#Find the keel interval - PAREI AQUI
       	 	# xcond = 

        else:
            datai_mean = np.mean(Datai, axis=1) #horizontal mean

        f = interp1d(z_aux, datai_mean) #funcion to interpolate the temperature field
        datai_mean_interp = f(z) #applying the function to obtain the temperature field to the new mesh

        zcond = z <= 40.0e3
        datai_mean_interp[datai_mean_interp <= 1.0e-7] = 0.0 #dealing with <=0 values inherited from interpolation
        datai_mean_interp[zcond] = 0.0

        T = np.zeros((Nx, Nz)) #(Nx, Nz) = transpose of original shape (Nz, Nx)
        
        for i in range(Nx): #len(Nx)
            T[i, :] = datai_mean_interp

        T = T.T #(Nz,Nx): transpose T to plot below
        print('shape T: ', np.shape(T))
    
    else:
        interp_kind = 'linear'
        # interp_kind = 'cubic'
        # interp_kind = 'quintic'
        print('Interpolation method: ' + interp_kind)
        scenario_infos.append('Interpolation method: ' + interp_kind)

        f = interp2d(x_aux, z_aux, Datai, kind=interp_kind)
        temper_interp = f(x, z) #(Nz, Nx)
        temper_interp[temper_interp <= 1.0e-7] = 0.0 #dealing with <=0 values inherited from interpolation
        
        #Setting temperature on vertical boundaries. Choose between:
        ##Use the mean temperature from final step of used scenario (mean)
        ##or
        ##Use the original thermal state used as input interpolated on new Nz (original)
        bound = 'mean'
        # bound = 'original'

        print('Temperature of boundaries: ' + bound)

        scenario_infos.append('Temperature of boundaries: ' + bound)
        if(bound == 'mean'):
            #Calc horizontal mean from interpolated field
            temper_interp_mean = np.mean(temper_interp, axis=1)
            zcond = z >= 660.0e3 #temperature field is from bottom to top
            temper_interp_mean[zcond] = 0.0
            
            if(mean_litho==True):

                if(thickness_litho == 80.0e3):
                    zcond1 = (z >= 660.0e3-thickness_litho) & (z < 660.0e3)
                else:
                    zcond1 = (z >= 560.0e3) & (z < 660.0e3)
                temper_interp = temper_interp.T #Change to (Nx, Nz)
                
                for i in range(Nx): #len(Nx)
                    temper_interp[i][zcond1] = temper_interp_mean[zcond1]

                temper_interp = temper_interp.T #Return to (Nz, Nx)

            #Apply horizontal mean to vertical boundaries
            for i in range(Nz):
                temper_interp[i][0] = temper_interp_mean[i]
                temper_interp[i][-1] = temper_interp_mean[i]

        else:
            #Cat the initial thermal state from scenario
            step_initial = int(steps[0].split('/')[-1][5:-4])
            time_fname = fpath + 'time_' + str(step_initial) + '.txt'
            time = np.loadtxt(time_fname, usecols=2, max_rows=1)
            T0i = read_data('temperature', step_initial, Nz_aux, Nx_aux, fpath)

            #interpolate in new grid
            T0 = T0i[:, 0]
            f = interp1d(z_aux, T0)
            T0_interp = f(z)
            T0_interp[T0_interp<=1.0e-7] = 0.0
            T0_interp = T0_interp[::-1]
            
            #apply to boundaries
            for i in range(Nz):
                temper_interp[i][0] = T0_interp[i]
                temper_interp[i][-1] = T0_interp[i]
         
        T = temper_interp[::-1]
        print('shape Temper: ', np.shape(T))

    np.savetxt("input_temperature_0.txt", np.reshape(T, (Nx * Nz)), header="T1\nT2\nT3\nT4")

############################################################################## 
# Plot temperature field and thermal profile
##############################################################################

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(16, 8), sharey=True)

im = ax0.contourf(X / 1.0e3, (Z - thickness_sa) / 1.0e3, T,
                  levels=np.arange(0, np.max(T) + 100, 100))
ax0.set_ylim((Lz - thickness_sa) / 1.0e3, -thickness_sa / 1000)
ax0.set_xlabel("km", fontsize=label_size)
ax0.set_ylabel("km", fontsize=label_size)
ax1.set_xlabel("$^\circ$C", fontsize=label_size)
cbar = fig.colorbar(im, orientation='horizontal', ax=ax0)
cbar.set_label("Temperature [°C]")

ax1.plot(T[:, 0], (z - thickness_sa) / 1.0e3, "-k")
T_xlim = 2000 #oC
code = 0

for label in list(interfaces.keys()):
    color = "C" + str(code)
    code += 1
    ax1.hlines(
        (interfaces[label][0] - thickness_sa) / 1.0e3, #y
        np.min(T[:, 0]), #xmin
        T_xlim, #np.max(T[:, 0]), #xmax
        label=f"{label}",
        color=color
    )

ax1.set_ylim((Lz - thickness_sa) / 1.0e3, -thickness_sa / 1000)
ax1.set_xlim(0, T_xlim)
ax0.grid(':k', alpha=0.7)
ax1.grid(':k', alpha=0.7)
ax1.legend(loc='lower left', fontsize=14)
plt.savefig("initial_temperature_field.png")
plt.close()


##############################################################################
# Boundary condition - velocity
##############################################################################
if(velocity_from_ascii == True):
    fac_air = 10.0e3

    # 1 cm/year
    vL = 0.005 / (365 * 24 * 3600)  # m/s

    h_v_const = thickness_litho + 20.0e3  #thickness with constant velocity 
    ha = Lz - thickness_sa - h_v_const  # difference

    vR = 2 * vL * (h_v_const + fac_air + ha) / ha  # this is to ensure integral equals zero

    VX = np.zeros_like(X)
    cond = (Z > h_v_const + thickness_sa) & (X == 0)
    VX[cond] = vR * (Z[cond] - h_v_const - thickness_sa) / ha

    cond = (Z > h_v_const + thickness_sa) & (X == Lx)
    VX[cond] = -vR * (Z[cond] - h_v_const - thickness_sa) / ha

    cond = X == Lx
    VX[cond] += +2 * vL

    cond = Z <= thickness_sa - fac_air
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

    ax0.plot(VX[:, 0]*1e10, (z - thickness_sa) / 1000, "k-", label="left side")
    ax1.plot(VZ[:, 0]*1e10, (z - thickness_sa) / 1000, "k-", label="left side")

    ax0.plot(VX[:, -1]*1e10, (z - thickness_sa) / 1000, "r-", label="right side")
    ax1.plot(VZ[:, -1]*1e10, (z - thickness_sa) / 1000, "r-", label="right side")

    ax0.legend(loc='upper left', fontsize=14)
    ax1.legend(loc='upper right', fontsize=14)

    ax0_xlim = ax0.get_xlim()
    ax1_xlim = ax1.get_xlim()

    ax0.set_yticks(np.arange(0, Lz / 1000, 40))
    ax1.set_yticks(np.arange(0, Lz / 1000, 40))

    ax0.set_ylim([Lz / 1000 - thickness_sa / 1000, -thickness_sa / 1000])
    ax1.set_ylim([Lz / 1000 - thickness_sa / 1000, -thickness_sa / 1000])

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


    if(variable_bcv == True):
        
        if(scenario_kind == 'accordion' or scenario_kind == 'accordion_keel' or scenario_kind == 'accordion_lit_hetero'):
            if(bellon == False):
                var_bcv = f""" 1
                            30.0 -1.0

                            """
            else:#Bellon case
                var_bcv = f""" 2
                            25.0 -1.0
                            90  -1.0E-15
                            """
        elif(scenario_kind == 'quiescence'):
            var_bcv = f""" 2
                        50.0 -1.0
                        150  -1.0E-15
                        """

        # Create the parameter file
        with open("scale_bcv.txt", "w") as f:
            for line in var_bcv.split("\n"):
                line = line.strip()
                if len(line):
                    f.write(" ".join(line.split()) + "\n")

if(sp_surface_processes == True):
    if(climate_change_from_ascii == True):
        #When climate effects will start to act - scaling to 1
        # climate = f'''
        #         2
        #         0 0.0
        #         10 0.02
        #     '''

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


#Save scenario infos
np.savetxt('infos_'+path[-1] + '.txt', scenario_infos, fmt="%s")

#Creating run files

run_gcloud = f'''
        #!/bin/bash
        MPI_PATH=$HOME/opt/petsc/arch-0-fast/bin
        MANDYOC_PATH=$HOME/opt/mandyoc
        NUMBER_OF_CORES=6
        touch FD.out
        $MPI_PATH/mpirun -n $NUMBER_OF_CORES $MANDYOC_PATH/mandyoc -seed 0,2 -strain_seed 0.0,1.0 | tee FD.out
        bash /home/joao_macedo/Doutorado/cenarios/mandyoc/scripts/zipper_gcloud.sh
        bash /home/joao_macedo/Doutorado/cenarios/mandyoc/scripts/clean_gcloud.sh
        sudo poweroff
    '''
with open('run_gcloud.sh', 'w') as f:
    for line in run_gcloud.split('\n'):
        line = line.strip()
        if len(line):
            f.write(' '.join(line.split()) + '\n')

run_linux = f'''
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

run_mac = f'''
        #!/bin/bash
        MPI_PATH=$HOME/opt/petsc/arch-0-fast/bin
        MANDYOC_PATH=$HOME/opt/mandyoc
        NUMBER_OF_CORES=6
        touch FD.out
        $MPI_PATH/mpirun -n $NUMBER_OF_CORES $MANDYOC_PATH/mandyoc -seed 0,2 -strain_seed 0.0,1.0 | tee FD.out
        bash $HOME/Doutorado/cenarios/mandyoc/scripts/zipper_gcloud.sh
        #bash $HOME/Doutorado/cenarios/mandyoc/scripts/clean_gcloud.sh
    '''
with open('run_mac.sh', 'w') as f:
    for line in run_mac.split('\n'):
        line = line.strip()
        if len(line):
            f.write(' '.join(line.split()) + '\n')

dirname = '${PWD##*/}'
run_aguia = f'''
        #!/usr/bin/bash

        #SBATCH --partition=SP2
        #SBATCH --ntasks={str(int(ncores))}
        #SBATCH --nodes=1
        #SBATCH --cpus-per-task=1
        #SBATCH --time 192:00:00 #16horas/"2-" para 2 dias com max 8 dias
        #SBATCH --job-name mandyoc-jpms
        #SBATCH --output slurm_%j.log #ou FD.out/ %j pega o id do job
        #SBATCH --mail-type=BEGIN,FAIL,END
        #SBATCH --mail-user=joao.macedo.silva@usp.br

        export PETSC_DIR=/temporario/jpmsilva/petsc
        export PETSC_ARCH=v3.15.5-optimized
        MANDYOC=/temporario/jpmsilva/mandyoc/bin/mandyoc
        MANDYOC_OPTIONS='-seed 0,2 -strain_seed 0.0,1.0'

        $PETSC_DIR/$PETSC_ARCH/bin/mpiexec -n {str(int(ncores))} $MANDYOC $MANDYOC_OPTIONS

        DIRNAME={dirname}

        zip $DIRNAME.zip interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh *.log
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

#zip input files
filename = 'inputs_'+path[-1]+'.zip'
files_list = ' infos*.txt interfaces.txt param.txt input*_0.txt run*.sh vel*.txt scale_bcv.txt *.png precipitation.txt climate.txt'
os.system('zip '+filename+files_list)