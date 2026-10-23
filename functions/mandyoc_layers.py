'''
Classes describing Mandyoc layers and their rheologies.
Copied from opt/scripts/generate_input_files_database.py so they can be shared
by svg2interfaces.py and the generate_input_files*.py scripts.
'''

import numpy as np

#CLASSES
class MandyocLayer:
    def __init__(self, layer_label, rheology: type,
                 density, effective_viscosity_scale_factor=1.0, radiogenic_heat_production=0.0,
                 base_depth=0.0e3, Nx=None):
        """"
        This class creates a layer with the given properties
        layer_label: str
            Name of the layer
        rheology: class
            Rheological properties of the layer onbtained from LithologicalUnit class
        density: float
            Density of the layer [kg/m3]
        interface: NoneType or np.array
            Interface of the layer.
            If None, the interface will be setted after the creation of the Layer.
            If np.array, the interface is defined by the given array.
        effective_viscosity_scale_factor: float
            Scale factor for the effective vistocisty
        radiogenic_heat_production: float
            Radiogenic heat production of the layer [W/kg]
        base_depth: float
            Depth of the layer base [m]
        Nx: int
            Number of points in x direction
        """

        self.layer_label = layer_label
        self.rheology = LithologicalUnit(rheology)
        self.density = density
        self.effective_viscosity_scale_factor = effective_viscosity_scale_factor
        self.radiogenic_heat_production = radiogenic_heat_production
        self.rheology_name = self.rheology.name
        self.pre_exponential_constant = self.rheology.pre_exponential_constant
        self.power_law_exponent = self.rheology.power_law_exponent
        self.activation_energy = self.rheology.activation_energy
        self.activation_volume = self.rheology.activation_volume
        self.base_depth = base_depth
        self.Nx = Nx

        self.interface = np.ones(Nx)*base_depth

class LithologicalUnit:
    """"
    This class calls the respective rheological properties of the given mineral

    mineral_name: class
        Mineral rheology written in CamelCase. For example, WetOlivine, DryOlivine, WetQuartz
    """
    def __init__(self, mineral_name: type):
        self.mineral_name = mineral_name() # mineral_name is a class, so we need to call it to get the object
        self.name = self.mineral_name.name
        self.pre_exponential_constant = self.mineral_name.pre_exponential_constant
        self.power_law_exponent = self.mineral_name.power_law_exponent
        self.activation_energy = self.mineral_name.activation_energy
        self.activation_volume = self.mineral_name.activation_volume

class WetOlivine:
    """
    Wet olivine rheological properties
    """
    def __init__(self):
        self.name = 'wet_olivine'
        self.pre_exponential_constant = 1.393e-14
        self.power_law_exponent = 3
        self.activation_energy = 429.0e3
        self.activation_volume = 15.0e-6

class DryOlivine:
    """
    Dry olivine rheological properties
    """
    def __init__(self):
        self.name = 'dry_olivine'
        self.pre_exponential_constant = 2.4168e-15
        self.power_law_exponent = 3.5
        self.activation_energy = 540.0e3
        self.activation_volume = 25.0e-6

class WetQuartz:
    """
    Wet quartz rheological properties
    """
    def __init__(self):
        self.name = 'wet_quartz'
        self.pre_exponential_constant = 8.574e-28
        self.power_law_exponent = 4.0
        self.activation_energy = 222.0e3
        self.activation_volume = 0.0

class DryQuartz:
    """
    Dry quartz rheological properties
    """
    def __init__(self):
        self.name = 'dry_quartz'
        self.pre_exponential_constant = 0.0
        self.power_law_exponent = 0.0
        self.activation_energy = 0.0
        self.activation_volume = 0.0

class Basalt:
    """
    Basalt rheological properties
    """
    def __init__(self):
        self.name = 'basalt'
        self.pre_exponential_constant = 8.574e-28
        self.power_law_exponent = 4.0
        self.activation_energy = 222.0e3
        self.activation_volume = 0.0

class Plagioclase:
    """
    Plagioclase rheological properties (Shelton and Tullis, 1981)
    """
    def __init__(self):
        self.name = 'plagioclase'
        self.pre_exponential_constant = 0.0
        self.power_law_exponent = 3.05
        self.activation_energy = 276.e3
        self.activation_volume = 0.0

class Air:
    """
    Air rheological properties
    """
    def __init__(self):
        self.name = 'air'
        self.pre_exponential_constant = 1.0e-18
        self.power_law_exponent = 1.0
        self.activation_energy = 0.0
        self.activation_volume = 0.0


RHEOLOGIES = [WetOlivine, DryOlivine, WetQuartz, DryQuartz, Basalt, Plagioclase, Air]

def get_rheology(name):
    '''
    Return the rheology class from a free-form name.
    "wet olivine", "wet_olivine", "Wet Olivine" and "WetOlivine" all return WetOlivine.
    '''
    key = name.replace(" ", "").replace("_", "").replace("-", "").lower()
    for rheology in RHEOLOGIES:
        if rheology.__name__.lower() == key:
            return rheology
    options = ", ".join(rheology().name for rheology in RHEOLOGIES)
    raise ValueError(f"Unknown rheology '{name}'. Options: {options}")
