#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Aug 19 10:18:56 2024

@author: jobueno
"""

from scipy.ndimage import median_filter
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
import glob, os
from shutil import copy
from scipy.ndimage import gaussian_filter1d


warnings.filterwarnings('ignore')

cdir = 'C:/Users/Usuario/Desktop/MR_va_30_15_ris0.6_hkoff_1350'
path_param = os.path.join(cdir,'param.txt')
#dfp = pd.read_csv('~/doutorado/general-layers-params.csv',delimiter=',')

var_simulacao = cdir.split('/')[-1].split('_')

def get_lasttime(fdir):
    files = glob.glob(os.path.join(cdir,'time*'))
    times = []
    for f in files:
        times.append(int(f[:-4].split('_')[-1]))
    
    return np.max(times)


def read_params(path_param):
    '''
    Carrega os parâmetros do arquivos param.txt em forma de dicionario
    '''
    params_form = {}
    ptemp = ''
    with open(path_param,'r') as param:
        for line in param:
            line = line.strip()
            if len(line)==0:
                continue
            elif line[0] == "#":
                continue
            
            line = line.split('#')[0]
            line = line.replace(' ','')
            pv = line.split('=')
            params_form[pv[0].lower()] = pv[1]
            ptemp = ptemp + line+'\n'

    return params_form

def read_data(file, veloc=False, surface=False):
    data = pd.read_csv(file, header=None, 
                       skiprows=2, comment='P')
    
    data = data.to_numpy()

    if not(veloc) and not(surface):
        data[np.abs(data) < 1.0e-200] = 0 #converter numeros pequenos e grandes
        data = np.reshape(data, (Nx,Nz), order='F') #(nx*nz,1) -> (nx,nz)
        data = data.T
        
    elif surface == True:
        return data
        
    else:
        vx = np.reshape(data[0::2], (Nx,Nz), order='F')
        vy = np.reshape(data[1::2], (Nx,Nz), order='F')
        data = (vx.T, vy.T)
    return data


def get_interfaces(xi, zi, dens, denss=[3378,3354,2800,2700,1000],processing_crust=False):
    '''
    Parameters
    ----------
    
    xi : np.array
        x domain.
    zi : np.array
        z domain.
    denss : list, optional
        references densities to find the interfaces. The default is [3378,3354,2800,2700,1000].

    Returns
    -------
    dic_interfaces : dictionary
        dictionary with the z of the interfaces among the x domain
        {0 : -200, -200.5, ...,} where 0 is the first interface.

    '''
    dic_interfaces={d:[] for d in range(len(denss)-1)}
    
    for i in range(len(xi)):
        #x = xi[i]
        densz = dens[:,i]
        for c in range(len(range_interfaces)-1):
            dmax = range_interfaces[c]
            dmin = range_interfaces[c+1]
            zint = zi[np.where((densz<=dmax) & (densz>dmin))] #intervalo em z com as interfaces
            
            if len(zint)==0: #quando não tem interfaces entre aquelas duas densidades
                #print(f'{dmax}-{dmin}')
                search_interface = True
                j = c + 1
                while search_interface == True:
                    j+=1
                    dmax = range_interfaces[c]
                    dmin = dmin - 100
                    zint = zi[np.where((densz<dmax) & (densz>dmin))]
                    
                    
                    search_interface=False if len(zint) > 0 else True
                zadd = np.min(zint)
            else:
                zadd = np.max(zint)
            dic_interfaces[c].append(zadd)
            
    if processing_crust == True:
        dic_interfaces[3] = median_filter(dic_interfaces[3],size=8,mode='nearest') #generalizar camadas depois
    return dic_interfaces

def heatflux_interface(temp,interface,xi,zi,k=2.25):
    '''
    

    Parameters
    ----------
    temp : 2d-array
        temperature.
    interface : 1d-array
        array with z depth of the interface along x.
    xi : 1d-array
        x domain.
    zi : 1d-array
        z domain.
    k : float, optional
        thermal conductivity. The default is 2.25 W/m/k (upper crust)

    Returns
    -------
    heatflux_crust : 1d-array (W/m²)
        heat flux in the given interface.
        
    [tempgrad,heatflux] : 2d-arrays
    gradient of the  (K/m)
    heat flux (W/m²)

    '''
    res = (zi[1]-zi[0])*1e3
    tempgrad = np.array(np.gradient(temp,res)) #Y,X - gradiente termico
    heatflux = -k_crust * tempgrad  #multiplicação assumindo k constante (nao importa porque só vou pegar a crosta)
    
    #modtempgrad = np.sqrt(tempgrad[0]**2 + tempgrad[1]**2) #módulo do gradiente termico
    modheatflux = np.sqrt(heatflux[0]**2 + heatflux[1]**2) #módulo do fluxo termico
    
    heatflux_crust = []
    
    for i in range(len(xi)):                    #iterando colunas ao longo de x
        modhf_z = modheatflux[:,i]              #fluxo na coluna
        gradhfcrust = modhf_z[zi==crust[i]]     #valor do gradiente na linha de interface naquela coluna
        
        gradhfcrust = float(gradhfcrust) #float((gradhfcrust + modhf_z[zi==(crust[i]-1)])/2) #média do fluxo na interface com a célula de baixo, 1 corresponde 2celulas, com resolução da malha de 0,5 km
        heatflux_crust.append(gradhfcrust)
    
    return np.array(heatflux_crust), tempgrad, heatflux

os.chdir(cdir)
params = read_params(path_param)


thick_air = 40 #km

Nx = int(params['nx'])
Nz = int(params['nz'])
Lx = int(float(params['lx'])) #m
Lz = int(float(params['lz'])) #m
step_final = 8200 #get_lasttime(cdir)     #int(input('STEPFINAL: '))#5000 #int(params['step_max'])
d_step = 25 #int(params['step_print'])  #int(input('STEP: '))
step_initial = 0

xi = np.linspace(0, Lx / 1e3, Nx)
zi = np.linspace(-Lz / 1e3, 0, Nz)
xx, zz = np.meshgrid(xi, zi)
range_interfaces=[3378,3354,2800,2700,1000]


tstep = 2800


qprocessing = True
hf = []
hf_t = []
crusttop = []
for tstep in range(step_initial, step_final, d_step):

    time = np.loadtxt(f'time_{tstep}.txt', dtype='str')  #[] = bug
    time = time[:, -1]
    time = time.astype("float")
    
    
    dens = read_data(f'density_{tstep}.txt') #ler densidade
    temp = read_data(f'temperature_{tstep}.txt') #ler temperatura

    #linhas de interface
    
    interfaces = get_interfaces(xi, zi, dens, range_interfaces,True)
    
    
    tmy = time[0]/1.0e6 #my
    
    
        
    crust = np.array(interfaces[3]) #interface da crosta superior
    #crust = median_filter(crust,size=12,mode='nearest')
    k_crust = 2.25 #W/m/K - condutividade termica
    
    
    
    heatflux_crust, tempgrad, heatflux = heatflux_interface(temp,crust,xi,zi,k=k_crust)
    
    if qprocessing==True:
        heatflux_crust = gaussian_filter1d(heatflux_crust,5,mode='nearest')
    
    hf.append(heatflux_crust)
    hf_t.append(tmy)
    crusttop.append(crust)

xxt, ttt = np.meshgrid(xi, hf_t)


fig1, axs = plt.subplots(2,1,sharex=True, figsize=(20,10))
hf_plot = axs[0].contourf(xxt,ttt, np.array(hf)*1e3,cmap='magma',levels=50,
                          vmin=30,vmax=400)
fig1.colorbar(hf_plot,cax=axs[0].inset_axes((1.01, 0, 0.02, 1)),
              label='Heat flux (mW/m²)', orientation='vertical')

tp_plot = axs[1].contourf(xxt,ttt, (np.array(crusttop)+thick_air)*1e3, cmap='coolwarm',
                levels=20,vmin=-5000, vmax=5000)

fig1.colorbar(tp_plot,cax=axs[1].inset_axes((1.01, 0, 0.02, 1)),
              label='z (m)', orientation='vertical')

axs[0].set_ylabel('time (myr)')
axs[1].set_ylabel('time (myr)')
#axs[1].set_ylabel('z (m)')
axs[1].set_xlabel('width (km)')

plt.tight_layout()
plt.savefig("hf_subsid.png", dpi=600)
plt.show()

#plots


def cf(dens, densmax=3380,densmin=1350):
    return np.round((dens-densmax)/(densmax-densmin)+1,3)

densmax=3380 #astenosfera
densmin=1350 #ar

dens_levels = np.append(np.arange(1350,2801,50),np.arange(3300,3381,10)) #[1800, 2150, 2500, 2700, 2800, 3354, 3378]
my_gradient = [
#https://eltos.github.io/gradient/#0:AAAAAA-27.2:4CBBFF-42:0025B3-66.5:FB443B-71.5:FF8E33-98.5:C3F8A8-100:63A053
    [cf(1350), '#9C3868'], #1350
    [cf(1900), '#9C3868'], #1900
    [cf(2200), '#0025B3'], #2200
    [cf(2700), '#FB443B'], #2700
    [cf(2800), '#FF8E33'], #2800
    [cf(3350), '#63A053'], #3350
    [cf(3380), '#C3F8A8']] #3380

my_cmap = LinearSegmentedColormap.from_list('my_gradient', my_gradient)
zmax = 5
zmin = -Lz/1e3+40

cond = ((zz < zmax) & (zz > zmin))
xx2 = xx[cond]
zz2 = zz[cond]

xx2 = np.reshape(xx2,(int(len(xx2)/Nx), Nx))
zz2 = np.reshape(zz2,(int(len(zz2)/Nx), Nx))

#células pra plotar os vetores
xskip=4
zskip=20

plt.figure(1, figsize=(20,10))
plt.contourf(xx, (zz + thick_air), dens, cmap=my_cmap,levels=dens_levels)

plt.colorbar()

plt.quiver(xx[::xskip,::zskip],zz[::xskip,::zskip]+ thick_air,-tempgrad[1][::xskip,::zskip],-tempgrad[0][::xskip,::zskip], 
           width=0.0009,scale=3,pivot='middle',color='grey')

for i in range(len(interfaces.keys())):
    #plt.plot(xi, np.array(interfaces[i])+thick_air,'-k')
    plt.plot(xi,gaussian_filter1d(np.array(interfaces[i]),5)+thick_air,'-k')

plt.ylim(-50, zmax)
plt.title(f't={tmy}')


plt.figure(2,figsize=(20,10))
plt.contourf(xx, (zz + thick_air), temp, levels=50, cmap='copper')
plt.colorbar()
plt.quiver(xx[::xskip,::zskip],zz[::xskip,::zskip]+ thick_air,-tempgrad[1][::xskip,::zskip],-tempgrad[0][::xskip,::zskip], 
           width=0.0009,scale=2,pivot='middle',color='white')
plt.ylim(-50, zmax)
plt.title(f't={tmy}')


plt.figure(3,figsize=(7,4))
plt.plot(xi, heatflux_interface(temp,crust,xi,zi,k=k_crust)[0]*1e3,'k-',label='output')
plt.title(f't={tmy}')
if qprocessing==True:
    plt.plot(xi,heatflux_crust*1e3,'r-',label='filtered')

plt.xlabel('km')
plt.ylabel('mW/m²')
plt.legend()
