### Este programa reproduz os modelos de flexurais elasticos de Ussami et al., 1999 e Horton e DeCelles 1997 ###

import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('des.mplstyle')

## FUNCOES ##

def w_num_ussami(x,p,Te=100000., drho=500., g=9.8,E=1.0E11,nu=0.25,rompida=False, continua=False):
    D = E*Te**3/(12*(1-nu**2)) #Rigidez flexural
    dx = x[1]-x[0]
    
    N = np.size(x)
    
    A = np.zeros((N,N))
    
    A[range(N),range(N)]= 6*D + dx**4*drho*g
    A[range(0,N-1),range(1,N)]=-4*D
    A[range(1,N),range(0,N-1)]=-4*D
    A[range(0,N-2),range(2,N)]=D
    A[range(2,N),range(0,N-2)]=D
    
    if rompida == True:
        A[0,0] = 2*D + dx**4*drhox[0]*g
        A[0,1] = -4*D
        A[0,2] = 2*D
        A[1,0] = -2*D
        A[1,1] = 5*D + dx**4*drhox[1]*g
    
    if continua == True:
        A[-1,-2] = -8*D
        A[-1,-3] = 2*D
        A[-2,-2] = 7*D + dx**4*drho[-2]*g
        
    w = np.linalg.solve(A,dx**4*p)
    
    return(w)

def calcula_min(w, dx, x): # RETORNA O PONTO DO TOPO DA OMBREIRA
    #print("aux1  aux2  derivada2, i")
    #coord =[]
	for i in np.arange(0,len(w)+1,1):
		
		if(i+1 == len(w)):
		    return None
		
		aux1 = (w[i+1]-w[i])/dx #derivada de um par
		aux2 = (w[i+2]-w[i+1])/dx # derivada do par seguinte
		derivada2 = (aux2-aux1)/dx #segunda derivada
		#print(aux1, aux2, derivada2)
		          
		if(aux1>=0 and aux2<0 and derivada2<0):#Verificando mudanca na concav.
		    ponto_min = w[i+1] #valor pra retornar
		    posicao = i+1 #indice pra retornar
		    #coord = np.append(coord, posicao)
		    #coord = np.append(coord, ponto_min)
		    return posicao

## MAIN ##


x_total = 2000.0E3
n = 101
x = np.linspace(0,x_total,n)
dx = x[1]-x[0]
fator =  111.12*np.cos(18.5*np.pi/180.0) #fator de conversao da lon na escala de dist do modelo

#DADOS DA TOPOGRAFIA QUE TIREI DO MODELO NUMERICO#
#x_topo = [0.0, 50000.0, 100000.0, 200000.0, 230000.0, 260000.0, 280000.0, 320000.0, 420000.0, 460000.0, 500000.0, 600000.0]
#h_topo_final = [4300.0, 4600.0, 3800.0, 3700.0, 4300.0, 3700.0, 3700.0, 2400.0, 1900.0, 2600.0, 700.0, 300.0]

x_topo, h_topo_final = np.loadtxt("perfil_ussami_graus.txt", unpack = True)
#print x_topo

x0 = (70.*np.pi/180.)*6371.8*np.cos(18.5*np.pi/180.0) #deixando meu 0 em -70
#print x0
x_topo_final = (x_topo*fator -  x_topo[0]*fator)*1000 #7382.34142605

h_topo_final = np.array(h_topo_final)-365.417

np.savetxt('CERTO_perfil_ussami_km.txt', np.array([x_topo_final, h_topo_final]).T) #.T e pra transpor

#Interpolacao dos pontos de topografia na malha do modelo
H = np.interp(x,x_topo_final,h_topo_final,right=0)

#Carga 
p = -2700*9.8*H #depende da carga que eu coloco em h_topo_final

#print x_topo
#print h_topo_final
#print H
#print x
#print p

#print x_topo[0]*fator

#### MODELO USSAMI ####

#Contrastes de Densidade ao longo do modelo

drhoxb=x*0+500 #constante em x: rho_m - rho_c)                                  
drhox=x*0+500 #variavel em x, em que:
drhox[x<=700000.] = 500. #rho_m - rho_c
drhox[x>700000.] = 700. #rho_sed - rho_m

#Variacao lateral de DRHO para diferentes Te's
w0 = w_num_ussami(x,p,Te=50000.,drho=drhox,rompida=True)
w1 = w_num_ussami(x,p,Te=100000.,drho=drhox,rompida=True)
w2 = w_num_ussami(x,p,Te=150000.,drho=drhox,rompida=True)

lista_pos_drho_var = [] # lista com  as pos do topo do forebulge para os diferentes Te's
lista_desloc_var = [] #lista com a dist ao centro do orogeno
tempo = []
t = 40.0E6
pos_drho_var = calcula_min(w0, dx, x) 
desloc = x[pos_drho_var] - x[17]
lista_pos_drho_var = np.append(lista_pos_drho_var, pos_drho_var)
lista_desloc_var = np.append(lista_desloc_var, desloc)
tempo = np.append(tempo, t)

pos_drho_var = calcula_min(w1, dx, x) 
desloc = x[pos_drho_var] - x[17]
lista_pos_drho_var = np.append(lista_pos_drho_var, pos_drho_var)
lista_desloc_var = np.append(lista_desloc_var, desloc)
tempo = np.append(tempo, t)
tempo = np.array(tempo)

pos_drho_var = calcula_min(w2, dx, x) 
desloc = x[pos_drho_var] - x[17]
lista_pos_drho_var = np.append(lista_pos_drho_var, pos_drho_var)
lista_desloc_var = np.append(lista_desloc_var, desloc)
tempo = np.append(tempo, t)

np.savetxt('USSAMI_pos_rho_var.txt', np.array([tempo, lista_desloc_var]).T)

#DRHO constante ao longo do modelo para diferentes Te's
wa = w_num_ussami(x,p,Te=50000.,drho=drhoxb,rompida=True)
wb = w_num_ussami(x,p,Te=100000.,drho=drhoxb,rompida=True)
wc = w_num_ussami(x,p,Te=150000.,drho=drhoxb,rompida=True)

lista_pos_drho_cte = [] # lista com  as pos do topo do forebulge para os diferentes Te's
lista_desloc_cte = [] #lista com a dist ao centro do orogeno

pos_drho_cte = calcula_min(wa, dx, x) 
desloc = x[pos_drho_cte] - x[17]
lista_pos_drho_var = np.append(lista_pos_drho_cte, pos_drho_cte)
lista_desloc_cte = np.append(lista_desloc_cte, desloc)

pos_drho_cte = calcula_min(wb, dx, x) 
desloc = x[pos_drho_cte] - x[17]
lista_pos_drho_cte = np.append(lista_pos_drho_cte, pos_drho_cte)
lista_desloc_cte = np.append(lista_desloc_cte, desloc)

pos_drho_cte = calcula_min(wc, dx, x) 
desloc = x[pos_drho_cte] - x[17]
lista_pos_drho_cte = np.append(lista_pos_drho_cte, pos_drho_cte)
lista_desloc_cte = np.append(lista_desloc_cte, desloc)

np.savetxt('USSAMI_pos_rho_cte.txt', np.array([tempo, lista_desloc_cte]).T)



#LENDO PERFIL DO TOPO DO EMBASAMENTO

lon_emb, t_emb = np.loadtxt("perfil_185.txt", usecols=(0,2), unpack=True)

x_emb = lon_emb*fator + x0# 7382.34142605 #km
prof_emb = t_emb*(-3.5E3)*0.5 #km
sigma_prof = -t_emb*0.5E3


## PLOTAGEM ##

fig1=plt.figure(figsize=(10,8))
fig1, ax1 = plt.subplots()

#PLOT drho variavel#
ax1.plot(x/1000., H/1000., color='xkcd:brown', alpha=0.7, label = r'$\mathrm{Topografia}$ $\mathrm{Andina}$')
ax1.plot(x/1000., w0/1000., '.', color='xkcd:cerulean blue', alpha=0.7, label = r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{50}$ $\mathrm{km}$')
ax1.plot(x/1000., w1/1000., color='xkcd:cerulean blue', alpha=0.7, label = r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{100}$ $\mathrm{km}$')
ax1.plot(x/1000., w2/1000., '--', color='xkcd:cerulean blue', alpha=0.7, label = r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{150}$ $\mathrm{km}$')

np.savetxt('Ussami_50_drho_var.txt', np.array([x/1000., w0/1000.]).T)
np.savetxt('Ussami_100_drho_var.txt', np.array([x/1000., w1/1000.]).T)
np.savetxt('Ussami_150_drho_var.txt', np.array([x/1000., w2/1000.]).T)

#PLOT drho constante#
ax1.plot(x/1000., wa/1000., '.', color='xkcd:dark green', alpha=0.7, label = r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{50}$ $\mathrm{km}$')
ax1.plot(x/1000., wb/1000., color='xkcd:dark green', alpha=0.7, label = r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{100}$ $\mathrm{km}$')
ax1.plot(x/1000., wc/1000., '--', color='xkcd:dark green', alpha=0.7, label = r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{150}$ $\mathrm{km}$')

np.savetxt('Ussami_50_drho_cte.txt', np.array([x/1000., wa/1000.]).T)
np.savetxt('Ussami_100_drho_cte.txt', np.array([x/1000., wb/1000.]).T)
np.savetxt('Ussami_150_drho_cte.txt', np.array([x/1000., wc/1000.]).T)


ax1.text(1650, 4, r'$\mathrm{d_{\rho}$ $\mathrm{constante}$', fontsize = 18, color='xkcd:dark green')

ax1.text(1650, 2, r'$\mathrm{d_{\rho}$ $\mathrm{variavel}$', fontsize = 18, color='xkcd:cerulean blue')

ax1.errorbar(x_emb, prof_emb/1000, yerr = sigma_prof/1000, xerr = None, color = 'red', fmt='.', alpha=0.7, elinewidth=1.0, capthick = 1.0, label = r'$\mathrm{Topo}$ $\mathrm{do}$ $\mathrm{Embasamento}$')

ax1.set_xlabel(r'$\mathrm{x(km)}$', fontsize = 16)
ax1.set_ylabel(r'$\mathrm{Deslocamento}$ $\mathrm{Vertical}$ $\mathrm{(km)}$', fontsize = 16)
ax1.legend(loc=4)
ax1.yaxis.set_ticks(np.arange(-24, max(H)/1000, 2.0)) #invertendo o sentido do eixo/discretizacao do eixo
ax1.set_xlim(-1,2000)
ax1.set_title(r'$\mathrm{Modelo}$ $\mathrm{Ussami}$ $\mathrm{\textit{et}}$ $\mathrm{\textit{al}.,}$ $\mathrm{1999}$', fontsize = 16)

plt.savefig("Modelo_USSAMI2.pdf")





