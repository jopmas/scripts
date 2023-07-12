### Este programa reproduz os modelos de flexurais elasticos de Ussami et al., 1999 e Horton e DeCelles 1997 ###

import os
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('des.mplstyle')

## FUNCOES ##

def w_num_decelles(x,p,Te=100000., drho=500., g=9.8,E=1.0E11,nu=0.25,rompida=False, continua=False):
    D = 7.0E23 #Rigidez flexural Nm
    dx = x[1]-x[0]
    
    N = np.size(x)
    
    A = np.zeros((N,N))
    
    A[range(N),range(N)]= 6*D + dx**4*drho*g
    A[range(0,N-1),range(1,N)]=-4*D
    A[range(1,N),range(0,N-1)]=-4*D
    A[range(0,N-2),range(2,N)]=D
    A[range(2,N),range(0,N-2)]=D
    
    if rompida == True:
        A[0,0] = 2*D + dx**4*drhox_dc[0]*g
        A[0,1] = -4*D
        A[0,2] = 2*D
        A[1,0] = -2*D
        A[1,1] = 5*D + dx**4*drhox_dc[1]*g
    
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

fator =  111.12*np.cos(18.0*np.pi/180.0) #fator de conversao da lon na escala de dist do modelo

#DADOS DA TOPOGRAFIA DeCelles#

x_topo, h_topo_final = np.loadtxt("perfil_decelles_graus.txt", unpack = True)


x0 = (70.*np.pi/180.)*6371.8*np.cos(18.0*np.pi/180.0)#/(2*np.pi)
print x_topo[0]
print x_topo[0]*fator
print 70.*fator
print x0

x_topo_final = (x_topo*fator - x_topo[0]*fator)*1000 #7323.135175 #tranformando em km x_km = x_long*110*cos(18); x_km(-70.) = -7323.135175

h_topo_final = np.array(h_topo_final)-510.9

np.savetxt('CERTO_perfil_decelles_km.txt', np.array([x_topo_final, h_topo_final]).T) #.T e pra transpor

#Interpolacao dos pontos de topografia na malha do modelo
H = np.interp(x,x_topo_final,h_topo_final,right=0)

#Carga 
p = -2700*9.8*H #depende da carga que eu coloco em h_topo_final
#print p
#### MODELO DECELLES ####

#print x_topo[0]*fator
#print x_topo_final
#print h_topo_final
#print H
#print np.size(H)
#print x
#print np.size(x)
#print p

rho_sed_DC = 2400.
rho_c_DC = 2650.
rho_m_DC = 3300.

drhoxb_dc=x*0+650. #(rho_m_DC - rho_c_DC) #constante em x: rho_m - rho_c)                                  
drhox_dc=x*0+650. #(rho_m_DC - rho_c_DC) #variavel em x, em que:
drhox_dc[x<=700000.] = 650. #rho_m_DC - rho_c_DC #rho_m - rho_c
drhox_dc[x>700000.] = 900. #rho_m_DC - rho_sed_DC #rho_sed - rho_m

#Variacao lateral de DRHO para diferentes Te's
w0dc = w_num_decelles(x,p,Te=50000.,drho=drhox_dc,rompida=True)


pos_drho_var = calcula_min(w0dc, dx, x)
desloc = x[pos_drho_var] - x[18]
tempo = 40.0E6
np.savetxt('pos_rho_var.txt', np.array([tempo, desloc]).T)

#w1dc = w_num_decelles(x,p,Te=100000.,drho=drhox_dc,rompida=True)
#w2dc = w_num_decelles(x,p,Te=150000.,drho=drhox_dc,rompida=True)

#DRHO constante ao longo do modelo para diferentes Te's
wadc = w_num_decelles(x,p,Te=50000.,drho=drhoxb_dc,rompida=True)

pos_drho_cte = calcula_min(wadc, dx, x)
desloc = x[pos_drho_cte] - x[18]
np.savetxt('pos_rho_cte.txt', np.array([tempo, desloc]).T)
#wbdc = w_num_decelles(x,p,Te=100000.,drho=drhoxb_dc,rompida=True)
#wcdc= w_num_decelles(x,p,Te=150000.,drho=drhoxb_dc,rompida=True)

#LENDO PERFIL DO TOPO DO EMBASAMENTO

lon_emb, t_emb = np.loadtxt("topo_emb.txt", usecols=(0,2), unpack=True)

x_emb = lon_emb*fator + x0 #km
prof_emb = t_emb*(-3.5E3)*0.5 #km
sigma_prof = -t_emb*0.5E3
#print t_emb
#print prof_emb
#print w0dc
## PLOTAGEM ##

fig, ax1 = plt.subplots()
fig.figsize = (10,8)

#PLOT drho variavel#
ax1.plot(x/1000., H/1000., color='xkcd:brown', alpha=0.7, label = r'$\mathrm{Topografia}$ $\mathrm{Andina}$')
ax1.plot(x/1000., w0dc/1000., '--', color='xkcd:cerulean blue', alpha=0.7, label = r'$\mathrm{d_{\rho}}$ $\mathrm{variavel}$')

#ax1.plot(x/1000., w1dc/1000., color='xkcd:cerulean blue', alpha=0.7, label = r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{100}$ $\mathrm{km}$')
#ax1.plot(x/1000., w2dc/1000., '--', color='xkcd:cerulean blue', alpha=0.7, label = r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{150}$ $\mathrm{km}$')
np.savetxt('DeCelles_drho_var.txt', np.array([x/1000., w0dc/1000.]).T)

#PLOT drho constante#
ax1.plot(x/1000., wadc/1000., '--', color='xkcd:dark green', alpha=0.7, label = r'$\mathrm{d_{\rho}}$ $\mathrm{constante}$')
#ax1.plot(x/1000., wbdc/1000., color='xkcd:dark green', alpha=0.7, label = r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{100}$ $\mathrm{km}$')
#ax1.plot(x/1000., wcdc/1000., '--', color='xkcd:dark green', alpha=0.7, label = r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{150}$ $\mathrm{km}$')
np.savetxt('DeCelles_drho_cte.txt', np.array([x/1000., wadc/1000.]).T)


#PLOT topo do embasamento

#ax1.plot(x_emb, prof_emb, '.', linewidth = 2.5, alpha = 0.7, label = r'$\mathrm{Topo}$ $\mathrm{do}$ $\mathrm{Embasamento}$')


ax1.errorbar(x_emb, prof_emb/1000, yerr = sigma_prof/1000, xerr = None, color = 'red', fmt='.', alpha=0.7, elinewidth=1.0, capthick = 1.0, label = r'$\mathrm{Topo}$ $\mathrm{do}$ $\mathrm{Embasamento}$')



ax1.set_xlabel(r'$\mathrm{x(km)}$', fontsize = 16)
ax1.set_ylabel(r'$\mathrm{Deslocamento}$ $\mathrm{Vertical}$ $\mathrm{(km)}$', fontsize = 16)
ax1.legend(loc=4)
ax1.yaxis.set_ticks(np.arange(-24, max(H)/1000, 2.0)) #invertendo o sentido do eixo/discretizacao do eixo
ax1.set_xlim(-1,2000)

ax1.set_title(r'$\mathrm{Modelo}$ $\mathrm{Horton}$ $\mathrm{\&}$ $\mathrm{DeCelles,}$ $\mathrm{1997}$'+'\n'+r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{42.9}$ $\mathrm{(km)}$''', fontsize = 16)

plt.savefig("Modelo_DeCelles2.pdf")






