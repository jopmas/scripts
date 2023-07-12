# Extrai o perfil da topografia e da resposta flexural de acordo com as idades informadas na lista idades #

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('des.mplstyle')


def extrai_topo(pasta, ax1):
	with open("setup.txt", "r") as f: # no arquivo setup retira o valor de Nz e fecha o arquivo
   		for l in f.readlines(): #le as linhas num array cada elemento do array eh uma linha
   			d = l.split('=') #elimina o = do texto
			if 'Nz' in d[0]: 
   		    		Nz = int(d[1])

	x = []
	y = []
	u = []
	v = []

	lon_emb = []
	t_emb = []
	lon_isp = []
	h_isp = []

	x,y = np.loadtxt("xy_coords.txt", skiprows=1,unpack=True)
	xx = x[0::Nz]

	#RETIRANDO A RESPOSTA FLEXURL EM 40E6 Myr

	u,v = np.loadtxt("uv_" + str(int(40.0E6)).zfill(9) + ".txt", skiprows=1, unpack= True) #Le os deslocamentos dos nos em x e y, respectivamente 
	vv = v[0::Nz] #Coloca em vv os valores do comeco ate o fim de v, com passo de Nz

	ax1.plot(xx/1.0E3, vv, linewidth = 1.0, alpha = 0.5, linestyle = '--', label=r'$\mathrm{H_{lit}}$ $\mathrm{=}$ $\mathrm{ }$' + r'$\mathrm{' + pasta[4::] + r'}$ $\mathrm{km}$')

	os.chdir("../") #saindo da pasta
	

def plot_Ussami(arq, ax1):

	x,w = np.loadtxt(arq, unpack = True)

	if("cte" in arq):
		ax1.plot(x, w*1000., '.', linewidth = 1.0, alpha = 0.5, label=r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{ }$' + r'$\mathrm{' + arq[15::] + r'}$ $\mathrm{km,}$ $\mathrm{\rho_{cte}}$')
	else:
		ax1.plot(x, w*1000., linewidth = 1.0, alpha = 0.5, linestyle = '-', label=r'$\mathrm{T_{e}}$ $\mathrm{=}$ $\mathrm{ }$' + r'$\mathrm{' + arq[15::] + r'}$ $\mathrm{km,}$ $\mathrm{\rho_{var}}$')




### MAIN ###

fig, ax1 = plt.subplots()
fig.figsize = (10,8)

#PLOT GEOMETRIA DOS ANDES

x_and = [0.0, 50000.0, 100000.0, 200000.0, 230000.0, 260000.0, 280000.0, 320000.0, 420000.0, 460000.0, 500000.0, 600000.0, 2500000.0]
y_and = [4000.0, 4300.0, 3500.0, 3400.0, 4000.0, 3400.0, 3400.0, 2100.0, 1600.0, 2300.0, 400.0, 0.0, 0.0]
ax1.plot(np.array(x_and)/1.0E3, np.array(y_and), '-', color = 'brown', linewidth = 1.5, alpha = 0.7, label = r'$\mathrm{Topografia}$ $\mathrm{Andina}$')

fator =  111.12*np.cos(18.5*np.pi/180.0) #fator de conversao da lon na escala de dist do modelo

#LENDO PERFIL DO TOPO DO EMBASAMENTO E ISOPACAS, DPS PLOT
lon_emb, t_emb = np.loadtxt("perfil_185.txt", usecols=(0,2), unpack=True)

x0=70.0*fator

x_emb = (lon_emb)*fator + x0 #7382.34142605 #km
prof_emb = t_emb*(-3.5E3)*0.5 #km
sigma_prof = -t_emb*0.5E3
print sigma_prof

#ax1.plot(x_emb, prof_emb, '.', linewidth = 2.5, alpha = 0.7, label = r'$\mathrm{Topo}$ $\mathrm{do}$ $\mathrm{Embasamento}$') #para que o -70 graus seja 0 em km
ax1.errorbar(x_emb, prof_emb, yerr = sigma_prof, xerr = None, fmt='.', color='k', alpha=0.7, elinewidth=1.0, label = r'$\mathrm{Topo}$ $\mathrm{do}$ $\mathrm{Embasamento}$', capthick = 1.0, mew=1.0)

#LENDO AS ISOPACAS DE USSAMI et al 1999, DPS PLOT
lon_isp, h_isp = np.loadtxt("isp_perfil.txt", usecols=(0,2), unpack=True)
ax1.plot((lon_isp*fator + x0), (-1.0)*h_isp, '.', color='k', linewidth = 2.5, alpha = 0.7, label = r'$\mathrm{Pantanal}$')


#PLOT RESPOSTAS FLEXURAIS

pastas = ["lit-100", "lit-125", "lit-150"]# ["lit-70", "lit-80", "lit-100", "lit-125", "lit-150"]#, "lit-200"] #pastas para extrair os perfis
print "Entrei no for"
for pasta in pastas:
	print "pasta: %s"%pasta
	os.chdir(pasta)
	extrai_topo(pasta, ax1)

#PLOT MODELOS USSAMI

arqs = ["Ussami_drho_cte_50.txt", "Ussami_drho_cte_100.txt", "Ussami_drho_cte_150.txt", "Ussami_drho_var_50.txt", "Ussami_drho_var_100.txt", "Ussami_drho_var_150.txt"]

for arq in arqs:
	plot_Ussami(arq, ax1)


#AJUSTES FINAIS

#COLOCANDO O SEGUNDO EIXO EM GRAUS
ax2 = ax1.twiny()
ax1Xs = ax1.get_xticks()

ax2Xs = []
for X in ax1Xs:
    ax2Xs.append((0.0095*X)-70) #medida/(X*cos(18.5)) = 0.009489*medida ~= 0.0095

ax2.set_xticks(ax1Xs)
ax2.set_xbound(ax1.get_xbound())
ax2.set_xticklabels(ax2Xs)

print ax1
print ax2
print ax1Xs
print ax2Xs

ax1.set_xlim(0,2500)
ax2.set_xlim(0,2500)

ax1.set_xlabel(r'$\mathrm{x(km)}$', fontsize = 16)
ax2.set_xlabel(r'$\mathrm{Longitude}$ $\mathrm{(graus)}$', fontsize = 16)

ax1.set_ylabel(r'$\mathrm{Deslocamento}$ $\mathrm{Vertical}$ $\mathrm{(m)}$', fontsize = 16)
#ax2.set_ylabel(r'$\mathrm{Deslocamento}$ $\mathrm{Vertical}$ $\mathrm{(m)}$', fontsize = 16)

ax1.legend(loc='best', fontsize = 12)	
title = ax1.set_title(r'$\mathrm{Resposta}$ $\mathrm{Flexural}$', fontsize = 18)
title.set_y(1.1)
fig.subplots_adjust(top=0.85)

#plt.savefig("verif_flex_TODOS.pdf")
#plt.savefig("verif_flex_TODOS.png")
plt.savefig("verif_flex_parcial_com_ussami.pdf")










