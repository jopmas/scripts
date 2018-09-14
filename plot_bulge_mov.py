# -*- coding: utf-8 -*-
# Extrai o movimento da ombreira (bulge)em relacao ao centro do orogeno e salva os valores em um .txt #

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plota(pasta):
	dic_color = {"lit-70":"xkcd:scarlet","lit-80":"xkcd:dark green","lit-100":"xkcd:light orange","lit-150":"xkcd:lighter purple", "lit-200":"xkcd:dark blue"}
	tempo,desloc = np.loadtxt(pasta+"_sr.txt", unpack= True) #Carrega as coordenadas xy nos vetores tempo, desloc
	
	plt.plot(tempo, desloc, marker='.', linewidth=2.5, markersize=6.0, alpha=0.7, color=dic_color[pasta])

#MAIN
plt.style.use('des.mplstyle')
fig=plt.figure(figsize=(10,8))
fig, ax = plt.subplots()

x=[2,40]
y1 = [777, 777, 1154, 1154]
x1=[-5,-5]
y=[-5,-5]


#Plot ptos NAOMI E DECELLES

#DECELLES
t,des = np.loadtxt("pos_rho_cte.txt", unpack=True)
plt.plot(t/1.0E6, des/1.0E3, 'k*', linewidth=2.5, markersize=6.0, alpha=0.7, label = r'$\mathrm{DeCelles}$ $\mathrm{d_{\rho}}$ $\mathrm{cte}$')
t,des = np.loadtxt("pos_rho_var.txt", unpack=True)
plt.plot(t/1.0E6, des/1.0E3, 'r*', linewidth=2.5, markersize=6.0, alpha=0.7, label = r'$\mathrm{DeCelles}$ $\mathrm{d_{\rho}}$ $\mathrm{var}$')

#USSAMI
t,des = np.loadtxt("USSAMI_pos_rho_cte.txt", unpack=True)

Tes = ['50', '100', '150']
for i in np.arange(0,len(des)):  
	plt.plot(t[i]/1.0E6, des[i]/1.0E3, '^', linewidth=2.5, markersize=6.0, alpha=0.7, label = r'$\mathrm{Te}$ $\mathrm{=}$ $\mathrm{ }$' + r'$\mathrm{' + Tes[i] + r'}$ $\mathrm{km}$ $\mathrm{Ussami}$ $\mathrm{d_{\rho}}$ $\mathrm{cte}$')

te=0
t,des = np.loadtxt("USSAMI_pos_rho_var.txt", unpack=True)
for i in np.arange(0,len(des)):
	plt.plot(t[i]/1.0E6, des[i]/1.0E3, '^', linewidth=2.5, markersize=6.0, alpha=0.7, label = r'$\mathrm{Te}$ $\mathrm{=}$ $\mathrm{ }$' + r'$\mathrm{' + Tes[i] + r'}$ $\mathrm{km}$ $\mathrm{Ussami}$ $\mathrm{d_{\rho}}$ $\mathrm{var}$')

#plt.plot(t/1.0E6, des/1.0E3, 'ro', linewidth=2.5, markersize=6.0, alpha=0.7, label = r'$\mathrm{USd_{\rho}}$ $\mathrm{var}$')

plt.plot(x, y, color = 'xkcd:scarlet', alpha = 0.7, label = r'$H_{\mathrm{lit}}$ = 70 km')
plt.plot(x, y, color = 'xkcd:dark green', alpha = 0.7, label = r'$H_{\mathrm{lit}}$ = 80 km')
plt.plot(x, y, color = 'xkcd:light orange', alpha = 0.7, label = r'$H_{\mathrm{lit}}$ = 100 km')
plt.plot(x, y, color = 'xkcd:lighter purple', alpha = 0.7, label = r'$H_{\mathrm{lit}}$ = 150 km')
plt.plot(x, y, color = 'xkcd:dark blue', alpha = 0.7, label = r'$H_{\mathrm{lit}}$ = 200 km')

#'xkcd:scarlet', 'xkcd:dark green', 'xkcd:light orange', 'xkcd:lighter purple'

pastas = ["lit-70","lit-80","lit-100", "lit-150", "lit-200"] #pastas para extrair os perfis

print "Entrei no for da crosta"

for pasta in pastas:
	#print "pasta: %s"%pasta
	plota(pasta)

xmin, xmax = 0.5, 40
ymin, ymax = 777, 1154

x = [xmin, xmax, xmax, xmin]
y = [ymin, ymin, ymax, ymax]

plt.fill(x,y, color = 'xkcd:cerulean blue', alpha = 0.3)


##### TG - PORTUGUES #####

#ax.text(11, 987.5, r'$\mathrm{Bacia}$ $\mathrm{do}$ $\mathrm{Pantanal}$ $\mathrm{(Ussami}$ $\mathrm{\textit{et.}}$ $\mathrm{\textit{al.}}$ $\mathrm{1999)}$', fontsize = 16)#, style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':10}) #TG

#plt.xlabel(r"$\mathrm{Tempo}$ $\mathrm{ }$ $\mathrm{(Myr)}$", fontsize = 16)#TG
#plt.ylabel(r'$\mathrm{Dist}\hat{\mathrm{a}}\mathrm{ncia}$ $\mathrm{do}$ $\mathrm{topo}$ $\mathrm{da}$ $\mathrm{ombreira}$ $\mathrm{ao}$ $\mathrm{centro}$ $\mathrm{do}$ $\mathrm{or}\acute{\mathrm{o}}\mathrm{geno}$ $\mathrm{(km)}$', fontsize = 16) #TG

#plt.title(r'$\mathrm{Evolu}\c{\mathrm{c}}\tilde{\mathrm{a}}o$ $\mathrm{da}$ $\mathrm{posi}\c{\mathrm{c}}\tilde{\mathrm{a}}\mathrm{o}$ $\mathrm{da}$ $\mathrm{ombreira}$ $\mathrm{para}$ $\mathrm{diferentes}$ $\mathrm{cen}\acute{\mathrm{a}}\mathrm{rios.}$'+ '\n' + r'$\mathrm{(Sem}$ $\mathrm{a}$ $\mathrm{tens}\tilde{\mathrm{a}}\mathrm{o}$ $\mathrm{regional)}$', fontsize = 18)

##### POSTER - PORTUGUES #####

ax.text(10, 987.5, r'$\mathrm{Bacia}$ $\mathrm{do}$ $\mathrm{Pantanal}$ $\mathrm{(Ussami}$ $\mathrm{\textit{et.}}$ $\mathrm{\textit{al.}}$ $\mathrm{1999)}$', fontsize = 20)#, style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

plt.xlabel(r"$\mathrm{Tempo}$ $\mathrm{ }$ $\mathrm{(Myr)}$", fontsize = 20)
plt.ylabel(r'$\mathrm{Dist}\hat{\mathrm{a}}\mathrm{ncia}$ $\mathrm{do}$ $\mathrm{topo}$ $\mathrm{da}$ $\mathrm{ombreira}$ $\mathrm{ao}$ $\mathrm{centro}$ $\mathrm{do}$ $\mathrm{or}\acute{\mathrm{o}}\mathrm{geno}$ $\mathrm{(km)}$', fontsize = 20)

plt.title(r'$\mathrm{Evolu}\c{\mathrm{c}}\tilde{\mathrm{a}}\mathrm{o}$ $\mathrm{da}$ $\mathrm{posi}\c{\mathrm{c}}\tilde{\mathrm{a}}\mathrm{o}$ $\mathrm{da}$ $\mathrm{ombreira}$ $\mathrm{para}$ $\mathrm{diferentes}$ $\mathrm{cen}\acute{\mathrm{a}}\mathrm{rios.}$'+ '\n' + r'$\mathrm{Regime}$ $\mathrm{n}\tilde{\mathrm{a}}\mathrm{o}$ $\mathrm{compresivo}$', fontsize = 20)


##### INGLES #####

#ax.text(11, 987.5, r'$\mathrm{Pantanal}$ $\mathrm{Basin}$ $\mathrm{(Ussami}$ $\mathrm{\textit{et.}}$ $\mathrm{\textit{al.}}$ $\mathrm{1999)}$', fontsize = 16)#, style='italic', bbox={'facecolor':'white', 'alpha':0.5, 'pad':10}) #apresentacao
#plt.xlabel(r"$\mathrm{Time}$ $\mathrm{ }$ $\mathrm{(Myr)}$", fontsize = 16)#Apresentacao
#plt.ylabel(r'$\mathrm{Distance}$ $\mathrm{from}$ $\mathrm{the}$ $\mathrm{forebulge}$ $\mathrm{top}$ $\mathrm{to}$ $\mathrm{orogen}$ $\mathrm{center}$ $\mathrm{ }$ $\mathrm{(km)}$', fontsize = 16) #Apresentacao
#plt.title(r'$\mathrm{Evolution}$ $\mathrm{of}$ $\mathrm{forebulge}$ $\mathrm{position}$ $\mathrm{for}$ $\mathrm{different}$ $\mathrm{scenarios}$', fontsize = 22)

plt.xlim([0,40.5])
plt.ylim([350,1650])

plt.legend(loc='best', fontsize = 8)


#plt.title(r'$\ddot{o}\acute{e}\grave{e}\hat{O}\breve{i}\bar{A}\tilde{n}\vec{q}$', fontsize=20)
plt.savefig("NOVO_dists_sr700.pdf")
plt.savefig("POSTER_dists_sr_TEST700.png", dpi=300)
