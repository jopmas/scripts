# Extrai o perfil da topografia e da resposta flexural de acordo com as idades informadas na lista idades #

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.style.use('des.mplstyle')


def extrai_topo(pasta):
	with open("setup.txt", "r") as f: # no arquivo setup retira o valor de Nz e fecha o arquivo
   		for l in f.readlines(): #le as linhas num array cada elemento do array eh uma linha
   			d = l.split('=') #elimina o = do texto
			if 'Nz' in d[0]: 
   		    		Nz = int(d[1])
	#print Nz
	plt.figure(figsize = (10,8))
	casos = {"lit-70":29,"lit-80":33,"lit-100":41, "lit-150":61,"lit-200":81}
	x =[]
	y =[]
	u =[]
	v =[]

	topo_data = np.loadtxt("topography.txt")
	x,y = np.loadtxt("xy_coords.txt", skiprows=1,unpack=True)
	xx = x[0::Nz]
	#print len(xx)
	#print len(topo_data[2][1::])
	#RETIRANDO A TOPOGRAFIA

	inicio_cor = 0
	fim_cor = len(topo_data)
	passo = 100
	colors = matplotlib.cm.get_cmap('inferno')
	#tamanho_topo = np.arange(inicio_cor,fim_cor,passo)
	cores_flex = []
	
	for i in np.arange(inicio_cor,fim_cor,passo):
		idades_norm = ((topo_data[i][0]/1.0E6 - 4)/50) #Nao sei pq do 4
		c = colors(idades_norm)
		cores_flex.append(c)
#		if(pasta == 'lit-70'):
		plt.plot(xx/1.0E3,topo_data[i][1:], label = str(topo_data[i][0]/1.0E6) + " Myr", color=c, linewidth = 2.5, alpha = 0.7)
	#	print i
	#print cores_flex[0]
	#RETIRANDO A RESPOSTA FLEXURAL

	inicio = 0.0E6
	fim = 50.0E6 #50 PQ NA HORA DE FAZER O ARANGE ELE NAO COLOCA O ULTIMO ELEMNTO!
	soma_idade = 10.0E6 

	idades = np.arange(inicio,fim,soma_idade)
	#print idades
	cor = cores_flex
	#print np.arange(0,len(idades))

	#print "no for das idades"
	for i in np.arange(0,len(idades)): #vou procurando nos indices 
		r = int(idades[i])
		#print "tempo: %d"%r
		u,v = np.loadtxt("uv_" + str(r).zfill(9) + ".txt", skiprows=1, unpack= True) #Le os deslocamentos dos nos em x e y, respectivamente 
		#print len(v)		
		vv = v[0::Nz] #Coloca em vv os valores do comeco ate o fim de v, com passo de 51
		#print len(vv)
		plt.plot(xx/1.0E3, vv, color = cores_flex[i], linewidth = 2.5, alpha = 0.7, linestyle = '--')

	print "sai da pasta %s"%pasta
	os.chdir("../")
	#pasta_temp = 
	print "salvando"
	
##### TG - PORTUGUES ####
	#plt.xlim(0,2500)
	#plt.xlabel(r'$\mathrm{x(km)}$', fontsize = 16)
	#plt.ylabel(r'$\mathrm{Deslocamento}$ $\mathrm{Vertical}$ $\mathrm{(m)}$', fontsize = 16)
	#plt.legend(loc='best', fontsize = 12)	
	#plt.title(r'$\mathrm{Resposta}$ $\mathrm{Flexural}$ $\mathrm{(linha}$ $\mathrm{tracejada)}$ $\mathrm{de}$ $\mathrm{acordo}$ $\mathrm{com}$ $\mathrm{o}$ $\mathrm{soerguimento}$ $\mathrm{dos}$ $\mathrm{Andes}$ $\mathrm{(linha}$ $\mathrm{cheia),}$' + '\n' + r'$\mathrm{H_{lit}}$ $\mathrm{=}$ $\mathrm{ }$' + r'$\mathrm{' + pasta[4::] + r'}$ $\mathrm{km}$' , fontsize = 18)
	#plt.savefig("topo_total_" + pasta + ".pdf")

##### POSTER- PORTUGUES ####
	
	if(pasta=="lit-150"):
		plt.xlim(0,1600)
		plt.text(1100, 4550, r'$\mathrm{Regime}$ $\mathrm{n}\tilde{\mathrm{a}}\mathrm{o}$ $\mathrm{compresivo}$', fontsize = 18)
	else:
		plt.xlim(0,2000)
	plt.xlabel(r'$\mathrm{x}$ $\mathrm{(m)}$', fontsize = 20)
	plt.ylabel(r'$\mathrm{Deslocamento}$ $\mathrm{Vertical}$ $\mathrm{(m)}$', fontsize = 20)
	plt.legend(loc='best', fontsize = 20)	
	plt.title(r'$\mathrm{Resposta}$ $\mathrm{flexural}$ $\mathrm{conforme}$ $\mathrm{o}$ $\mathrm{soerguimento}$ $\mathrm{dos}$ $\mathrm{Andes}$' + '\n' + r'$\mathrm{H_{lit}}$ $\mathrm{=}$ $\mathrm{ }$' + r'$\mathrm{' + pasta[4::] + r'}$ $\mathrm{km}$', fontsize = 20)
	plt.text(300, -11000, r'$\mathrm{Resposta}$ $\mathrm{flexural}$', fontsize = 16)
	plt.text(300, 3800, r'$\mathrm{Topografia}$ $\mathrm{andina}$', fontsize = 16)
	plt.savefig("POSTER_topo_total_" + pasta + ".png", dpi=300)

##### INGLES #####
	#plt.xlim(0,2500)
	#plt.xlabel('x(km)', fontsize = 16)
	#plt.ylabel('Vertical Displacement (m)', fontsize = 16)
	#plt.legend(loc='best', fontsize = 12)	
	#plt.title(r'$\mathrm{Flexural}$ $\mathrm{response}$ $\mathrm{(dashed}$ $\mathrm{line)}$ $\mathrm{according}$ $\mathrm{to}$ $\mathrm{Andes}$ $\mathrm{uplifting}$ $\mathrm{(solid}$ $\mathrm{line),}$' + '\n' + r'$\mathrm{H_{lit}}$ $\mathrm{=}$ $\mathrm{ }$' + r'$\mathrm{' + pasta[4::] + r'}$ $\mathrm{km}$' , fontsize = 18)
	#plt.savefig("ENG_topo_total_" + pasta + ".pdf")
	#plt.show()

#MAIN
pastas = ["lit-70", "lit-80", "lit-100", "lit-150", "lit-200"] #pastas para extrair os perfis
print "Entrei no for"
for pasta in pastas:
	print "pasta: %s"%pasta
	os.chdir(pasta)
	extrai_topo(pasta)
