
# Extrai o movimento da ombreira (bulge)em relacao ao centro do orogeno e salva os valores em um .txt #

import os
import numpy as np
import matplotlib.pyplot as plt


def calcula_min(vv, dx, xx): #vv=valores de um array
    #print("aux1  aux2  derivada2, i")
    #coord =[]
	for i in np.arange(0,len(vv),1):
		
		if(i+1 == len(vv)):
		    return None
		
		aux1 = (vv[i+1]-vv[i])/dx #derivada de um par
		aux2 = (vv[i+2]-vv[i+1])/dx # derivada do par seguinte
		derivada2 = (aux2-aux1)/dx #segunda derivada
		#print(aux1, aux2, derivada2)
		          
		if(aux1>=0 and aux2<0 and derivada2<0):#Verificando mudanca na concav.
		    ponto_min = vv[i+1] #valor pra retornar
		    posicao = i+1 #indice pra retornar
		    #coord = np.append(coord, posicao)
		    #coord = np.append(coord, ponto_min)
		    return posicao

def extrai_mov(pasta):
	idades = []
	i = 2.0E6
	while i <= 40.0E6:
		idades = np.append(idades, i)
		i = i + 2.0E6

	x,y = np.loadtxt("xy_coords.txt", skiprows=1, unpack= True) #Carrega as coordenadas xy nos vetores x, y
	casos = {"lit-70":29,"lit-80":33,"lit-100":41,"lit-150":61, "lit-200":81}
	xx = x[0::casos[pasta]]/1E3 #Coloca em xx os valores do comeco ate o fim de x com o passo de acordo com o dic casos
	dx = xx[1]-xx[0]
	print "no for das idades"

	desloc = []
	tempo = []
	for r in idades:
		r = int(r)
		print "tempo: %d"%r
		u,v = np.loadtxt("uv_" + str(r).zfill(9) + ".txt", skiprows=1, unpack= True) #Le os deslocamentos dos nos em x e y, respectivamente 
		vv = v[0::casos[pasta]] #Coloca em vv os valores do comeco ate o fim de v, com passo de 51
				
		#plt.plot(r/1.0E6, xx[calcula_min(vv,dx)] - xx[60], color = 'blue', marker = "o") #60 e a posicao do centro do orogeno , label = (str(r/1E6) + " Ma")
		
		posicao = int(calcula_min(vv,dx,xx))
		desloc = np.append(desloc, xx[posicao] - xx[60])
		tempo = np.append(tempo, r/1.0E6)
		
	#plt.plot(tempo, desloc, color = 'black', linestyle = '-')
	print "sai das pasta"
	os.chdir("../")
	print "salvando"
	np.savetxt(pasta+"_sr.txt",np.array([tempo, desloc]).T) #.T e pra transpor


#MAIN

#crostas=["crosta-30", "crosta-45"]
pastas = ["lit-70","lit-80","lit-100","lit-150", "lit-200"]#pastas para extrair os perfis
print "Entrei no for da pastas"

#for crosta in crostas:
#	print "pasta: %s"%crosta
#	os.chdir(crosta)

for pasta in pastas:
	print "pasta: %s"%pasta
	os.chdir(pasta)
	extrai_mov(pasta)
os.chdir("../")
