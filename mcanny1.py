"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
CANY GRADIENTS
ABSTRACT: We define a function that receive maps and return gradients.
------------------------------------------------------------------------------
"""


import math
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from numpy import matrix
choices={-4:[-1,1],-3:[-1,0],-2:[-1,-1],-1:[0,-1],1:[0,1],2:[1,1],3:[1,0],4:[1,-1]}
from PIL import Image, ImageFilter
""""Funcion que recibe un mapa y retorna el mapa de los gradientes
"""
def convolucion(mapa,matriz):
    size=mapa.shape
    mapa2=np.zeros(size,dtype=float)
    for i in range(1,size[0]-1):
        for j in range(1,size[1]-1):
            valor=mapa.item(i-1,j-1)*matriz[0][0]+mapa.item(i-1,j)*matriz[0][1]+mapa.item(i-1,j+1)*matriz[0][2]
            +mapa.item(i,j-1)*matriz[1][0]+mapa.item(i,j)*matriz[1][1]+mapa.item(i,j+1)*matriz[1][2]
            +mapa.item(i+1,j-1)*matriz[2][0]+mapa.item(i+1,j)*matriz[2][1]+mapa.item(i+1,j+1)*matriz[2][2]
            mapa2.itemset((i,j),valor)
    # mapa2[0:size[0],0]=mapa[0:size[0],0]
    return mapa2
def gradi(mapa,i,j,x):
    s=mapa.shape
    p=choices.get(x)
    if i+p[0]<s[0] and i+p[0]>=0 and j+p[1]<s[1] and j+p[1]>=0:
        g=(mapa[i+p[0],j+p[1]]-mapa[i,j])/np.linalg.norm(p)
        g=abs(g)
    else:
        g=0
    return g

def canny1(mapa):
    #matriz=(1/16)*np.array([[1,2,1],[2,4,2],[1,2,1]])
    #mapa=convolucion(mapa,matriz)
    # plt.matshow(mapa,fignum=1,label="Strings")
    # plt.title('Mapa suavizado')
    # plt.colorbar()
    # plt.show()
    s=mapa.shape
    mapag=np.zeros(s,dtype=float)
    mapadir=np.zeros(s,dtype=int)
    for x in [-4,-3,-2,-1,1,2,3,4]:
        for i in range(0,s[0]):
            for j in range(0,s[1]):
                g=gradi(mapa,i,j,x)
                if abs(g)>abs(mapag[i,j]):
                    mapag[i,j]=abs(g)
                    mapadir[i,j]=x
    return mapag,mapadir
 #tamaño = (3,3)
    #coeficientes=[1, 1, 1, 1, 1, 1, 1, 1, 1]
    #   no es necesario poner la variable factor = 9 ya que la suma de los coeficientes da nueve para este caso
    #mapa = mapa.filter(ImageFilter.Kernel(tamaño, coeficientes))


# def gradi(mapa,i,j,p,s): 
# 		#Funcion que calcula el gradiente en un punto i,j de la matriz y en una direccion. p es la direccion sobre la que se calcula y el parametro s son las dimensiones de la matriz que vienen dadas como un control para no salirnos del punto
# 	if i+p[0]<s[0]and i+p[0]>=0 and j+p[1]<s[1] and j+p[1]>=0:
# 		g=(mapa[i+p[0],j+p[1]]-mapa[i,j])/np.linalg.norm(p)
# 		g=np.abs(g)
# 	else:
# 		g=0
# 	return g

# def canny1(mapa):
# 	s=mapa.shape
# 	mapag=np.zeros(s,dtype=float)
# 	mapadir=np.zeros(s,dtype=float)
# 	listadir=[-4,-3,-2,-1,1,2,3,4]
# 	for x in listadir:
# 		p=choices.get(x)
# 		for i in range(0,s[0]):
# 			for j in range(0,s[1]):
# 				g=gradi(mapa,i,j,p,s)
# 				if g>mapag[i,j]:
# 					mapa[i,j]=g
# 					mapadir[i,j]=x
# 	return mapag,mapadir
