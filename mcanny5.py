"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
CANNY5
ABSTRACT: This mehtod searches for edges, taking in account the directionality of the gradients. Finally it return the lengths of the edges.
------------------------------------------------------------------------------
"""

import math
import healpy as hp
import numpy as np
from numpy import matrix
import matplotlib.pyplot as plt
from mcanny3 import get_perpendicular
from mcanny3 import buscador
choices={-4:[-1,1], -3:[-1,0],-2:[-1,-1],-1:[0,-1],1:[0,1],2:[1,1],3:[1,0],4:[1,-1]}
def buscador_en_edge(p_inicial,dl,d_entrada,edge,flags,l,l2,lista):
	#l es mgr y l2 es mdir
	indicador=1
	a=0 #control para no entrar en bucle
	while indicador==1:
		listadir=[-4,-3,-2,-1,1,2,3,4]
		listadir.remove(dl)
		listadir.remove(-dl)
		if d_entrada in listadir:
			listadir.remove(d_entrada)
		for x in listadir:
			indfor=1
			d=choices.get(x)
			d0=d[0]
			d1=d[1]
			if p_inicial[0]+d0<l.shape[0] and p_inicial[1]+d1<l.shape[1] and p_inicial[0]+d0>=0 and p_inicial[1]+d1>=0:
				#si estamos dentro de los putnos analizables
				if [p_inicial[0]+d0,p_inicial[1]+d1] in lista:
					#si el punto esta en la lista de puntos con gradiente significativo
					if flags[lista.index([p_inicial[0]+d0,p_inicial[1]+d1])]==1:
						#si el punto aun no ha sido analizado
						dp=get_perpendicular(dl)
						listadir2=[-4,-3,-2,-1,1,2,3,4]
						listadir2.remove(dp)
						listadir2.remove(-dp)
						if l2[p_inicial[0]+d0,p_inicial[1]+d1] in listadir2:
							#si la direccion es paralela o casi paralela anadimos el punto a la lista del borde
							###edge.append([p_inicial[0]+d0,p_inicial[1]+d1])
							###flags[lista.index([p_inicial[0]+d0,p_inicial[1]+d1])]=0
							#actualizamos el punto y ponemos a cero los indicadores de flags (del analisis de puntos)
							p_inicial=[p_inicial[0]+d0,p_inicial[1]+d1]
							d_entrada=x
							#indicador=0
							break
			indfor=indfor+1
			if indfor==len(listadir)-1:
				indicador=0
		a=a+1
		if a==len(lista):
			indicador=0
		# a=a+1 #este otro indicador sirve para no repetir puntos
		# if a>len(listadir):
		# 	indicador=0
	return edge,flags,indicador,d0,d1



def canny5(mgr,mdir):
	#metodo contador de bordes
	lista=[]
	m=np.where(mgr!=0)
	#anadimos los puntos donde no es cero el gradiente o la direccion
	for i in range(0,len(m[0])):
		lista.append([m[0][i],m[1][i]])
	flags=np.ones(len(lista))
	cont=0
	listaedges=[]
	for p in lista:
		#bucle sobre cada punto en la lista
		if flags[lista.index(p)]==1:
			#si el punto no ha sido contado ya
			edge=[]#comienza un borde 
			edge.append(p)
			flags[lista.index(p)]=0
			dl=mdir[p[0],p[1]]#direccion del gradiente
			listadir=[-4,-3,-2,-1,1,2,3,4]
			listadir.remove(dl)
			listadir.remove(-dl)
			for x in listadir:
				#bucle sobre las posibles direcciones en las que movernos que no se correspondan a las ya analizadas
				d=choices.get(x)#vector direccion
				d0=d[0]
				d1=d[1]
				if(p[0]+d0)<mgr.shape[0] and (p[1]+d1)<mgr.shape[1] and (p[0]+d0)>=0 and (p[1]+d1)>=0: #si esta dentro del mapa
					if [p[0]+d0,p[1]+d1] in lista:
						#si el punto esta en la lista de puntos que analizar
						if flags[lista.index([p[0]+d0,p[1]+d1])]==1: #si el punto siguiente esta marcado con un uno
							dp=get_perpendicular(dl)
							listadir2=[-4,-3,-2,-1,1,2,3,4]
							listadir2.remove(dp)
							listadir2.remove(-dp)
							if mdir[p[0]+d0,p[1]+d1] in listadir2:
								#si la direccion del gradiente esta entre las paralelas o casi paralelas la direccion del gradiente del primer punto
								edge.append([p[0]+d0,p[1]+d1])
								flags[lista.index([p[0]+d0,p[1]+d1])]=0
								#en cuanto anadimos un punto a un edge se pone a cero flags
								indicador=0
								# controlw=1
								p2=[p[0]+d0,p[1]+d1]
								edge,flags,indicador,d0,d1=buscador_en_edge(p2,mdir[p2[0],p2[1]],x,edge,flags,mgr,mdir,lista)

								# while indicador==0:
								# 	p2=[p[0]+d0,p[1]+d1]#actualizamos el punto en cada iteracion
								# 	edge,flags,indicador,d0,d1=buscador_en_edge(p2,mdir[p2[0],p2[1]],x,edge,flags,mgr,mdir,lista)
								# 	controlw=controlw+1
								# 	if controlw==len(lista):
								# 		indicador=3 #Si se han hecho demasiadas iteraciones salimos del while
			listaedges.append(edge)
	longitudes=[]
	# colorplot=[]
	for i in range(0,len(listaedges)):
		longitudes.append(len(listaedges[i]))
		# colorplot.append(i)
	if len(longitudes)>=1:
		lmax=max(longitudes)
		lmin=min(longitudes)
		ndiv=lmax-lmin
		if ndiv==0:
			ndiv=1
	else:
		ndiv=1
	plt.hist(longitudes,ndiv)
	plt.xlabel('Longitud de los bordes')
	plt.ylabel('Frecuencia')
	plt.show()
	color=1
	mapaedges=np.zeros(mgr.shape)
	for edge in listaedges:
		for p in edge:
			mapaedges[p[0],p[1]]=color
		color=color+1
	plt.matshow(mapaedges,fignum=6,label="edges")
	plt.title('Edges')
	plt.colorbar()
	plt.show()


	return listaedges