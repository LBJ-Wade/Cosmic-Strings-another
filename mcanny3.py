
"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
CANNY 3 AND CANNY 4
ABSTRACT: We make constraints on gradients stregnths so that we can consider them as a part of an edge, based in thresholds. For points marked as in dounbt, cany4 decide whether they belong or not to an edge.
------------------------------------------------------------------------------
"""
import math
import healpy as hp
import numpy as np
from numpy import matrix
choices={-4:[-1,1], -3:[-1,0],-2:[-1,-1],-1:[0,-1],1:[0,1],2:[1,1],3:[1,0],4:[1,-1]}
def get_perpendicular(dl):
	choices2={-4:-2,-3:-1,-2:-4,-1:-3,1:3,2:4,3:1,4:2}
	dp=choices2.get(dl)
	return dp
""""Vemos si estan dentro de ciertas cotas con canny3"""
def canny3(localmax,localdir,gm,tc,tu,tl):
	s=localmax.shape
	mgr=np.zeros(s)
	mdir=np.zeros(s)
	for i in range(0,s[0]):
		for j in range(0,s[1]):
			if localmax[i,j]>=gm*tu and localmax[i,j]<=gm*tc:
				mgr[i,j]=1
				mdir[i,j]=localdir[i,j]
			if localmax[i,j]>=gm*tl and localmax[i,j]<gm*tu:
				mgr[i,j]=0.5
				mdir[i,j]=localdir[i,j]
	return mgr,mdir
"""Veamos si los 0.5 estan cerca de otros bordes con caracteristicas similares"""
def buscador(p,dent,dl,mgr,mdir,lista,flags):
	#dent es direccion de entrada, dl la del gradiente y lista la lista de posibles bordes, flags nos da su indicador 1 o 0.5
	listadir=[-4,-3,-2,-1,1,2,3,4]
	listadir.remove(dl)
	listadir.remove(-dl)
	if dent in listadir:
		listadir.remove(dent)
	for x in listadir:
		indicador=1
		d=choices.get(x)
		d0=d[0]
		d1=d[1]
		if p[0]+d0<mgr.shape[0] and p[1]+d1<mgr.shape[1] and p[0]+d0>=0 and p[1]+d1>=0:
			if mgr[p[0]+d0,p[1]+d1]==0.5: #Punto marcado como en duda
				if flags[lista.index([p[0]+d0,p[1]+d1])]==1:
					listadir2=[-4,-3,-2,-1,1,2,3,4]
					dp=get_perpendicular(dl)
					listadir2.remove(dp)
					listadir2.remove(-dp)
					if mdir[p[0]+d0,p[1]+d1] in listadir2:
						pos=lista.index([p[0]+d0,p[1]+d1])
						flags[pos]=0.5
						indicador=0
						break
			if mgr[p[0]+d0,p[1]+d1]==1:
				dp=get_perpendicular(dl)
				listadir2=[-4,-3,-2,-1,1,2,3,4]
				listadir2.remove(dp)
				listadir2.remove(-dp)
				if mdir[p[0]+d0,p[1]+d1] in listadir2:
					indicador=2
					break
	return indicador,d,d0,d1,flags





def canny4(mgr,mdir):
	lista=[]
	m=np.where(mgr!=0)
	for i in range(0,len(m[0])):
		lista.append([m[0][i],m[1][i]])
	flags=np.ones(len(lista))
	#flags se utiliza para ver que puntos han sido analizados
	for p in lista:
		if mgr[p[0],p[1]]==1 and flags[lista.index(p)]==1:
			dl=mdir[p[0],p[1]]
			listadir=[-4,-3,-2,-1,1,2,3,4]
			if dl in listadir:
				listadir.remove(dl)
				listadir.remove(-dl)
			for x in listadir:
				#Buscamos entre estas direcciones aquellas en las que podamos encontrar un gradiente en la direccion adecuada.
				indicador=0
				v=choices.get(x)
				v0=v[0]
				v1=v[1]
				dent=x
				if p[0]+v0<mgr.shape[0] and p[1]+v1<mgr.shape[1] and p[0]+v0>=0 and p[1]+v1>=0:
					aux=[]
					aux.append(p)
					if mgr[p[0]+v0,p[1]+v1]==0.5:
						#Si encontramos un punto en el que el gradiente valga 0.5
						listados=[-4,-3,-2,-1,1,2,3,4]
						dp=get_perpendicular(dl)
						if dp in listados:
							listados.remove(dp)
							listados.remove(-dp)
						#quitamos las direcciones perpendiculares al gradiente
						if mdir[p[0]+v0,p[1]+v1] in listados:
							#Si el gradiente de este punto es paralelo al gradiente del primer punto
							p2=p
							pos=lista.index([p[0]+v0,p[1]+v1])
							flags[pos]=0.5
							indicador=0
							while indicador==0:
								p2=[p2[0]+v0,p2[1]+v1]
								aux.append(p2)
								indicador,dent,v0,v1,flags=buscador(p2,dent,dl,mgr,mdir,lista,flags)
				if indicador==2:
					aux.append([p2[0]+v0,p2[1]+v1])
					for i in range(0,len(aux)):
						lugar=lista.index(aux[i])
						mdir[lista[lugar][0],lista[lugar][1]]=1
						flags[lugar]=0
					break
	for i in range(0,mgr.shape[0]):
		for j in range(0,mgr.shape[1]):
			if mgr[i,j]==0.5:
				mgr[i,j]=0
				mdir[i,j]=0
	return mgr,mdir

