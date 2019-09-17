"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
CANNY
ABSTRACT: We take all the Canny methods into one
------------------------------------------------------------------------------
"""

import math
import healpy as hp
import numpy as np
from numpy import matrix
import matplotlib.pyplot as plt
from mcanny3 import buscador,canny3,canny4,get_perpendicular
from mcanny1 import canny1
from mecanny22 import canny2
from mcanny5 import canny5
choices={-4:[-1,1], -3:[-1,0],-2:[-1,-1],-1:[0,-1],1:[0,1],2:[1,1],3:[1,0],4:[1,-1]}
def canny(mapa,tc,tu,tl,gm):
	plt.matshow(mapa,fignum=1,label="mapa inicial")
	plt.title('Mapa de los valores iniciales')
	plt.colorbar()
	plt.show()
	mapag,mapadir=canny1(mapa)
	plt.matshow(mapag,fignum=2,label="mapa de gradientes")
	plt.title('Mapa de los gradientes')
	plt.colorbar()
	plt.show()
	localmax,localdir,maximoglobal=canny2(mapa,mapag,mapadir)
	plt.matshow(localmax,fignum=2,label="mapa de gradientes")
	plt.title('Mapa de los gradientes máximos en la dirección del gradiente')
	plt.colorbar()
	plt.show()
	mgr,mdir=canny3(localmax,localdir,gm,tc,tu,tl)
	mgr,mdir=canny4(mgr,mdir)
	plt.matshow(mgr,fignum=4,label="mapa binario")
	plt.title('Mapa de edges')
	plt.colorbar()
	plt.show()
	plt.matshow(mdir,fignum=5,label="mapa de direcciones")
	plt.title('Mapa de las direcciones')
	plt.colorbar()
	plt.show()
	listaedges=canny5(mgr,mdir)
	# longitudes=[]
	# for i in range(0,len(listaedges)):
	# 	longitudes.append(len(listaedges[i]))
	# if len(longitudes)>=1:
	# 	lmax=max(longitudes)
	# 	lmin=min(longitudes)
	# 	ndiv=lmax-lmin
	# 	if ndiv==0:
	# 		ndiv=1
	# else:
	# 	ndiv=1
	# plt.hist(longitudes,ndiv)
	# plt.xlabel('Longitud de los bordes')
	# plt.ylabel('Frecuencia')
	# plt.show()
	return listaedges,mgr,mdir,mapag

#Ejemplo



# mapita=np.zeros((45,45))
# for i in range(0,23):
# 	mapita[4,4+i]=4*i
# 	mapita[9+i,16]=6
# tu=2
# tc=80
# tl=1
# gm=1
# listaedges,mapabin,mapadir=canny(mapita,tc,tu,tl,gm)
# print listaedges
#mapi = hp.read_map('map_synfast_seed12345.fits')
#mapar=hp.visufunc.cartview(mapi,return_projected_map=true)
#mapabin,mapadir,listaedges=canny(mapar,tc,tu,tl,gm)