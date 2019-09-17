"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
CANNY 2 (LOCAL MAX)
ABSTRACT: Function which decides if a gradient is a local maximum, a condition to belong to an edge..
------------------------------------------------------------------------------
"""

import math
import healpy as hp
import numpy as np
from numpy import matrix
from mecanny1 import convolucion
choices={-4:[-1,1], -3:[-1,0],-2:[-1,-1],-1:[0,-1],1:[0,1],2:[1,1],3:[1,0],4:[1,-1]}
def canny2(mapag,mapadir):
	s=mapag.shape
	localmax=np.zeros(s)
	localdir=np.zeros(s)
	for i in range(1,s[0]-1):
		for j in range(1,s[1]-1):
			for x in [-1,-2,3,4]:
				if mapadir[i,j]==x or mapadir[i,j]==-x:
					p=choices.get(x)
					p0=p[0]
					p1=p[1]
					q0=-p[0]
					q1=-p[1]
					if mapag[i,j]>mapag[i+p0,j+p1] and mapag[i,j]>mapag[i+q0,j+q1]:
						localmax[i,j]=mapag[i,j]
						localdir[i,j]=abs(mapadir[i,j])
						break
					# elif mapadir[i,j]!=mapa
					elif mapag[i,j]==mapag[i+p0,j+p1]:
						mapag[i,j]=0
						break
						# localmax[i+p0,j+p0]=mapag[i+p0,j+p0]
						# localdir[i+p0,j+p0]=abs(mapadir[i+p0,j+p0])
	maximoglobal=np.max(localmax)
	return localmax,localdir,maximoglobal
