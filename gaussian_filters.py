"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
WAVELET FUNCTION
ABSTRACT: We present the gaussian filters and fucntions that will be used in the analysis. We also do an exaple fo prove that the code is working properly.
------------------------------------------------------------------------------
"""
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as spsi
#The objective of this method is to create a gaussian filter of size nxn and scale of R pixels
def gaussian_filter(n,R):
	#FOR n ODD
	filt=np.zeros([n,n],dtype='float')
	fx=np.zeros([n,n])
	fy=np.zeros([n,n])
	for i in range(0,n):
		x = (i - (n-1)/2)/R
		for j in range(0,n):
			y = (j - (n-1)/2)/R
			filt[i,j]=np.exp(-(x**2 + y**2)/2)
			fx[i,j]=-x*filt[i,j]
			fy[i,j]=-y*filt[i,j]
	return filt,fx,fy
#The objective of this method is to calculate the fft of the map and convolve with the filters (direction x and y)
def filter2map(mapa,R):
	#The map has size nxn
	n=mapa.shape[0]
	filt,fx,fy=gaussian_filter(n,R)
	mapax = spsi.convolve2d(mapa, fx,mode='same')
	mapay=spsi.convolve2d(mapa,fy,mode='same')
	# for i in range(0,n):
	#   for j in range(0,n):
	#       if np.abs(mapax[i,j])<1e-16:
	#           mapax[i,j]=0
	#       if np.abs(mapay[i,j])<1e-16:
	#           mapay[i,j]=0
	return fx,fy,mapax,mapay
#This function gives us the direction of the gradient
def direction_filters(mapax,mapay):
	n=mapay.shape[0]
	directions=np.zeros([n,n])
	for i in range(0,n):
		for j in range(0,n):
			if np.abs(mapay[i,j]-mapax[i,j])<1e-15 and np.abs(mapax[i,j])<=1e-15:
				directions[i,j]=2*np.pi
			else:
				if mapax[i,j]>=0 and mapay[i,j]>=0:
					if mapax[i,j]>1e-16:
						directions[i,j]=np.arctan(mapay[i,j]/mapax[i,j])
					else:
						directions[i,j]=np.pi/2
				elif mapax[i,j]>=0 and mapay[i,j]<0:
					if mapax[i,j]>1e-16:
						directions[i,j]=-np.arctan(np.abs(mapay[i,j])/np.abs(mapax[i,j]))+2*np.pi
					else:
						directions[i,j]=3*np.pi/2
				elif mapax[i,j]<0 and mapay[i,j]<0:
					directions[i,j]=np.arctan(np.abs(mapay[i,j])/np.abs(mapax[i,j]))+np.pi
				elif mapay[i,j]>=0 and mapax[i,j]<0:
					directions[i,j]=-np.arctan(np.abs(mapay[i,j])/np.abs(mapax[i,j]))+np.pi
	return directions
#This method gives the gradient in the direction where it is maximum
def max_grad_filtered(mapax,mapay,directions):
	n=mapay.shape[0]
	maxgrad=np.zeros([n,n])
	for i in range(0,n):
		for j in range(0,n):
			if directions[i,j]!=2*np.pi:
				maxgrad[i,j]=np.cos(directions[i,j])*mapax[i,j]+np.sin(directions[i,j])*mapay[i,j]
			else:
				maxgrad[i,j]=0
	return maxgrad
#This final method returns the maximum gradient and directions
def grad_filters(mapa,R):
	fx,fy,mapax,mapay=filter2map(mapa,R)
	directions=direction_filters(mapax,mapay)
	maxgrad=max_grad_filtered(mapax,mapay,directions)
	# maxgrad2=max_grad_filtered(mapax,mapay,directions+np.pi/2)
	return maxgrad,directions,mapax,mapay

#examples of filters
n=21
R=[0.01,0.5,1,3,5]
b=np.zeros([n,n])
b[8:17,12]=np.ones(9)
plt.matshow(b)
plt.show()
plt.cla()
plt.close()

for i in range(0,len(R)):
	fx,fy,mapx,mapy=filter2map(b,R[i])
	plt.matshow(mapx)
	plt.show()
	plt.cla()
	plt.close()
	plt.matshow(mapy)
	plt.show()
	plt.cla()
	plt.close()
	plt.matshow(fx)
	plt.title('Scale'+str(R[i]))
	plt.show()
	plt.cla()
	plt.close()
	plt.matshow(fy)
	plt.title('Scale'+str(R[i]))
	plt.show()
	plt.cla()
	plt.close()