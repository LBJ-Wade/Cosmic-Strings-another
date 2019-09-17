"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
FISHER WITH CANNY GRADIENTS
ABSTRACT: Here we have funcitons to generate patches and taking que characteristic statistics, as well as function to obtaine the covariance matrix of the Fisher discriminant and the Fisher discriminant themselves (always for both null and alternative hypothesis). We also include a function to draw histogrmas and a fucntion which takes the previous to obtain statistic, fisher discriminant and simulations in once.
------------------------------------------------------------------------------
"""

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from time
import time
import os
from mcannybueno import canny
from metodocanny1 import gradient
import scipy.stats as sps
#FIRST THE STRINGS MAP IS IMPORTED
strings=hp.read_map(os.getcwd()+'/simulations/STRINGS-smooth.fits')
#SMOOTHED CMB
cmb=hp.read_map(os.getcwd()+'/simulations/CMB-smooth.fits')
#GAUSSIAN NOISE
noise=hp.read_map(os.getcwd()+'/simulations/NOISE.fits')
#Method which takes just some simulations
def simulations_2048_canny(cmb,strings,noise,gu,change_noise,sim_ini,sim_end,bol_noise,bol_strings):
	nside=16
	npixangle=12*nside**2
	ipix=np.arange(12*nside**2)
	ipix=list(ipix)
	theta,fi=hp.pixelfunc.pix2ang(nside,ipix, nest=False, lonlat=True)
	if bol_strings==True:
		cmbstringsnoise=gu*strings+change_noise*noise+cmb
		sizesim=sim_end-sim_ini
		datastringscmb=np.zeros([sizesim,4],dtype=float)
		for i in range(0,sizesim):
			map1=hp.visufunc.cartview(cmbstringsnoise, xsize=105, lonra=[theta[i+sim_ini]-1.5,theta[i+sim_ini]+1.5], latra=[fi[i+sim_ini]-1.5,fi[i+sim_ini]+1.5],return_projected_map=True)
			plt.close()
			mapag=gradient(map1)
			datastringscmb[i,0]=i+sim_ini
			#standar deviation of the map
			datastringscmb[i,1]=np.std(mapag)
			mapag2=mapag.flatten()
			#skewness
			datastringscmb[i,2]=sps.skew(mapag2)
			#kurtosis
			datastringscmb[i,3]=sps.kurtosis(mapag2)
		np.savetxt(os.getcwd()+'/statistics/canny/STATS-cmb'+str(change_noise)+'n'+str(gu)+'strings'+str(sim_ini)+'-'+str(sim_end)+'CANNY.dat', datastringscmb, fmt='%.4e')
		cmbstringsnoise_stat=datastringscmb
	if bol_noise==True:
		cmbnoise=change_noise*noise+cmb
		datastringscmb=np.zeros([sizesim,4],dtype=float)
		for i in range(0,sizesim):
			map1=hp.visufunc.cartview(cmbnoise,  xsize=105, lonra=[theta[i+sim_ini]-1.5,theta[i+sim_ini]+1.5], latra=[fi[i+sim_ini]-1.5,fi[i+sim_ini]+1.5],return_projected_map=True)
			plt.close()
			mapag=gradient(map1)
			#Once we have gradient maps, we can take some statistic to compare.
			#number of patch
			datastringscmb[i,0]=i
			#standar deviation of the map
			datastringscmb[i,1]=np.std(mapag)
			mapag2=mapag.flatten()
			#skewness
			datastringscmb[i,2]=sps.skew(mapag2)
			#kurtosis
			datastringscmb[i,3]=sps.kurtosis(mapag2)
		#We save the data for just noise and cmb
		np.savetxt(os.getcwd()+'/statistics/canny/STATS-cmb'+str(change_noise)+'noisecmbSTD'+str(sim_ini)+'-'+str(sim_end)+'CANNY.dat', datastringscmb, fmt='%.4e')
		cmbnoise_stat=datastringscmb
	if bol_noise==True and bol_strings==True:
		return cmbstringsnoise_stat,cmbnoise_stat
	elif bol_strings==True and bol_noise==False:
		return cmbstringsnoise_stat
	elif bol_noise==True and bol_strings==False:
		return cmbnoise_stat

def covmatrix(stats_cmbnoise,stats_cmbstringsnoise,gu,cn):
	s=stats_cmbnoise.shape[0]
	vectorh0=stats_cmbnoise[:,1:4]
	vectorh1=stats_cmbstringsnoise[:,1:4]
	#Mean vectors
	avh0=np.mean(vectorh0,0)
	avh1=np.mean(vectorh1,0)
	#Now we need the covariance matrix
	#step1
	differencesh0=vectorh0-avh0
	differencesh1=vectorh1-avh1
	#step2: Definition of covariance matrices
	covh0=np.zeros([3,3])
	covh1=np.zeros([3,3])
	for i in range(0,3):
		for j in range(0,i+1):
			sumah0=0
			sumah1=0
			for k in range(0,s):
				sumah0=sumah0+differencesh0[k,i]*differencesh0[k,j]
				sumah1=sumah1+differencesh1[k,i]*differencesh1[k,j]
			covh0[i,j]=1/s*sumah0
			covh1[i,j]=1/s*sumah1
			covh0[j,i]=covh0[i,j]
			covh1[j,i]=covh1[i,j]
	return covh0,covh1

def fisher_canny(cmbnoise_stats,cmbstringsnoise_stats,gu,cn,sim_ini_fisher,sim_end,covh0,covh1):
	sim_ini=sim_ini_fisher
	cmbstringsnoise_stats=cmbstringsnoise_stats[:,1:4]
	cmbnoise_stats=cmbnoise_stats[:,1:4]
	Fisherh0=np.zeros([sim_end-sim_ini,1])
	Fisherh1=np.zeros([sim_end-sim_ini,1])
	avh1=np.mean(cmbstringsnoise_stats,0)
	avh0=np.mean(cmbnoise_stats,0)
	difh0h1=avh0-avh1
	Winv=np.linalg.inv(covh0+covh1)
	diff=np.transpose(difh0h1)
	aux=np.dot(diff,Winv)
	for i in range(sim_ini_fisher,sim_end):
		Fisherh0[i-sim_ini_fisher]=np.dot(aux,cmbnoise_stats[i,:])
		Fisherh1[i-sim_ini_fisher]=np.dot(aux,cmbstringsnoise_stats[i,:])
	np.savetxt(os.getcwd()+'/statistics/canny/Fisherh0'+str(cn)+'noise'+str(gu)+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'CANNY.dat', Fisherh0, fmt='%.4e')
	np.savetxt(os.getcwd()+'/statistics/canny/Fisherh1'+str(cn)+'noise'+str(gu)+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'CANNY.dat', Fisherh1, fmt='%.4e')
	return Fisherh0,Fisherh1


def fisher_histogram_canny(fisherh0,fisherh1,nbins,gu,cn):
	meanh0=np.mean(fisherh0)
	meanh1=np.mean(fisherh1)
	#fisher0,count0=np.histogram(fisherh0,bins=nbins,density=True)
	#fisher1,count1=np.histogram(fisherh1,bins=nbins,density=True)
	plt.title('FISHER DISCRIMINANT '+str(gu)+' strings '+str(cn)+' noise')
	plt.hist(fisherh0, label='h0-CMB+NOISE',bins=nbins,color='r',density=True)
	plt.axvline(x=meanh0,label='Mean h0',color='gold',linestyle='--')
	plt.hist(fisherh1, bins =nbins,label='h1-CMB+STRINGS+NOISE',color='c',density=True)
	plt.axvline(x=meanh1, label='Mean h1',color='olive',linestyle='--')
	plt.legend()
	plt.savefig(os.getcwd()+'/statistics/canny/Histograms/Fisher'+str(gu)+'str'+str(cn)+'noise.png')
	plt.show()
	plt.clf()
	plt.close()

def sim2fisher_canny(cmb,strings,noise,gu,change_noise,sim_ini,sim_end,sim_ini_fisher,nbins):
	cn=change_noise
	bol_noise=True
	bol_strings=True
	cmbstringsnoise_stat,cmbnoise_stat=simulations_2048_canny(cmb,strings,noise,gu,change_noise,sim_ini,sim_end,bol_noise,bol_strings)
	covh0,covh1=covmatrix(cmbnoise_stat[0:sim_ini_fisher,:],cmbstringsnoise_stat[0:sim_ini_fisher,:],gu,cn)
	Fisherh0,Fisherh1=fisher_canny(cmbnoise_stat,cmbstringsnoise_stat,gu,cn,sim_ini_fisher,sim_end,covh0,covh1)
	fisher_histogram_canny(Fisherh0,Fisherh1,nbins,gu,cn)