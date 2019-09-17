"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
FISHER WITH CANNY GRADIENTS
ABSTRACT: Here we have funcitons to generate patches and taking que characteristic statistics, as well as function to obtaine the covariance matrix of the Fisher discriminant and the Fisher discriminant themselves (always for both null and alternative hypothesis). We also include a function sim2fisher to draw histogrmas and a fucntion which takes the previous to obtain statistic, fisher discriminant and simulations in once.
------------------------------------------------------------------------------
"""
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
from gaussian_filters import grad_filters
from histograms import fisher_histogram_wavelet
#from metodocanny1 import gradient
import scipy.stats as sps
import sys
import matplotlib 
matplotlib.rcParams.update({'font.size': 13})

#strings=hp.read_map('/home/maria/Escritorio/tfgfis/Codigo/simulations/STRINGS-smooth.fits')
#SMOOTHED CMB
#cmb=hp.read_map('/home/maria/Escritorio/tfgfis/Codigo/simulations/CMB-smooth.fits')
##GAUSSIAN NOISE
#noise=hp.read_map('/home/maria/Escritorio/tfgfis/Codigo/simulations/NOISE.fits')
#We calculate the statistics of the simulations using filters
def simulations_2048_filt(cmb,strings,noise,gu,change_noise,sim_ini,sim_end,R,bol_noise,bol_strings):
	r=len(R)
	nside=16
	npixangle=12*nside**2
	ipix=np.arange(12*nside**2)
	ipix=list(ipix)
	cmbnoise=change_noise*noise+cmb
	cmbstringsnoise=gu*strings+cmbnoise
	del cmb
	del strings
	del noise
	theta,fi=hp.pixelfunc.pix2ang(nside,ipix, nest=False, lonlat=True)
	if bol_strings==True:
		sizesim=sim_end-sim_ini
		datastringscmb=np.zeros([sizesim,r*3+1],dtype=float)
		for j in range(0,r):
			for i in range(0,sizesim):
				map1=hp.visufunc.cartview(cmbstringsnoise, xsize=105, lonra=[theta[i+sim_ini]-1.5,theta[i+sim_ini]+1.5], latra=[fi[i+sim_ini]-1.5,fi[i+sim_ini]+1.5],return_projected_map=True)
				plt.close()
				mapag,directions,mapax,mapay=grad_filters(map1,R[j])
				datastringscmb[i,0]=i+sim_ini
				#standar deviation of the map
				datastringscmb[i,1+3*j]=np.std(mapag)
				mapag2=mapag.flatten()
				#skewness
				datastringscmb[i,2+3*j]=sps.skew(mapag2)
				#kurtosis
				datastringscmb[i,3+3*j]=sps.kurtosis(mapag2)
		np.savetxt(os.getcwd()+'/statistics/filt/STATS-cmb'+str(change_noise)+'n'+str(gu)+'strings'+str(sim_ini)+'-'+str(sim_end)+'FILT.dat', datastringscmb, fmt='%.4e')
		cmbstringsnoise_stat=datastringscmb
	if bol_noise==True:
		datastringscmb=np.zeros([sizesim,r*3+1],dtype=float)
		for j in range(0,r):
			for i in range(0,sizesim):
				map1=hp.visufunc.cartview(cmbnoise, xsize=105, lonra=[theta[i+sim_ini]-1.5,theta[i+sim_ini]+1.5], latra=[fi[i+sim_ini]-1.5,fi[i+sim_ini]+1.5],return_projected_map=True)
				plt.close()
				mapag,directions,mapax,mapay=grad_filters(map1,R[j])
				datastringscmb[i,0]=i+sim_ini
				#standar deviation of the map
				datastringscmb[i,1+3*j]=np.std(mapag)
				mapag2=mapag.flatten()
				#skewness
				datastringscmb[i,2+3*j]=sps.skew(mapag2)
				#kurtosis
				datastringscmb[i,3+3*j]=sps.kurtosis(mapag2)
		np.savetxt(os.getcwd()+'/statistics/filt/STATS-cmb'+str(change_noise)+'n'+str(sim_ini)+'-'+str(sim_end)+'FILT.dat', datastringscmb, fmt='%.4e')
		cmbnoise_stat=datastringscmb
	if bol_noise==True and bol_strings==True:
		return cmbstringsnoise_stat,cmbnoise_stat
	elif bol_strings==True and bol_noise==False:
		return cmbstringsnoise_stat
	elif bol_noise==True and bol_strings==False:
		return cmbnoise_stat


#COVARIANCE MATRICES
def covmatrix(stats_cmbnoise,stats_cmbstringsnoise,gu,cn):
	s=stats_cmbnoise.shape[0]
	vectorh0=stats_cmbnoise[:,1:]
	vectorh1=stats_cmbstringsnoise[:,1:]
	#Mean vectors
	avh0=np.mean(vectorh0,0)
	avh1=np.mean(vectorh1,0)
	#Now we need the covariance matrix
	#step1
	differencesh0=vectorh0-avh0
	differencesh1=vectorh1-avh1
	#step2: Definition of covariance matrices
	covh0=np.zeros([int(len(avh0)),int(len(avh0))])
	covh1=np.zeros([int(len(avh0)),int(len(avh0))])
	for i in range(0,int(len(avh0))):
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

def fisher_filt(cmbnoise_stats,cmbstringsnoise_stats,gu,cn,sim_ini_fisher,sim_end,covh0,covh1):
	sim_ini=sim_ini_fisher
	cmbstringsnoise_stats=cmbstringsnoise_stats[:,1:]
	cmbnoise_stats=cmbnoise_stats[:,1:]
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
	np.savetxt(os.getcwd()+'/statistics/filt/Fisherh0'+str(cn)+'noise'+str(gu)+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'FILT.dat', Fisherh0, fmt='%.4e')
	np.savetxt(os.getcwd()+'/statistics/filt/Fisherh1'+str(cn)+'noise'+str(gu)+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'FILT.dat', Fisherh1, fmt='%.4e')
	return Fisherh0,Fisherh1

def sim2fisher(cmb,strings,noise,gu,change_noise,sim_ini,sim_end,sim_ini_fisher,R,nbins):
	cn=change_noise
	bol_noise=True
	bol_strings=True
	cmbstringsnoise_stat,cmbnoise_stat=simulations_2048_filt(cmb,strings,noise,gu,change_noise,sim_ini,sim_end,R,bol_noise,bol_strings)
	cmbnoise_statcov=cmbnoise_stat[0:sim_ini_fisher,:]
	cmbstringsnoise_statcov=cmbstringsnoise_stat[0:sim_ini_fisher,:]
	covh0,covh1=covmatrix(cmbnoise_statcov,cmbstringsnoise_statcov,gu,cn)
	Fisherh0,Fisherh1=fisher_filt(cmbnoise_stat,cmbstringsnoise_stat,gu,cn,sim_ini_fisher,sim_end,covh0,covh1)
	fisher_histogram_wavelet(Fisherh0,Fisherh1,nbins,gu,cn)
