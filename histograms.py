"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
HISTOGRAM FUNCTIONS
ABSTRACT: Here we include functions to generate histograms related to the Fisher discrimiants, for both canny gradient calculations and wavelet analysis. Note that the only difference between both is the way the Fisher statistics are stored. The methods stats2histo and draw_fisher_histogram_* are different in their inputs.
------------------------------------------------------------------------------
"""
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
# from fisherfilt import fisher_filt
from gaussian_filters import grad_filters
from metodocanny1 import gradient
import scipy.stats as sps
import matplotlib 
matplotlib.rcParams.update({'font.size': 13})

#Method to create histograms and save them
def fisher_histogram_wavelet(fisherh0,fisherh1,nbins,gu,cn):
	meanh0=np.mean(fisherh0)
	meanh1=np.mean(fisherh1)
	#fisher0,count0=np.histogram(fisherh0,bins=nbins,density=True)
	#fisher1,count1=np.histogram(fisherh1,bins=nbins,density=True)
	plt.title('FISHER DISCRIMINANT G\u03BC='+str(gu)+' Noise= '+str(cn)+' $\sigma_{CMB}$')
	#1/float(len(fisherh0))*
	plt.hist(fisherh0, label='h0-CMB+NOISE',bins=nbins,color='r',density=False,stacked=True)
	plt.axvline(x=meanh0,label='Mean h0',color='gold',linestyle='--')
	#1/float(len(fisherh0))*
	plt.hist(fisherh1, bins =nbins,label='h1-CMB+STRINGS+NOISE',color='c',density=False,stacked=True)
	plt.axvline(x=meanh1, label='Mean h1',color='olive',linestyle='--')
	plt.legend()
	plt.savefig(os.getcwd()+'/statistics/filt/Histograms/Fisher'+str(gu)+'str'+str(cn)+'noise.png')
	plt.xlabel('Fisher discriminant')
	plt.show()
	plt.clf()
	plt.close()

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
	for i in range(0,sim_end-sim_ini_fisher):
		Fisherh0[i]=np.dot(aux,cmbnoise_stats[i,:])
		Fisherh1[i]=np.dot(aux,cmbstringsnoise_stats[i,:])
	np.savetxt(os.getcwd()+'/statistics/filt/Fisherh0'+str(cn)+'noise'+str(gu)+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'FILT.dat', Fisherh0, fmt='%.4e')
	np.savetxt(os.getcwd()+'/statistics/filt/Fisherh1'+str(cn)+'noise'+str(gu)+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'FILT.dat', Fisherh1, fmt='%.4e')
	return Fisherh0,Fisherh1

def stats2histo(stats_cmbnoise,stats_cmbstringsnoise,sim_ini,sim_end,sim_ini_fisher,gu,cn,nbins):
	covh0,covh1=covmatrix(stats_cmbnoise[0:sim_ini_fisher,:],stats_cmbstringsnoise,gu,cn)
	covh0,covh1=covmatrix(stats_cmbnoise[0:sim_ini_fisher,:],stats_cmbstringsnoise,gu,cn)
	fisherh0,fisherh1=fisher_filt(cmbnoise_stats,cmbstringsnoise_stats,gu,cn,sim_ini_fisher,sim_end,covh0,covh1)
	fisher_histogram_wavelet(fisherh0,fisherh1,nbins,gu,cn)


def fisher_histogram_canny(fisherh0,fisherh1,nbins,gu,cn):
	meanh0=np.mean(fisherh0)
	meanh1=np.mean(fisherh1)
	#fisher0,count0=np.histogram(fisherh0,bins=nbins,density=True)
	#fisher1,count1=np.histogram(fisherh1,bins=nbins,density=True)
	plt.title('FISHER DISCRIMINANT G\u03BC'+str(gu)+' Noise= '+str(cn)+'$\sigma_{CMB}$')
	plt.hist(fisherh0, label='h0-CMB+NOISE',bins=nbins,color='r',density=True)
	plt.axvline(x=meanh0,label='Mean h0',color='gold',linestyle='--')
	plt.hist(fisherh1, bins =nbins,label='h1-CMB+STRINGS+NOISE',color='c',density=True)
	plt.axvline(x=meanh1, label='Mean h1',color='olive',linestyle='--')
	plt.legend()
	plt.savefig(os.getcwd()+'/statistics/canny/Histograms/Fisher'+str(gu)+'str'+str(cn)+'noise.png')
	# plt.show()
	plt.clf()
	plt.cla()
	plt.close()




def draw_fisher_histograms_wavelet(gu,cn,sim_ini_fisher,sim_end,nbins):
	for i in gu:
		for j in cn:
			Fisherh0=np.loadtxt(os.getcwd()+'/statistics/filt/Fisherh0'+str(round(j,9))+'noise'+str(round(i,9))+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'FILT.dat')
			Fisherh1=np.loadtxt(os.getcwd()+'/statistics/filt/Fisherh1'+str(round(j,9))+'noise'+str(round(i,9))+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'FILT.dat')
			fisher_histogram_wavelet(Fisherh0,Fisherh1,nbins,round(i,9),round(j,9))

def draw_fisher_histograms_canny(gu,cn,sim_ini_fisher,sim_end,nbins):
	for i in gu:
		for j in cn:
			Fisherh0=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh0'+str(round(j,9))+'noise'+str(round(i,9))+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'CANNY.dat')
			Fisherh1=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh1'+str(round(j,9))+'noise'+str(round(i,9))+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'CANNY.dat')
			fisher_histogram_canny(Fisherh0,Fisherh1,nbins,round(i,9),round(j,9))


# gu=[10*1e-7,5*1e-6,100*1e-7,3.75e-6,2.5e-6,3.125e-6,5.625e-6,4.375e-6,6.875e-6,1.25e-6,2.5e-6]
# list.sort(gu)
# nbins=20
# cn=[1]
# sim_ini_fisher=2000
# sim_end=3000
# draw_fisher_histograms_wavelet(gu,cn,sim_ini_fisher,sim_end,nbins)

# gu=1e-07
# cn=1
# nbins=125
# sim_ini=0
# sim_end=1000
# sim_ini_fisher=500
# cmbnoise_stats=np.loadtxt(os.getcwd()+'/statistics/filt/STATS-cmb1n1-1000FILT.dat')
# cmbstringsnoise_stats=np.loadtxt(os.getcwd()+'/statistics/filt/STATS-cmb1n1e-07strings1-1000FILT.dat')
# stats2histo(cmbnoise_stats,cmbstringsnoise_stats,sim_ini,sim_end,sim_ini_fisher,gu,cn,nbins)

#fisherh0=np.loadtxt(os.getcwd()+'/statistics/filt/Fisherh01noise1e-06strings500-1000FILT.dat')
# fisherh1=np.loadtxt(os.getcwd()+'/statistics/filt/Fisherh11noise1e-06strings500-1000FILT.dat')
# gu=1e-06
# cn=1
# nbins=150
# fisher_histogram_wavelet(fisherh0,fisherh1,nbins,gu,cn)

# fisherh0=np.loadtxt(os.getcwd()+'/statistics/filt/Fisherh01noise5e-06strings1500-2000FILT.dat')
# fisherh1=np.loadtxt(os.getcwd()+'/statistics/filt/Fisherh11noise5e-06strings1500-2000FILT.dat')
# gu=5e-06
# cn=1
# nbins=150
# fisher_histogram_wavelet(fisherh0,fisherh1,nbins,gu,cn)

# fisherh0=np.loadtxt(os.getcwd()+'/statistics/filt/Fisherh01noise1e-05strings500-1000FILT.dat')
# fisherh1=np.loadtxt(os.getcwd()+'/statistics/filt/Fisherh11noise1e-05strings500-1000FILT.dat')
# gu=1e-05
# cn=1
# nbins=150
# fisher_histogram_wavelet(fisherh0,fisherh1,nbins,gu,cn)

# fisherh0=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh01.0noise1e-05strings500-750CANNY.dat')
# fisherh1=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh11.0noise1e-05strings500-750CANNY.dat')
# gu=1e-05
# cn=1
# nbins=100
# fisher_histogram_canny(fisherh0,fisherh1,nbins,gu,cn)

# fisherh0=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh01.0noise1e-06strings1000-1500CANNY.dat')
# fisherh1=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh11.0noise1e-06strings1000-1500CANNY.dat')
# gu=1e-06
# cn=1
# nbins=100
# fisher_histogram_canny(fisherh0,fisherh1,nbins,gu,cn)


# fisherh0=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh01.0noise1e-07strings500-750CANNY.dat')
# fisherh1=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh11.0noise1e-07strings500-750CANNY.dat')
# gu=1e-07
# cn=1
# nbins=100
# fisher_histogram_canny(fisherh0,fisherh1,nbins,gu,cn)

# fisherh0=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh01.0noise5e-06strings1000-1500CANNY.dat')
# fisherh1=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh11.0noise5e-06strings1000-1500CANNY.dat')
# gu=5e-06
# cn=1
# nbins=100
# fisher_histogram_canny(fisherh0,fisherh1,nbins,gu,cn)