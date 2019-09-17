""" Fisher discriminant for all scales and for all parameteres needed"""
"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
CHI2 TEST
ABSTRACT: Calculation of patches, applying gradient analysis, either sim2fisher or sim2fisher_canny. We save in files.dat the statiscis and we also save into files the Fisher discriminant for both H0 and H1, so that we will do a psoterior analysis in fisheranalysis.py.
------------------------------------------------------------------------------
"""
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
import multiprocessing as mp
from gaussian_filters import grad_filters
#from metodocanny1 import gradient
import scipy.stats as sps
from fisherfilt import simulations_2048_filt,covmatrix,fisher_filt,sim2fisher

stringsroute=os.getcwd()+'/simulations/STRINGS-smooth.fits'
strings=hp.read_map(stringsroute)
del(stringsroute)

#SMOOTHED CMB
start=time()
cmbroute=os.getcwd()+'/simulations/CMB-smooth.fits'
cmb=hp.read_map(cmbroute)
del(cmbroute)
#GAUSSIAN NOISE
noiseroute=os.getcwd()+'/simulations/NOISE.fits'
noise=hp.read_map(noiseroute)
del(noiseroute)
R=np.array([0.25,0.5,0.75,1,2,3])
change_noise=0.25
cn=change_noise
sim_ini=0
sim_end=50
sim_ini_fisher=25
gu=[1e-7,1e-6]
nbins=6
l=[cmb,strings,noise,gu,change_noise,sim_ini,sim_end,R]
if __name__=='__main__':
	with mp.Pool(processes=1) as pool:
		results=[pool.apply_async(sim2fisher,(cmb,strings,noise,i,change_noise,sim_ini,sim_end,sim_ini_fisher,R,nbins)) for i in gu]
		results=[p.get() for p in results]
finish=time()
print(finish-start)
#sim2fisher(cmb,strings,noise,gu[0],change_noise,sim_ini,sim_end,sim_ini_fisher,R)