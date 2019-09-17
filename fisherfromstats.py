"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
FISHER FROM STATS
ABSTRACT: We calculate the Fisher discriminant values, histograms, etc.
Then we use this method to do all the histograms at once.
------------------------------------------------------------------------------
"""

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from time import time
import os
from gaussian_filters import grad_filters
from histograms import fisher_histogram_wavelet,draw_fisher_histograms_canny,fisher_histogram_canny
from fisherfilt import covmatrix,fisher_filt
#from metodocanny1 import gradient
import scipy.stats as sps
import sys
import matplotlib 

matplotlib.rcParams.update({'font.size': 13})


gu=5e-7
cn=0.25
change_noise=cn
sim_end=3000
sim_ini=0
sim_ini_fisher=2000
nbins=20

def fisher_from_stats(gu,cn,sim_ini,sim_ini_fisher,sim_end,nbins):
	change_noise=cn
	statscmb=np.loadtxt(os.getcwd()+'/statistics/filt/STATS-cmb'+str(change_noise)+'n'+str(sim_ini)+'-'+str(sim_end)+'FILT.dat')
	stats_cmbstringsnoise=np.loadtxt(os.getcwd()+'/statistics/filt/STATS-cmb'+str(change_noise)+'n'+str(gu)+'strings'+str(sim_ini)+'-'+str(sim_end)+'FILT.dat')
	covh0,covh1=covmatrix(statscmb[0:sim_ini_fisher,:],stats_cmbstringsnoise[0:sim_ini_fisher,:],gu,cn)
	fisherh0,fisherh1=fisher_filt(statscmb,stats_cmbstringsnoise,gu,cn,sim_ini_fisher,sim_end,covh0,covh1)
	fisher_histogram_wavelet(fisherh0,fisherh1,nbins,gu,cn)


# fisher_from_stats(gu,cn,sim_ini,sim_ini_fisher,sim_end,nbins)
# gu=[5e-7,1e-6,5e-6]
# nbins=20
# for i in range(0,3):
# 	fisherh1=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh1'+str(cn)+'noise'+str(gu[i])+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'CANNY.dat')
# 	fisherh0=np.loadtxt(os.getcwd()+'/statistics/canny/Fisherh0'+str(cn)+'noise'+str(gu[i])+'strings'+str(sim_ini_fisher)+'-'+str(sim_end)+'CANNY.dat')
# 	fisher_histogram_canny(fisherh0,fisherh1,nbins,gu[i],cn)

gulist=[5e-7,7.5e-7,8.75e-7,9.375e-7,9.68755e-7,1e-6,1.03125e-6,1.0625e-6,1.09375e-6,1.125e-6,1.25e-6,1.3e-6,1.4e-6]
for gu in gulist:
	fisher_from_stats(gu,cn,sim_ini,sim_ini_fisher,sim_end,nbins)