"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
GENERATION OF 300 MAPS OF STRINGS AND CMB
ABSTRACT: We generate map froms models and from them, we extract the angular power spectra. Both CMB and Strings are smoothed.
------------------------------------------------------------------------------
"""

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
#from cldlmaps import l_cl_dl
#from cmbmaps2 import smooth_map
matplotlib.rcParams.update({'font.size': 14})

N=0
N2=10

nside=2048
lmax=2509
npix=12*nside*nside

def l_cl(mapa,lmax):
	"""
	map
	lmax
	This routine creates l, cl and dl for a given map when the maximum l is given
	"""
	mapcl=hp.sphtfunc.anafast(mapa, map2=None, nspec=None, lmax=lmax, mmax=lmax, iter=3, alm=False, pol=False, use_weights=False, datapath=None, gal_cut=0, use_pixel_weights=False)
	l=[]
	dl=[]
	for i in range(0,len(mapcl)):
		l.append(i)
		dl.append(mapcl[i]*l[i]*(l[i]+1)/2/np.pi)
	l=np.array(l)
	return l,mapcl
def smooth_map(mapa,fwhm,lmax):
	"""map:Healpy map
	fwhm: fwhm in arcmins
	lmax: maximum l studied
	name: Name and route for the maps generated
	We do three steps
	STEP 1: Getting the window for 5'
	STEP 2: Getting the gaussian filter
	STEP 3: Taking both window functions and composing them
	STEP 4: Smoothing the map
	"""
	fwhm=fwhm*1/10800*np.pi
	#STEP 1
	gl=hp.sphtfunc.gauss_beam(fwhm, lmax=lmax, pol=False)
	#STEP 2
	nside=hp.pixelfunc.get_nside(mapa)
	pl=hp.sphtfunc.pixwin(nside, pol=False, lmax=lmax)
	#STEP 3
	wl=gl*pl #Element*element
	#STEP 4
	smoothedmap=hp.sphtfunc.smoothing(mapa, fwhm=0.0, sigma=None, beam_window=wl, pol=False, iter=3, lmax=lmax, mmax=lmax, use_weights=False, use_pixel_weights=True, datapath=None, verbose=True)
	return smoothedmap,wl[0:lmax+1]
 datadl=np.loadtxt('/home/maria/Escritorio/tfgfis/Canny/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt',comments='#',skiprows=1)
strings=hp.read_map('/home/maria/Documentos/simulaciones/map1n_allz_rtaapixlw_2048_1.fits')

lcmb=datadl[:,0]
dl=datadl[:,1]
cmbcl=[]
for i in range(0,len(lcmb)):
	cmbcl.append(dl[i]/lcmb[i]/(lcmb[i]+1)*2*np.pi)cmb_l,cmb_cl=l_cl(cmbsmooth,lmax)
l=np.zeros(len(lcmb)+2)
l[2:]=lcmb
lcmb=l
cl=np.zeros(len(cmbcl)+2)
cl[2:]=cmbcl
cmbcl=cl

for i in range(N,N+N2):
	cmbmap=hp.sphtfunc.synfast(cmbcl,nside, lmax=lmax, mmax=None, alm=False, pol=False, pixwin=False, fwhm=0.0, sigma=None, new=False, verbose=True)
	cmbsmooth,wl=smooth_map(cmbmap,5,lmax)
	np.savetxt(os.getcwd()+'/CLCMBsim'+str(i)+'.dat',noise_cl)
	stdcmb=np.std(cmbsmooth)
	del(cmbmap)
	cmb_l,cmb_cl=l_cl(cmbsmooth,lmax)
	strings_smooth=hp.sphtfunc.synfast(cmbcl,nside, lmax=lmax, mmax=None, alm=False, pol=False, pixwin=False, fwhm=0.0, sigma=None, new=False, verbose=True)
	strings_smooth,wl=smooth_map(strings,5,lmax,)
	cmb_l,cmb_cl=l_cl(cmbsmooth,lmax)
	np.savetxt(os.getcwd()+'/CLSTRINGSmap'+str(i)+'.dat')
	noise=np.random.randn(12*nside*nside)
	l,noise_cl=l_cl(noise,lmax-2)
	np.savetxt(os.getcwd()+'/CLNOISEsim'+str(i)+'.dat',noise_cl)

