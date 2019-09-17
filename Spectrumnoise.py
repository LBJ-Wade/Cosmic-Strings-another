
"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
SPECTRUM ANALYSIS
ABSTRACT:
------------------------------------------------------------------------------
"""
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import time
#from cldlmaps import l_cl_dl
#from cmbmaps2 import smooth_map
matplotlib.rcParams.update({'font.size': 14})
def smooth_map(mapa,fwhm,lmax):
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
	return smoothedmap,wl[0:lmax]



#Planck data for CMB
datadl=np.loadtxt('/home/maria/Escritorio/tfgfis/Canny/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt',comments='#',skiprows=1)
lmax=2509
lclstrings=np.loadtxt(os.getcwd()+'/cl_model_strings.dat')
strings_l=lclstrings[0:lmax,0]
strings_cl=lclstrings[0:lmax,1]
#CMB map from the data
lcmb=datadl[:,0]
dl=datadl[:,1]
cmbcl=[]
for i in range(0,len(lcmb)):
	cmbcl.append(dl[i]/lcmb[i]/(lcmb[i]+1)*2*np.pi)
cmbcl=np.array(cmbcl)
l=np.zeros(len(lcmb)+2)
l[2:]=lcmb
lcmb=l
cl=np.zeros(len(cmbcl)+2)
cl[2:]=cmbcl
cmbcl=cl
nside=2048
cmbmap=hp.sphtfunc.synfast(cmbcl,nside, lmax=lmax, mmax=None, alm=False, pol=False, pixwin=False, fwhm=0.0, sigma=None, new=False, verbose=True)
cmbsmooth,wl=smooth_map(cmbmap,5,lmax,)
stdcmb=np.std(cmbsmooth)
sizemin=2509
# print(stdcmb)


stdcmb=101.04
wn=4*np.pi/(12*nside*nside)*stdcmb*stdcmb*0.25*0.25
print(wn)
gulist=[1e-8,5e-8,7e-8,1e-7,1.2e-7,1.4e-7,1.6e-7,1.8e-7,2e-7,5e-7]
# gulist=[1e-7]
detectionsp=[]
N=200
for j in range(0,len(gulist)):
	xilist=[]
	for i in range(0,N):
		gu=gulist[j]
		cmb_cl=np.loadtxt(os.getcwd()+'/statistics/gaussianity/CLmap'+str(i)+'.dat')
		stringsmap_cl=np.loadtxt(os.getcwd()+'/statistics/gaussianity/CLSTRINGSmap'+str(i)+'.dat')
		noisecl=np.loadtxt(os.getcwd()+'/statistics/gaussianity/CLNOISEsim'+str(i)+'.dat')*stdcmb**2*0.25**2
		sizemin=min([len(cmb_cl),len(strings_cl),len(wl),len(cmbcl),len(noisecl)])
		noisecl=noisecl[0:sizemin]
		cmb_cl=stringsmap_cl[0:sizemin]*gu*gu+cmb_cl[0:sizemin]+noisecl
		cosmicvar=(cmbcl*wl**2+strings_cl*gu**2+wn)**2
		cosmicvar=cosmicvar[0:sizemin]/(lcmb[0:sizemin]+0.5)
		cosmicvar=cosmicvar[0:sizemin]
		strings_cl=strings_cl[0:sizemin]
		cmbcl=cmbcl[0:sizemin]
		cmb_cl=cmb_cl[0:sizemin]
		wl=wl[0:sizemin]
		strings_cmbsim_cl=strings_cl*cmb_cl
		strings_strings_cl=strings_cl*strings_cl
		strings_cmb_cl=strings_cl*cmbcl*(wl**2)
		strings_noise_cl=strings_cl*wn
		sum_strings_noise=sum(strings_noise_cl/cosmicvar)
		sum_strings_strings=sum(strings_strings_cl[2:sizemin]/cosmicvar[2:sizemin])
		# print(sum_strings_strings.shape)
		sum_strings_cmbsim=sum(strings_cmbsim_cl[2:sizemin]/cosmicvar[2:sizemin])
		# print(sum_strings_cmbsim.shape)
		sum_strings_cmb=sum(strings_cmb_cl[2:sizemin]/cosmicvar[2:sizemin])
		# print(sum_strings_cmb.shape)
		xi=(sum_strings_cmbsim-sum_strings_cmb-sum_strings_noise)/sum_strings_strings
		# print(1/np.sqrt(sum_strings_strings))
		#gu=np.sqrt(np.abs(xi))
		xilist.append(xi)
	p95=(np.percentile(xilist,95))
	# np.sqrt(np.mean())
	mean=np.mean(xilist)
	xilist=np.array(xilist)
	stdgu=np.std(np.sqrt(xilist[xilist>0]))
	std=np.std(xilist)
	stdgu=std/np.sqrt(mean)/2
	# xilist.sort()
	if mean-2*std<0:
		detection=0
	else:
		detection=1
	detectionsp.append([gulist[j],mean,p95,std,np.sqrt(mean),np.sqrt(p95),stdgu,detection])
	plt.hist(xilist,bins=12,color='r',density=True)
	plt.xlabel('\u03BE')
	plt.title('$G\mu$= '+str(gulist[j]))
	plt.axvline(p95,label='95-percentile',color='olive',linestyle='--')
	# r=xilist.index(p95)
	# print(r)
	# plt.show()
	plt.cla()
	plt.close()
detectionsp=np.array(detectionsp)
print(detectionsp)
np.savetxt(os.getcwd()+'/statistics/gaussianity/xilist.dat',xilist)
detectionsp=np.array(detectionsp)
print(detectionsp)
np.savetxt(os.getcwd()+'/statistics/gaussianity/1spectrum-detection-withnoise.dat',detectionsp)
# np.savetxt(os.getcwd()+'/statistics/gaussianity/AAspectrum-nonoise.dat',xilist)
datos=detectionsp
detected=[]
notdetected=[]
detectedx=[]
notdetectedx=[]
errordet=[]
for i in range(0,len(datos[:,0])):
	if datos[i,len(datos[1,:])-1]==1:
		detected.append(datos[i,4])
		detectedx.append(datos[i,0])
		errordet.append(datos[i,6])
	else:
		notdetected.append(datos[i,5])
		notdetectedx.append(datos[i,0])
detected=np.array(detected)
detectedx=np.array(detectedx)
notdetected=np.array(notdetected)
notdetectedx=np.array(notdetectedx)
errordet=np.array(errordet)
# print(detected)
# print(detectedx)
# print(errordet)
plt.plot(detectedx,detected,'-',color='b')
plt.plot(notdetectedx,notdetected,linestyle='',marker='v',color='c')
plt.xlabel('G$\mu$')
plt.ylabel('G$\mu$ (estimated)')
# plt.xscale('log')
# plt.yscale('log')
plt.errorbar(detectedx,detected,yerr=errordet,xerr=None,fmt='', ecolor=None,elinewidth=None, capsize=None,barsabove=True, lolims=False, uplims=False,xlolims=False, xuplims=False, errorevery=1,capthick=None)
plt.title('Detection- $\chi^2$ test')
#plt.errorbar(detectedx, detected, yerr=errordet,fmt='', ecolor='g', capthick=2)
plt.show()