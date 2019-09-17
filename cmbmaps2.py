"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
CMB MAPS
ABSTRACT:This method is used only to create the figures of the CMB, Strings and Noise maps, as well as angular power spectrum plots of them.
------------------------------------------------------------------------------
"""
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

#WE PLOT THE CL-L OF THE FITTING OF THE STRINGS BASED ON INE SIMULATION COEFFICIENTS
strings=np.loadtxt('/home/maria/Escritorio/tfgfis/Codigo/cl_model_strings.dat',dtype=float)
lstring=strings[:,0]
clstring=strings[:,1]#*1e-12
dlstring=lstring*(lstring+1)*clstring/2/np.pi
lll=np.loadtxt('/home/maria/Escritorio/tfgfis/Codigo/lclstrings.dat',dtype=float)
lnot=lll[0:int(len(lll)/2)]
clnot=lll[int(len(lll)/2):]#*1e-12
dlnot=lnot*(lnot+1)*clnot/2/np.pi
lmax=min(len(dlnot),len(dlstring))
p1=plt.plot(lstring[6:lmax],dlstring[6:lmax],'-',label='String simulation')
p5=plt.plot(lnot[6:lmax],dlnot[6:lmax],'*',color='g',label='Model of strings')
plt.xlabel('$\ell$')
plt.ylabel('$D_ \ell$')
# plt.xscale('log')
#plt.yscale('log')
#plt.legend()
plt.show()
plt.clf()
plt.cla()
plt.close()


#Planck data for CMB
datadl=np.loadtxt('/home/maria/Escritorio/tfgfis/Canny/COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt',comments='#',skiprows=1)
lcmb=datadl[:,0]
dl=datadl[:,1]

p5=plt.plot(lcmb,dl,label='CMB')
plt.xlabel('$\ell$')
# plt.legend()
plt.ylabel('$D_\ell$ [$\mu K^2$]')
#plt.savefig('/home/maria/Escritorio/tfgfis/Codigo/simulations/dllend.jpg')
plt.show()
plt.clf()
plt.cla()
plt.close()
#Strings simulations
strings=hp.read_map('/home/maria/Documentos/simulaciones/map1n_allz_rtaapixlw_2048_1.fits')
lmax=2510
strings=strings*2.726e6

def smooth_map(mapa,fwhm,lmax,name):
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
	hp.fitsfunc.write_map(name, smoothedmap, overwrite=True)
	return smoothedmap

strings_smooth=smooth_map(strings,5,lmax,'/home/maria/Escritorio/tfgfis/Codigo/simulations/STRINGS-smooth.fits')
del(strings)
lcmb=datadl[:,0]
dl=datadl[:,1]
cmbcl=[]
for i in range(0,len(lcmb)):
	cmbcl.append(dl[i]/lcmb[i]/(lcmb[i]+1)*2*np.pi)
cmbcl=np.array(cmbcl)
nside=2048
cmbmap=hp.sphtfunc.synfast(cmbcl,nside, lmax=lmax, mmax=None, alm=False, pol=False, pixwin=False, fwhm=0.0, sigma=None, new=False, verbose=True)


gu=1e-6
strings_smooth=strings_smooth*gu
hp.visufunc.mollview(strings_smooth,fig=4)
plt.title('STRINGS-smooth')
plt.savefig('/home/maria/Escritorio/tfgfis/Codigo/simulations/STRINGS-smooth-1e-6.jpg')
plt.show()
plt.clf()
plt.cla()
plt.close()
#Loading all the needed maps

#Defining muG as a parameter of the scale of the strings

cmb=smooth_map(cmbmap,5,lmax,'/home/maria/Escritorio/tfgfis/Codigo/simulations/CMB-smooth.fits')
noise=hp.read_map('/home/maria/Escritorio/tfgfis/Codigo/simulations/NOISE.fits')
#These last two maps are the combination of strings and cmb, and noise, strings and cmb
noise=noise*0.25
# cmbnoise=cmb+noise
cmbstringsnoise=cmb+noise+strings_smooth
cmbstrings=cmb+strings_smooth

#gu=1e-7
lmax=2510
def l_cl_dl(mapa,lmax):
	"""
	map
	lmax
	This routine creates l, cl and dl for a given map when the maximum l is given
	"""
	mapcl=hp.sphtfunc.anafast(mapa, map2=None, nspec=None, lmax=lmax, mmax=lmax, iter=3, alm=False, pol=False, use_weights=True, datapath=None, gal_cut=0, use_pixel_weights=False)
	l=[]
	dl=[]
	for i in range(0,len(mapcl)):
		l.append(i)
		dl.append(mapcl[i]*l[i]*(l[i]+1)/2/np.pi)
	l=np.array(l)
	dl=np.array(dl)
	return l,mapcl,dl

strings_l,strings_cl,strings_dl=l_cl_dl(strings_smooth,lmax)
cmb_l,cmb_cl,cmb_dl=l_cl_dl(cmb,lmax)
noise_l,noise_cl,noise_dl=l_cl_dl(noise,lmax)

del(cmb)
del(cmbmap)
del(noise)
cmbstrings_l,cmbstrings_cl,cmbstrings_dl=l_cl_dl(cmbstrings,lmax)
cmbstringsnoise_l,cmbstringsnoise_cl,cmbstringsnoise_dl=l_cl_dl(cmbstringsnoise,lmax)
#Plotting data
p1=plt.plot(strings_l,strings_dl,label='Strings')
p2=plt.plot(cmb_l,cmb_dl,label='CMB')
p3=plt.plot(noise_l,noise_dl,label='Noise')
p4=plt.plot(cmbstrings_l,cmbstrings_dl,label='CMB-Strings')
p5=plt.plot(cmbstringsnoise_l,cmbstringsnoise_dl,label='CMB-Strings-Noise')
plt.title('')
plt.xscale('log')
plt.xlabel('$\ell$')
plt.legend()
plt.ylabel('$D_\ell$ [$\mu K^2$]')
plt.savefig('/home/maria/Escritorio/tfgfis/Codigo/simulations/dll.jpg')
plt.show()
plt.clf()
plt.cla()
plt.close()

p1=plt.plot(noise_l[1500:2507],noise_dl[1500:2507],label='Noise')
p5=plt.plot(cmbstrings_l[1500:2507],cmbstrings_dl[1500:2507],label='CMB-Strings')
#plt.title('dl-l')
#plt.xscale('log')
plt.xlabel('$\ell$')
plt.legend()
plt.ylabel('$D_\ell$ [$\mu K^2$]')
#plt.savefig('/home/maria/Escritorio/tfgfis/Codigo/simulations/dllend.jpg')
plt.show()
plt.clf()
plt.cla()
plt.close()



strings=strings*gu
stringview=hp.visufunc.mollview(strings,fig=3)
plt.title('STRINGS G\u03BC='+ str(gu))
plt.savefig('/home/maria/Escritorio/tfgfis/Codigo/simulations/STRINGS(ONLY).jpg')
plt.show()
plt.clf()
plt.cla()
plt.close()
#Smooth map

hp.visufunc.mollview(strings_smooth,fig=4)
plt.title('STRINGS-smooth G\u03BC='+ str(gu))
plt.savefig('/home/maria/Escritorio/tfgfis/Codigo/simulations/STRINGS-smooth.jpg')
#plt.show()
plt.clf()
plt.cla()
plt.close()

#NOT SMOOTHED MAPS
strings=hp.read_map('/home/maria/Documentos/simulaciones/map1n_allz_rtaapixlw_2048_1.fits')*2.7265
strings_l,strings_cl,strings_dl=l_cl_dl(strings,lmax)
lcmb=datadl[:,0]
dl=datadl[:,1]
p1=plt.plot(lcmb,dl)
p2=plt.plot(strings_l,strings_dl)
p3=plt.plot(noise_l,noise_dl)
plt.title('')
plt.xlabel('l')
plt.xscale('log')
plt.ylabel('Dl')
plt.show()


p1=plt.plot(lcmb,dl)
plt.title('CMB')
plt.xlabel('l')
plt.xscale('log')
plt.ylabel('Dl')
plt.show()

p2=plt.plot(strings_l,strings_dl)
plt.title('STRINGS')
plt.xlabel('l')
plt.xscale('log')
plt.ylabel('Dl')
plt.show()

p3=plt.plot(noise_l,noise_dl)
plt.title('NOISE')
plt.xlabel('l')
plt.xscale('log')
plt.ylabel('Dl')
plt.show()
plt.clf()
plt.cla()
plt.close()



# lstring,clstring=np.loadtxt('/home/maria/Escritorio/tfgfis/Codigo/cl_model_strings.dat')
# p5=plt.plot(lstring,clstring,'-')
# plt.title('Model of strings')
# plt.xlabel('$\ell$')
# plt.ylabel('$D_\ell$')
# plt.show()
# plt.clf()
# plt.cla()
# plt.close()
