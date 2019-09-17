"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
MODELING Gmu FOR CMB+STRINGS+NOISE AND CHI2 TEST
ABSTRACT: We do a CHI 2 test for detection. In a first step we calculate models (averages) for the statistcs of some values of Gmu. Later we export to ocvate this models to calculate the coefficients of a polynomial of degree 8 using least square method. We import this coeffcients and define the CHI2. Finally we obtain a distribution of Gmu, so we can study detectability with it. We plot detectability, detection plots and example of two histograms.
------------------------------------------------------------------------------
"""
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
from time import time
from gaussian_filters import grad_filters
from metodocanny1 import gradient
import scipy.stats as sps
from histograms import fisher_histogram_wavelet
import matplotlib 
from scipy.optimize import curve_fit
import scipy
matplotlib.rcParams.update({'font.size': 14})
def mean_vector(stats_cmbstringsnoise):
	vectorh1=stats_cmbstringsnoise[:,1:]
	avh1=np.mean(vectorh1,0)
	return avh1
gu_list=[1.25e-6,1.125e-6,1.03125e-6,9.375e-7,9.0625e-7,1.0625e-6,1e-6,1.09375e-6,9.68755e-7,8.75e-7,5e-7,5e-6,7.5e-7,4e-6,1.3e-6,6e-7,7e-7,1e-5,4.5e-6,1.35e-6,2e-7,6.5e-7,6e-6,3e-6,2e-06]
gu_list.sort()
print(gu_list)
cn=0.25
change_noise=cn
sim_ini=0
sim_end=3000
nmean=500
av=np.zeros([len(gu_list),18])
for i in gu_list:
	stats_cmbstringsnoise=np.loadtxt(os.getcwd()+'/statistics/filt/STATS-cmb'+str(change_noise)+'n'+str(i)+'strings'+str(sim_ini)+'-'+str(sim_end)+'FILT.dat')
	posi=gu_list.index(i)
	av[posi,:]=mean_vector(stats_cmbstringsnoise[0:nmean,:])
np.savetxt(os.getcwd()+'/statisticsmodel.dat',av)
print(av)



#COEFFICIENTS WERE CALCULATED IN OCTAVE, BECAUSE THE RESULTS FITTED BETTER

coefficients=np.loadtxt(os.getcwd()+'/coefficients.dat')
def poly8(x,coefficients):
	suma=np.zeros(len(coefficients[:,0]))
	for i in range(0,9):
		suma=suma+coefficients[i,:]*x**i
	return suma


def covmatrix(stats_cmbstringsnoise,avh1,gu,cn):
	s=stats_cmbstringsnoise.shape[0]
	#Now we need the covariance matrix
	#step1
	vectorh1=stats_cmbstringsnoise[:,1:]
	differencesh1=vectorh1-avh1
	#step2: Definition of covariance matrices
	covh1=np.zeros([int(len(avh1)),int(len(avh1))])
	for i in range(0,int(len(avh1))):
		for j in range(0,i+1):
			sumah0=0
			sumah1=0
			for k in range(0,s):
				# sumah0=sumah0+differencesh0[k,i]*differencesh0[k,j]
				sumah1=sumah1+differencesh1[k,i]*differencesh1[k,j]
			# covh0[i,j]=1/s*sumah0
			covh1[i,j]=1/s*sumah1
			# covh0[j,i]=covh0[i,j]
			covh1[j,i]=covh1[i,j]
	return covh1


nmean=500
ncov=2500
nsim=3000


covav=np.zeros([18,18])
for i in gu_list:
	stats_cmbstringsnoise=np.loadtxt(os.getcwd()+'/statistics/filt/STATS-cmb'+str(change_noise)+'n'+str(i)+'strings'+str(sim_ini)+'-'+str(sim_end)+'FILT.dat')
	posi=gu_list.index(i)
	avh1=mean_vector(stats_cmbstringsnoise[0:nmean,:])
	covh1=covmatrix(stats_cmbstringsnoise[nmean:ncov,:],avh1,i,cn)
	covav=covh1+covav
covav=covav/len(gu_list)
icov=np.linalg.inv(covav)
def dot3(x,y,z):
	# y: square matrix-like
	# all of them having compatible dimensions
	prod=np.dot(np.dot(x,y),z)
	return prod
def xi_sq(x,data,coefficients,icov):
	#data is a vector of the same size as coefficients
	xisq=dot3(data-coefficients[8,:]-coefficients[7,:]*x-coefficients[6,:]*x**2-coefficients[5,:]*x**3-coefficients[4,:]*x**4-coefficients[3,:]*x**5-coefficients[2,:]*x**6-coefficients[1,:]*x**7-coefficients[0,:]*x**8,icov,data-coefficients[8,:]-coefficients[7,:]*x-coefficients[6,:]*x**2-coefficients[5,:]*x**3-coefficients[4,:]*x**4-coefficients[3,:]*x**5-coefficients[2,:]*x**6-coefficients[1,:]*x**7-coefficients[0,:]*x**8)
	return xisq


def chitest_model(gu,cn,nmean,ncov,nsim,sim_ini,sim_end,covav,coefficients):
	change_noise=cn
	stats_cmbstringsnoisegu=np.loadtxt(os.getcwd()+'/statistics/filt/STATS-cmb'+str(change_noise)+'n'+str(gu)+'strings'+str(sim_ini)+'-'+str(sim_end)+'FILT.dat')
	vectorgu=stats_cmbstringsnoisegu[:,1:]
	chitable=np.zeros([nsim-ncov,2])
	chimodel=np.zeros([nsim-ncov,len(gu_list)])

	#print('coefficients '+str(c))
	minimizinggu=[]
	for i in range(ncov,nsim):
		x0=gu
		res=scipy.optimize.minimize(xi_sq, x0, args=(vectorgu[i,:],coefficients,icov), method='Nelder-Mead', jac=None, hess=None, hessp=None, tol=None, callback=None, options={})
		#print(res.x)
		#res=scipy.optimize.minimize_scalar(xi_sq,args=(vectorgu[i,:],coefficients,icov)) #Bad results...
		minimizinggu.append(float(res.x))
	return minimizinggu


datos=[]
# gu_list.append(1.75e-6))
gu_list.append(1.8e-6)
gu_list.append(1.6e-6)
# gu_list.append(8e-6)
gu_list.append(1.5e-6)
gu_list.remove(6.5e-7)
gu_list.remove(7e-7)
gu_list.remove(9.0625e-7)
gu_list.remove(9.375e-7)
gu_list.remove(1.0625e-6)
gu_list.remove(9.68755e-7)
gu_list.remove(8.75e-7)
gu_list.remove(1.3e-6)
gu_list.remove(1.03125e-6)
gu_list.remove(1.09375e-6)
gu_list.remove(1.35e-6)
gu_list.sort()
#gu_list=[1e-6,1e-5,5e-6]
for gu in gu_list:
	minimizinggu=chitest_model(gu,cn,nmean,ncov,nsim,sim_ini,sim_end,covav,coefficients)
	#print(minimizinggu)
	std=np.std(minimizinggu)
	plt.hist(minimizinggu,label='$G\mu$',color='r',density=True,bins=10)
	plt.title('G$\mu$='+str(gu)+' - $\chi ^2$ Analysis')
	p95=np.percentile(minimizinggu,95)
	mean=np.mean(minimizinggu)
	plt.axvline(x=p95, label='95-percentile',color='olive',linestyle='--')
	plt.axvline(x=np.mean(minimizinggu), label='Mean',color='c',linestyle='--')
	plt.legend()
	plt.xlabel('$G\mu$')
	#plt.show()
	plt.clf()
	plt.close()
	if mean-2*std<0:
		s='NOT'
		bol=0
	else:
		s='DETECTION'
		bol=1
	ratio=std/mean
	datos.append([gu,mean,p95,std,bol,ratio])
	print('Percentile 95: '+str(p95))
	print('Mean: '+str(np.mean(minimizinggu)))
datos=np.array(datos)
np.savetxt(os.getcwd()+'/DETECTION-CHIsq.dat',datos)


detected=[]
notdetected=[]
detectedx=[]
notdetectedx=[]
errordet=[]
for i in range(0,len(datos[:,0])):
	if datos[i,4]==1:
		detected.append(datos[i,1])
		detectedx.append(datos[i,0])
		errordet.append(datos[i,3])
	else:
		notdetected.append(datos[i,2])
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
plt.ylabel('G$\mu$ (detected)')
# plt.xscale('log')
# plt.yscale('log')
plt.errorbar(detectedx,detected,yerr=errordet,xerr=None,fmt='', ecolor=None,elinewidth=None, capsize=None,barsabove=True, lolims=False, uplims=False,xlolims=False, xuplims=False, errorevery=1,capthick=None)
plt.title('Detection- $\chi^2$ test')
#plt.errorbar(detectedx, detected, yerr=errordet,fmt='', ecolor='g', capthick=2)

plt.show()


#Detectability
plt.plot(datos[:,0], datos[:,3]/datos[:,1],'-',color='r')
plt.axhline(y=0.5,color='g',linestyle='--')
plt.ylabel('$\overline{\sigma/G\mu}$')
plt.xlabel('$G\mu$')
plt.title('')
plt.show()
plt.clf()
plt.close()


gu=5e-7
minimizinggu=chitest_model(gu,cn,nmean,ncov,nsim,sim_ini,sim_end,covav,coefficients)
#print(minimizinggu)
plt.hist(minimizinggu,label='$G\mu$',color='r',density=True,bins=9)
plt.title('G$\mu$='+str(gu)+' - $\chi ^2$ Analysis')
p95=np.percentile(minimizinggu,95)
plt.axvline(x=p95, label='95-percentile',color='olive',linestyle='--')
plt.axvline(x=np.mean(minimizinggu), label='Mean',color='c',linestyle='--')
plt.legend()
plt.xlabel('$G\mu$')
plt.show()
plt.clf()
plt.close()
std=np.std(minimizinggu)
p95=np.percentile(minimizinggu,95)
mean=np.mean(minimizinggu)
if mean-2*std<0:
	s='NOT DETECTION'
	bol=0
else:
	s='DETECTION'
	bol=1
print('----------------------------------------------------\n Gmu  '+str(gu)+'\n  Mean= '+str(mean)+'\n  STD= '+str(std)+'\n'+s+'\n Percentile 95= '+str(p95))

gu=1e-5
minimizinggu=chitest_model(gu,cn,nmean,ncov,nsim,sim_ini,sim_end,covav,coefficients)
#print(minimizinggu)
plt.hist(minimizinggu,label='$G\mu$',color='r',density=True,bins=10)
plt.title('G$\mu$='+str(gu)+' - $\chi ^2$ Analysis')
p95=np.percentile(minimizinggu,95)
plt.axvline(x=p95, label='95-percentile',color='olive',linestyle='--')
plt.axvline(x=np.mean(minimizinggu), label='Mean',color='c',linestyle='--')
plt.legend()
plt.xlabel('$G\mu$')
plt.show()
plt.clf()
plt.close()
std=np.std(minimizinggu)
p95=np.percentile(minimizinggu,95)
mean=np.mean(minimizinggu)
if mean-2*std<0:
	s='NOT DETECTION'
	bol=0
else:
	s='DETECTION'
	bol=1
print('----------------------------------------------------\n G\mu  '+str(gu)+'\n  Mean= '+str(mean)+'\n  STD= '+str(std)+'\n'+s+'\n Percentile 95= '+str(p95))