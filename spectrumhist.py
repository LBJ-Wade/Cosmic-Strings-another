"""
-------------------------------------------------------------------------------
AUTHOR: Maria Martin Vega
-------------------------------------------------------------------------------
SPECTRUM HISTOGRAMS
ABSTRACT:From the simulations created in spectrum analysis we get the upper bound for Gmu 
------------------------------------------------------------------------------
"""

import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams.update({'font.size': 14})
gulist=np.loadtxt(os.getcwd()+'/statistics/gaussianity-spectrum2.dat')
print(gulist)
# gulist2=np.loadtxt(os.getcwd()+'/statistics/gaussianity-spectrum3.dat')
# gulist=np.concatenate((gulist2,gulist))
# print(gulist)
# gulist=np.array(gulist)

plt.hist(gulist,label='\u03BE',color='r',density=True,bins=10)
plt.title('Spectrum analysis')
p95=np.percentile(gulist,95)
plt.axvline(x=p95, label='95-percentile',color='olive',linestyle='--')
plt.legend()
plt.xlabel('\u03BE')
plt.show()
plt.clf()
plt.close()
print(p95)
print(np.sqrt(p95))
#Result 1.9649653054976795e-08

strings=np.loadtxt('/home/maria/Escritorio/tfgfis/Codigo/cl_model_strings.dat')
lstring=strings[:,0]
clstring=strings[:,1]*1e-12
dlstring=lstring*(lstring+1)*clstring/2/np.pi
# p5=plt.plot(lstring[4:],dlstring[4:],'-')
# plt.title('Model of strings')
# plt.xlabel('$\ell$')
# plt.ylabel('$D_ \ell$')
# plt.xscale('log')
# # plt.yscale('log')
# plt.show()
# plt.clf()
# plt.cla()
# plt.close()

lll=np.loadtxt('/home/maria/Escritorio/tfgfis/Codigo/lclstrings.dat')
lnot=lll[0:int(len(lll)/2)]
clnot=lll[int(len(lll)/2):]*1e-12
dlnot=lnot*(lnot+1)*clnot/2/np.pi

plt.plot(lstring[4:],dlstring[4:],'-',label='String simulation')
plt.plot(lnot[1:],dlnot[1:],'*',color='g',label='Model of strings')
plt.xlabel('$\ell$')
plt.ylabel('$D_ \ell$')
plt.xscale('log')
plt.yscale('log')
# plt.legend()
plt.show()
plt.clf()
plt.cla()
plt.close()
print(np.std(gulist))