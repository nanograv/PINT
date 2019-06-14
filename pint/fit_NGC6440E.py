#! /usr/bin/env python
from __future__ import print_function, division
import pint.toa
import pint.models
import pint.fitter
import pint.residuals
import pint.models.model_builder as mb
from pint.phase import Phase
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from pint.utils import make_toas

#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import astropy.units as u
import os

datadir = os.path.dirname(os.path.abspath(str(__file__)))
parfile = os.path.join(datadir, 'NGC6440E.par')
timfile = os.path.join(datadir, 'NGC6440E.tim')

# Define the timing model
m = mb.get_model(parfile)

# Read in the TOAs
t = pint.toa.get_TOAs(timfile)

# Examples of how to select some subsets of TOAs
# These can be un-done using t.unselect()
#
# Use every other TOA
# t.select(np.where(np.arange(t.ntoas) % 2))

# Use only TOAs with errors < 30 us
# t.select(t.get_errors() < 30 * u.us)

# Use only TOAs from the GBT (although this is all of them for this example)
# t.select(t.get_obss() == 'gbt')
#t.select(t.get_mjds() > 53750 * u.d)
# Print a summary of the TOAs that we have
t.print_summary()

# These are pre-fit residuals
rs = pint.residuals.resids(t, m).phase_resids
xt = t.get_mjds()
#plt.plot(xt, rs, 'x', label = 'pre-fit')
#plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
#plt.xlabel('MJD')
#plt.ylabel('Residual (phase)')
#plt.grid()
#plt.show()

# Now do the fit
print("Fitting...")
f = pint.fitter.WlsFitter(t, m)
print(f.fit_toas())

#get scaling factor
M, params, units, scale_by_F0 = f.get_designmatrix()
fac = M.std(axis=0)[1:]
#fac[0] = 1.0

#get mean vector
params = f.get_fitparams_num()#OrderedDict
print('params\n',params)
mean_vector = params.values()#vector
#mean_vector.insert(0,0)

# Print some basic params --> get covariance matrix
ucov_mat = (((f.resids.unscaled_cov_matrix[1:]).T)[1:]).T
f.resids.show_matrix(ucov_mat,"Unscaled Cov Matrix",switchRD=False)
f.resids.show_matrix(f.resids.scaled_cov_matrix,"Scaled Cov Matrix",switchRD=True)
print("Mean vector is", mean_vector)
print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
print("RMS in phase is", f.resids.phase_resids.std())
print("RMS in time is", f.resids.time_resids.std().to(u.us))
print("\n Best model is:")
print(f.model.as_parfile())
print('-'*100)
'''
#fix negatives in covariance matrix
ucov_mat[2][0:2] = -1*ucov_mat[2][0:2]
ucov_mat[3][0:2] = -1*ucov_mat[3][0:2]
ucov_mat[4][0:2] = -1*ucov_mat[4][0:2]
f.resids.show_matrix(ucov_mat, "Unscaled cov with negative fix")
'''
'''
#histograms
print("mean vector", mean_vector)
print("errors", np.sqrt(np.diag(ucov_mat)))
#scale by fac for calculation
mean_vector *= fac
ucov_mat = ((ucov_mat*fac).T*fac).T
nums = [[],[],[],[],[],[]]
for i in range(20000):
    a,b,c,d,e,f = np.random.multivariate_normal(mean_vector,ucov_mat)
    nums[0].append(a)
    nums[1].append(b)
    nums[2].append(c)
    nums[3].append(d)
    nums[4].append(e)
    nums[5].append(f)
#scale back to real units
mean_vector /= fac   
for i in range(6):
    nums[i] /= fac[i]
ucov_mat = ((ucov_mat/fac).T/fac)
for i in range(6):
    data = nums[i]
    mean = np.mean(data)
    std = np.std(data)
    plt.hist(nums[i], bins=400)
    plt.title(params.keys()[i]+" mean: "+str(mean)+" std: "+str(std))
    plt.show()
'''

f_rand = deepcopy(f)#create a copy of the fitter object (have to copy the fitter (rather than the model) to use set_params)
mrand = f_rand.model

#scale by fac    
print(mean_vector, fac)
mean_vector *= fac
ucov_mat = ((ucov_mat*fac).T*fac).T

for i in range(15):
    params_rand_num = np.random.multivariate_normal(mean_vector,ucov_mat) #vector of covariant random numbers for parameters (be sure mean_vec and cov are in same parameter order
    #scale back to real units
    for j in range(len(mean_vector)):
        params_rand_num[j] /= fac[j]
    params_rand = OrderedDict(zip(params.keys(),params_rand_num))
    print("randomized parameters in Odict",params_rand)
    f_rand.set_params(params_rand)
    #rs = pint.residuals.resids(t, mrand).time_resids.to(u.us).value
    #rs = mrand.phase(t)-f.model.phase(t)
    #rs = ((rs.int+rs.frac).value/m.F0.value)*10**6
    minMJD = t.get_mjds().min()
    maxMJD = t.get_mjds().max()
    x = make_toas(minMJD-((maxMJD-minMJD)*1.7),maxMJD+((maxMJD-minMJD)*1.7),100,mrand)
    x2 = make_toas(minMJD,maxMJD,100,mrand)
    rs = f_rand.model.phase(x)-f.model.phase(x)
    rs2 = f_rand.model.phase(x2)-f.model.phase(x2)
    #from calc_phase_resids in residuals
    #rs -= Phase(rs.int[0],rs.frac[0])
    #rs2 -= Phase(rs2.int[0],rs2.frac[0])
    rs -= Phase(0.0,rs2.frac.mean())
    rs = ((rs.int+rs.frac).value/m.F0.value)*10**6
    if i < 1:
        plt.plot(x.get_mjds(), rs, 'k-', alpha=0.3, label='random' )
    else:
        plt.plot(x.get_mjds(), rs, 'k-', alpha=0.3)
    #plt.xlim(minMJD.value,maxMJD.value)
#scale back to real units
mean_vector /= fac
ucov_mat = ((ucov_mat/fac).T/fac)
        
#params_rand = OrderedDict(zip(params.keys(),params_rand_num))
#print('original model copy\n',mrand)
#print('-'*100)
#print("randomized parameters", params_rand_num)
#print("randomized parameters in Odict",params_rand)
#f_rand.set_params(params_rand)
#print('-'*100)
#print('new parameters model\n',mrand)

#rs = pint.residuals.resids(t, mrand).time_resids.to(u.us).value
##rs = mrand.phase(t)-m.phase(t)
##rs = ((rs.int+rs.frac).value/m.F0.value)*10**6
#xt = t.get_mjds()
#plt.plot(xt, rs, 'x', label='Random')
#plt.title("%s Random Variables Timing Residuals" % m.PSR.value)
#plt.xlabel('MJD')
#plt.ylabel('Residual (phase)')
#plt.grid()
#plt.show()



#create new models:
    #loop that 
    # - makes a copy of the existing model (f.model)
    # - uses the cov matrix and mean vector to produce random values for the parameters
    # - assigns those values to the parameters of the model
    # - plots the "pre-fit" residuals of the model (but not plt.show() yet)
    #for all ten (say) new models
#plot new models on same graph 
#save all models as .par files?

print(np.mean(f.resids.time_resids.to(u.us).value))
plt.errorbar(xt.value,
             f.resids.time_resids.to(u.us).value,
             t.get_errors().to(u.us).value, fmt='x', label = 'post-fit')
plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (us)')
plt.legend()
plt.grid()
plt.show()
