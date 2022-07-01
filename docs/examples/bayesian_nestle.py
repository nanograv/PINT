#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pint.models
import pint.toa
import pint.bayesian


# In[35]:


import numpy as np
import nestle
import corner
import sys


# In[36]:


model, toas = pint.models.get_model_and_toas('J1028-5819_example.par', 'J1028-5819_example.tim')


# In[37]:


# This object provides lnlikelihood, lnprior and prior_transform functions
spnta = pint.bayesian.SPNTA(model, toas, prior_dist='uniform', prior_width=3)


# In[38]:


# This order is the same as model.free_params.
print("Free parameters : ", spnta.free_params)


# In[39]:


# The default print_progress function in nestle causes too much slowdown.
def print_progress(info):
    if info['it'] % 20 == 0:
        print("\r\033[Kit={:6d} logz={:8f}".format(info['it'], info['logz']),
              end='')
        sys.stdout.flush()


# In[ ]:


# Now run the sampler.
res = nestle.sample(spnta.lnlikelihood, spnta.prior_transform, spnta.ndim, 
                    method='multi', npoints=150,
                    callback=print_progress)


# In[ ]:


fig = corner.corner(res.samples, weights=res.weights, 
                    labels=spnta.free_params, 
                    range=[0.9999]*spnta.ndim)


# In[ ]:


param_means, param_cov = nestle.mean_and_cov(res.samples, weights=res.weights)
param_stds = np.diag(param_cov)**0.5


# In[ ]:


print("Parameter means and standard deviations")
print("===============================================")
print("Param\t\tMean\t\t\tStd")
print("===============================================")
for par, mean, std in zip(spnta.free_params, param_means, param_stds):
    print(f"{par}\t\t{mean:0.15e}\t{std}")
print("===============================================")
print()

print("===============================================")
np.set_printoptions(precision=2)
print("Parameter Covariance Matrix")
print("===============================================")
print(param_cov)


# In[ ]:




