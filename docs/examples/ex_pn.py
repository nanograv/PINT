#!/usr/bin/env python
import numpy as np
from copy import deepcopy
import pint.models
from pint.models.parameter import strParameter
import astropy.units as u
import pint.toa
from pint.residuals import Residuals
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
quantity_support()

modelin = pint.models.get_model('waves.par')
model2 = deepcopy(modelin)
component, order, from_list, comp_type = model2.map_component("Wave")
from_list.remove(component)

realtoas = pint.toa.get_TOAs('waves_withpn.tim')
res = Residuals(realtoas, modelin)
res2 = Residuals(realtoas, model2)

fig,ax=plt.subplots(figsize=(16,9))
ax.errorbar(res.toas.get_mjds(),res.time_resids,yerr=res.toas.get_errors(),fmt=".")
ax.errorbar(res2.toas.get_mjds(),res2.time_resids,yerr=res2.toas.get_errors(),fmt=".")
ax.grid(True)


realtoas.compute_pulse_numbers(modelin)
fig,ax=plt.subplots(figsize=(16,9))
ax.errorbar(res.toas.get_mjds(),res.time_resids,yerr=res.toas.get_errors(),fmt=".")
ax.errorbar(res2.toas.get_mjds(),res2.time_resids,yerr=res2.toas.get_errors(),fmt=".")
ax.grid(True)


import pint.fitter as fit

modelin.WAVE1.quantity = (0.0*u.s,0.0*u.s)
modelin.WAVE2.quantity = (0.0*u.s,0.0*u.s)
modelin.WAVE3.quantity = (0.0*u.s,0.0*u.s)
modelin.WAVE4.quantity = (0.0*u.s,0.0*u.s)
modelin.WAVE5.quantity = (0.0*u.s,0.0*u.s)

modelin.WAVE1.frozen=False
modelin.WAVE2.frozen=False
modelin.WAVE3.frozen=False
modelin.WAVE4.frozen=False
modelin.WAVE5.frozen=False

prefitresids = Residuals(realtoas,modelin)

f = fit.WLSFitter(realtoas,modelin)
f.fit_toas()

fig,ax=plt.subplots(figsize=(16,9))
ax.errorbar(prefitresids.toas.get_mjds(),prefitresids.time_resids,yerr=prefitresids.toas.get_errors(),fmt=".")
ax.errorbar(f.resids.toas.get_mjds(),f.resids.time_resids,yerr=f.resids.toas.get_errors(),fmt=".")
ax.grid(True)

plt.show()
