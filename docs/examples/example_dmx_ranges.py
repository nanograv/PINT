# %% [markdown]
# ## DMX ranges demonstration

# %%
import astropy.units as u

import pint.toa
import pint.models
import pint.fitter
import pint.utils
import pint.config

import matplotlib.pyplot as plt
from astropy.visualization import quantity_support

quantity_support()

# %% [markdown]
# This example shows how to construct and add a set of DMX components to a model.
# In this case the model we load already has a DMX component, so we remove it first
# and replace it with one we construct.

# %%
m1855 = pint.models.get_model(pint.config.examplefile("B1855+09_NANOGrav_9yv1.gls.par"))
ts1855 = pint.toa.get_TOAs(
    pint.config.examplefile("B1855+09_NANOGrav_9yv1.tim"), usepickle=True
)
ts1855.print_summary()

# %%
# Remove and build a new DMX component
m1855.remove_component("DispersionDMX")

# %%
# Build the new component with maximum DMX bin of 15 days ensuring that there are TOAs above and below 1000 MHz in each gin
mask, dmx_comp = pint.utils.dmx_ranges(
    ts1855, binwidth=15.0 * u.d, divide_freq=1000.0 * u.MHz
)

# %%
# Mask out any TOAs that couldn't be included in a DMX range
ts1855.select(mask)

# %%
# Add it to the model
m1855.add_component(dmx_comp, validate=True)

# %%
# Now run the fit with the new DMX component
glsfit = pint.fitter.GLSFitter(toas=ts1855, model=m1855)
glsfit.fit_toas(maxiter=1)
glsfit.print_summary()

# %%
# You can print some statistics on the DMXs like this:
pint.utils.dmxstats(glsfit.model, glsfit.toas)

# %%
# dmxparse will pull out the DMX values and compute correct errors from the covariance matrix, which we can easily plot
dmx = pint.utils.dmxparse(glsfit)

# %%
fig, ax = plt.subplots(figsize=(16, 9))
ax.errorbar(dmx["dmxeps"], dmx["dmxs"], yerr=dmx["dmx_verrs"], fmt="s", label="NRT DMX")
ax.grid(True)
ax.set_ylabel("DMX")
plt.show()
