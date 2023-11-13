# %% [markdown]
# ## Pulse numbers demonstration

# %%
from copy import deepcopy
import pint.models
import astropy.units as u
import pint.toa
from pint.residuals import Residuals
import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
from astropy import log
import pint.config
import pint.logging
import pint.fitter as fit

# setup logging
pint.logging.setup(level="INFO")

quantity_support()

# %%
modelin = pint.models.get_model(pint.config.examplefile("waves.par"))
model2 = deepcopy(modelin)
component, order, from_list, comp_type = model2.map_component("Wave")
from_list.remove(component)
# modelin.TRACK.value = "0"
# model2.TRACK.value = "0"

# %%
realtoas = pint.toa.get_TOAs(pint.config.examplefile("waves_withpn.tim"))
res = Residuals(realtoas, modelin)
res2 = Residuals(realtoas, model2)

# %%
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_title("Residuals using PN from .tim file")
ax.errorbar(
    res.toas.get_mjds(),
    res.time_resids,
    yerr=res.toas.get_errors(),
    fmt=".",
    label="With WAVES",
)
ax.errorbar(
    res2.toas.get_mjds(),
    res2.time_resids,
    yerr=res2.toas.get_errors(),
    fmt=".",
    label="Without WAVES",
)
ax.legend()
ax.grid(True)

# %%
realtoas.compute_pulse_numbers(modelin)
res = Residuals(realtoas, modelin)
res2 = Residuals(realtoas, model2)

# %%
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_title("Residuals using PN from compute_pulse_numbers")
ax.errorbar(
    res.toas.get_mjds(),
    res.time_resids,
    yerr=res.toas.get_errors(),
    fmt=".",
    label="With WAVES",
)
ax.errorbar(
    res2.toas.get_mjds(),
    res2.time_resids,
    yerr=res2.toas.get_errors(),
    fmt=".",
    label="Without WAVES",
)
ax.legend()
ax.grid(True)

plt.show()

# %%
modelin.WAVE1.quantity = (0.0 * u.s, 0.0 * u.s)
modelin.WAVE2.quantity = (0.0 * u.s, 0.0 * u.s)
modelin.WAVE3.quantity = (0.0 * u.s, 0.0 * u.s)
modelin.WAVE4.quantity = (0.0 * u.s, 0.0 * u.s)
modelin.WAVE5.quantity = (0.0 * u.s, 0.0 * u.s)

modelin.WAVE1.frozen = False
modelin.WAVE2.frozen = False
modelin.WAVE3.frozen = False
modelin.WAVE4.frozen = False
modelin.WAVE5.frozen = False

# %%
prefitresids = Residuals(realtoas, modelin)

# %%
try:
    f = fit.WLSFitter(realtoas, modelin)
    f.fit_toas()
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_title("Residuals using PN from compute_pulse_numbers")
    ax.errorbar(
        prefitresids.toas.get_mjds(),
        prefitresids.time_resids,
        yerr=prefitresids.toas.get_errors(),
        fmt=".",
        label="Prefit",
    )
    ax.errorbar(
        f.resids.toas.get_mjds(),
        f.resids.time_resids,
        yerr=f.resids.toas.get_errors(),
        fmt=".",
        label="Postfit",
    )
    ax.legend()
    ax.grid(True)
    plt.show()
except:
    log.error("The fit crashed!  We need to fix issue #593 to get it working.")
