# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Solar Wind Models

# %% [markdown]
# The standard solar wind model in PINT is implemented as the `NE_SW` parameter ([Edwards et al. (2006)](https://ui.adsabs.harvard.edu/abs/2006MNRAS.372.1549E/abstract)), which is the solar wind electron density at 1 AU (in cm$^{-3}$).  This assumes that the electron density falls as $r^{-2}$ away from the Sun.  With `SWM=0` this is all that is allowed.
#
# However, in [You et al. (2007)](https://ui.adsabs.harvard.edu/abs/2007ApJ...671..907Y/abstract) and [You et al. (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.422.1160Y/abstract), they extend the model to other radial power-law indices $r^{-p}$ (also see [Hazboun et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...929...39H/abstract)).  This is now implemented with `SWM=1` in PINT (and the power-law index is `SWP`).
#
# Finally, it is clear that the solar wind model can vary from year to year (or even over shorter timescales).  Therefore we now have a new `SWX` model (like `DMX`) that implements a separate solar wind model over different time intervals.

# %% [markdown]
# With the new model, though, there is covariance between the power-law index `SWP` and `NE_SW`, since most of the fit is determined by the maximum excess DM in the data.  Therefore for the `SWX` model we have reparameterized it to use `SWXDM`: the max DM at conjunction.  This makes the covariance a lot less.  And to ensure continuity, this is explicitly the **excess** DM, so the DM from the `SWX` model at opposition is 0.

# %%
from io import StringIO
import numpy as np

from astropy.time import Time
import astropy.coordinates
from pint.models import get_model
from pint.fitter import Fitter
from pint.simulation import make_fake_toas_uniform
import pint.utils
import pint.gridutils
import pint.logging

import matplotlib.pyplot as plt

pint.logging.setup(level="WARNING")

# %% [markdown]
# ## Demonstrate the change in covariance going from NE_SW to DMMAX

# %%
par = """
PSR J1234+5678
F0 1
DM 10
ELAT 3
ELONG 0
PEPOCH 54000
EPHEM DE440
"""

# %%
# basic model using standard SW
model0 = get_model(StringIO("\n".join([par, "NE_SW 30\nSWM 0"])))

# %%
toas = pint.simulation.make_fake_toas_uniform(
    54000,
    54000 + 365,
    153,
    model=model0,
    obs="gbt",
    add_noise=True,
)

# %%
# standard model with variable index
model1 = get_model(StringIO("\n".join([par, "NE_SW 30\nSWM 1\nSWP 2"])))
# SWX model with 1 segment
model2 = get_model(
    StringIO(
        "\n".join(
            [par, "SWXDM_0001 1\nSWXP_0001 2\nSWXR1_0001 53999\nSWXR2_0001 55000"]
        )
    )
)
model2.SWXDM_0001.quantity = model0.get_max_dm()

# %%
# parameter grids
p = np.linspace(1.5, 2.5, 13)

ne_sw = model0.NE_SW.quantity * np.linspace(0.5, 1.5, 15)
dmmax = np.linspace(0.5, 1.5, len(ne_sw)) * model0.get_max_dm()

# %%
f1 = Fitter.auto(toas, model1)
chi2_SWM1 = pint.gridutils.grid_chisq(f1, ("NE_SW", "SWP"), (ne_sw, p))[0]

# %%
f2 = Fitter.auto(toas, model2)
chi2_SWX = pint.gridutils.grid_chisq(f2, ("SWXDM_0001", "SWXP_0001"), (dmmax, p))[0]

# %%
fig, ax = plt.subplots(figsize=(16, 9))
ax.contour(
    dmmax / model0.get_max_dm(),
    p,
    chi2_SWX - chi2_SWX.min(),
    np.linspace(2, 100, 10),
    colors="b",
)
ax.contour(
    ne_sw / model0.NE_SW.quantity,
    p,
    chi2_SWM1 - chi2_SWM1.min(),
    np.linspace(2, 100, 10),
    colors="r",
    linestyles="--",
)
ax.set_ylabel("p")
ax.set_xlabel("NE_SW or DMMAX / best-fit")

# %% [markdown]
# ## SW model limits & scalings

# %% [markdown]
# With the new `SWX` model, since it is only excess DM that starts at 0, in order to make the new model agree with the old you may need to scale some quantities

# %%
# default model
model = get_model(StringIO("\n".join([par, "NE_SW 1"])))
# SWX model with a single segment to match the default model
model2 = get_model(
    StringIO(
        "\n".join(
            [par, "SWXDM_0001 1\nSWXP_0001 2\nSWXR1_0001 53999\nSWXR2_0001 55000"]
        )
    )
)
# because of the way SWX is scaled, scale the input
scale = model2.get_swscalings()[0]
model2.SWXDM_0001.quantity = model.get_max_dm() * scale
toas = make_fake_toas_uniform(54000, 54000 + 365.25, 53, model=model, obs="gbt")

t0, elongation = pint.utils.get_conjunction(
    model.get_psr_coords(),
    Time(54000, format="mjd"),
    precision="high",
)

x = toas.get_mjds().value - t0.mjd
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(x, model.solar_wind_dm(toas), ".:", label="Old Model")
ax.plot(x, model2.swx_dm(toas), label="Scaled New Model")
ax.plot(
    x,
    model2.swx_dm(toas) + model.get_min_dm(),
    "x--",
    label="Scaled New Model + Offset",
)
model2.SWXDM_0001.quantity = model.get_max_dm()
ax.plot(
    x,
    model2.swx_dm(toas) + model.get_min_dm(),
    label="Unscaled New Model + Offset",
)
ax.plot(
    x,
    model2.swx_dm(toas),
    "+-",
    label="Unscaled New Model",
)
ax.set_xlabel("Days Since Conjunction")
ax.set_ylabel("Solar Wind DM (pc/cm**3)")
ax.legend()


# %% [markdown]
# ## Utility functions

# %% [markdown]
# A few functions to help move between models or separate model `SWX` segments

# %% [markdown]
# ### Find the next conjunction (time of SW max)
# The `low` precision version just interpolates the Sun's ecliptic longitude to match that of the pulsar.  The `high` precision version uses better coordinate conversions to do this.  It also returns the elongation at conjunction

# %%
t0, elongation = pint.utils.get_conjunction(
    model.get_psr_coords(),
    Time(54000, format="mjd"),
    precision="high",
)
print(f"Next conjunction at {t0}, with elongation {elongation}")

# %% [markdown]
# As expected, the elongation is just about 3 degrees (the ecliptic latitude of the pulsar)

# %% [markdown]
# ### Divide the input times (TOAs) into years centered on each conjunction
# This returns integer indices for each year

# %%
toas = make_fake_toas_uniform(54000, 54000 + 365.25 * 3, 153, model=model, obs="gbt")
elongation = astropy.coordinates.get_sun(
    Time(toas.get_mjds(), format="mjd")
).separation(model.get_psr_coords())
t0 = pint.utils.get_conjunction(
    model.get_psr_coords(), model.PEPOCH.quantity, precision="high"
)[0]
indices = pint.utils.divide_times(Time(toas.get_mjds(), format="mjd"), t0)
fig, ax = plt.subplots(figsize=(16, 9))
for i in np.unique(indices):
    ax.plot(toas.get_mjds()[indices == i], elongation[indices == i].value, "o")
ax.set_xlabel("MJD")
ax.set_ylabel("Elongation (deg)")

# %% [markdown]
# ### Get max/min DM from standard model, or NE_SW from SWX model

# %%
model0 = get_model(StringIO("\n".join([par, "NE_SW 30\nSWM 0"])))
# standard model with variable index
model1 = get_model(StringIO("\n".join([par, "NE_SW 30\nSWM 1\nSWP 2.5"])))
# SWX model with 1 segment
model2 = get_model(
    StringIO(
        "\n".join(
            [par, "SWXDM_0001 1\nSWXP_0001 2\nSWXR1_0001 53999\nSWXR2_0001 55000"]
        )
    )
)
# one value of the scale is returned for each SWX segment
scale = model2.get_swscalings()[0]
print(f"SW scaling: {scale}")
model2.SWXDM_0001.quantity = model0.get_max_dm() * scale

# %%
# Max is at conjunction, Min at opposition
print(
    f"SWM=0: NE_SW = {model0.NE_SW.quantity:.2f} Max DM = {model0.get_max_dm():.4f}, Min DM = {model0.get_min_dm():.4f}"
)
# Max and Min depend on NE_SW and SWP (covariance)
print(
    f"SWM=1 and SWP={model1.SWP.value}:  NE_SW = {model1.NE_SW.quantity:.2f} Max DM = {model1.get_max_dm():.4f}, Min DM = {model1.get_min_dm():.4f}"
)
# For SWX, the max/min values reported do not assume that it goes to 0 at opposition (for compatibility)
print(
    f"SWX and SWP={model2.SWXP_0001.value}: NE_SW = {model2.get_ne_sws()[0]:.2f} Max DM = {model2.get_max_dms()[0]:.4f}, Min DM = {model2.get_min_dms()[0]:.4f}"
)
print(
    f"SWX and SWP={model2.SWXP_0001.value}: Scaled NE_SW = {model2.get_ne_sws()[0]/scale:.2f} Scaled Max DM = {model2.get_max_dms()[0]/scale:.4f}, Scaled Min DM = {model2.get_min_dms()[0]/scale:.4f}"
)

# %% [markdown]
# The scaled values above agree between the `SWM=0` and `SWX` models.

# %%

# %% [markdown]
# ## Solar wind as a regression on a solar activity proxy

# %% [markdown]
# The models above let the density vary in time only by fitting it piecewise (`SWX`). The
# other option is to tie it to something measured independently. Solar wind density
# tracks the solar cycle, and so do several published activity indices, so
# `SolarWindProxyRegression` makes the density at 1 AU a linear regression on one of them:
#
# $$n_E(t) = \mathrm{SWPRBETA0} + \mathrm{SWPRBETA1} \times P(t - \mathrm{SWPRLAG})$$
#
# with the usual geometry on top, $\mathrm{DM}_\mathrm{SW}(t) = n_E(t) S(t)$. Two free
# parameters can model a density that moves with the solar cycle over a long data set, against one
# for `NE_SW` and one per segment for `SWX`.
#
# | parameter | units | meaning |
# |---|---|---|
# | `SWPRBETA0` | cm$^{-3}$ | intercept, the density at zero proxy flux |
# | `SWPRBETA1` | cm$^{-3}$ per proxy unit | slope |
# | `SWPRLAG` | day | lag applied to the proxy before interpolation |
# | `SWPRSMOOTH` | day | width of the smoothing applied to the proxy, default 81 |
# | `SWPRSMOOTHKERNEL` | - | which kernel, a key of `pint.solar_indices.SMOOTHING_KERNELS` |
# | `SWPRSOURCE` | - | which series to use, a key of `pint.solar_indices.SOURCES` |
# | `SWM`, `SWP` | - | geometry, same meaning as in the standard model |
#
# Since each proxy has its own units and the proxies are not normalized, `SWPRBETA1` is
# cm$^{-3}$ per sfu for F10.7 and cm$^{-3}$ per sunspot for the sunspot number.
# This makes the coefficients mean the same thing for every pulsar whatever its observing
# span, and keeps them from changing when the proxy file grows, but
# it means that swapping in a series in different units rescales `SWPRBETA1`.
#
# This component shares `category = "solar_wind"` with the standard model, so a model
# holds one or the other, never both: `SWPRBETA0` and `NE_SW` are exactly degenerate,
# being constants multiplying the same geometry.
#
# Deviations of the solar wind dispersive delay from the inferred proxy regression can be
# handled by a correlated noise component such as `PLSWNoise` or `TimeDomainSWNoise`.

# %%
import warnings

import astropy.units as u

from pint import solar_indices

proxy_par = "\n".join([par, "SWPRBETA0 2.0\nSWPRBETA1 0.05\nSWPRSOURCE F107"])
model_proxy = get_model(StringIO(proxy_par))
proxy_component = model_proxy.components["SolarWindProxyRegression"]
print(
    f"SWPRSOURCE = {model_proxy.SWPRSOURCE.value}, "
    f"SWPRSMOOTH = {model_proxy.SWPRSMOOTH.value} d"
)

# %% [markdown]
# ### Where the proxy series comes from
#
# `SWPRSOURCE` is resolved on first evaluation, by `pint.solar_indices`, so
# `get_model(par)` gives a model that evaluates with no further setup: the par file on its
# own still specifies the model completely. The series are downloaded and cached:
#
# | key | series | cadence |
# |---|---|---|
# | `f107` | 10.7 cm radio flux adjusted to 1 AU, [DRAO Penticton](https://www.spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-en.php) | daily |
# | `ssn` | international sunspot number, [NOAA SWPC](https://services.swpc.noaa.gov/json/solar-cycle/) | monthly |
# | `ace_density` | in-situ proton density, [ACE SWEPAM](https://izw1.caltech.edu/ACE/ASC/level2/) | daily, manual export |
#
# `get_series` returns a `SolarIndexSeries`: the epochs as an `astropy.time.Time`, the
# values as an `astropy.units.Quantity`, and the meta-data of the file it read, with
# interpolation, bounds checking and smoothing as methods.
#
# Resolution goes through two tiers, the same way clock files do: an explicit `path=` or a
# directory in `$PINT_SOLAR_INDICES`, then the cached download. `set_proxy` below takes input arrays
# directly, but cannot write a par file that no longer determines the model.
# Downloading, caching, expiration and failure handling are all done through
# `pint.observatory.global_clock_corrections.get_file`.
#
# `SWPRSMOOTH` is in days and is converted to samples using the series' median cadence, so
# 81 days means 81 samples of daily F10.7 and 3 samples of the monthly sunspot number. The
# default of 81 is the conventional F10.7A window, about 3 rotations of the sun, which
# averages over associated rotational structure while leaving the longer trends in the data.
#
# `SWPRSMOOTHKERNEL` names the kernel itself, an entry of
# `pint.solar_indices.SMOOTHING_KERNELS`, built from
# [astropy convolution kernels](https://docs.astropy.org/en/stable/convolution/):
# `BOX` is an unweighted mean, `GAUSSIAN`, `TRIANGLE` and `HANN` are tapered, and
# `TRAPEZOID` keeps a flat top over three quarters of the width. `SWPRSMOOTH` is the
# full width at half maximum for all of them, so they are comparable at one width.

# %%
try:
    f107 = solar_indices.get_series("f107")
    print(f107)
    print(f"mean {f107.values.mean():.1f}, cadence {f107.cadence_days:.0f} d")
    # what a reproducible result records about its input
    for key in ("url", "retrieved", "sha256"):
        print(f"  {key}: {f107.get_metadata()[key]}")
except OSError as e:
    print(
        f"could not reach the DRAO table ({e}); the rest of this example uses a "
        "synthetic proxy"
    )

# %% [markdown]
# ### Choosing the smoothing
#
# Smoothing is what separates the deterministic mean function from whatever stochastic
# model is used alongside it. Unsmoothed daily F10.7 carries structure on the ~27 day
# solar rotation which is better modeled as a correlated noise process than a deterministic function.
#
# The kernel is applied with `boundary="extend"`, so the series is padded with its own
# endpoint value. Within half a kernel of either end the smoothed value is partly an
# average of that padding, which matters at the start of a data set that begins before
# the proxy record does.

# %%
print(
    "kernels a par file can name:",
    sorted(k.upper() for k in solar_indices.SMOOTHING_KERNELS),
)
print("any astropy Kernel1D also works from Python, but a par file cannot name it")

if "f107" in dir():
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    # top: one kernel, several widths, over the whole record
    lo, hi = f107.mjd.min(), f107.mjd.max()
    inside = (f107.mjd >= lo) & (f107.mjd <= hi)
    axes[0].plot(f107.mjd, f107.values.value, color="0.8", lw=0.6, label="raw")
    for width in (27, 81, 365):
        box = f107.smoothed(width)
        axes[0].plot(box.mjd, box.values.value, lw=1.4, label=f"BOX, {width} d")
    axes[0].set_xlim(lo, hi)
    axes[0].set_title("one kernel, several widths")

    # bottom: every kernel at the same width, where the width is the FWHM
    lo, hi = 57500, 58230
    inside = (f107.mjd >= lo) & (f107.mjd <= hi)
    axes[1].plot(
        f107.mjd[inside], f107.values.value[inside], color="0.8", lw=0.7, label="raw"
    )
    for name in sorted(solar_indices.SMOOTHING_KERNELS):
        series = f107.smoothed(81, name)
        axes[1].plot(
            series.mjd[inside], series.values.value[inside], lw=1.4, label=name.upper()
        )
    axes[1].set_xlim(lo, hi)
    axes[1].set_title(
        "every kernel at SWPRSMOOTH = 81 d, two years on the cycle-24 decline"
    )

    for ax in axes:
        ax.set_ylabel(f"F10.7 [{f107.unit}]")
        ax.legend(ncol=6, fontsize=9)
    axes[-1].set_xlabel("MJD")
    plt.tight_layout()
else:
    print("no series fetched above, so there is nothing to smooth here")

# %% [markdown]
# **Use the defaults for F10.7**: `SWPRSMOOTHKERNEL BOX` with `SWPRSMOOTH 81`. An
# unweighted 81 day mean is the F10.7A convention, three synodic solar rotations, so the
# smoothed proxy is the same quantity the literature means by smoothed F10.7, and
# `SWPRBETA1` is comparable with a published value.
#
# The tapered kernels suppress the solar rotation more sharply at the same width,
# because a flat window has sidelobes in frequency and a tapered one does not.

# Note that the published smoothing for the monthly sunspot number is a tapered 13 month mean,
# and it is not offered here, but `BOX` on `SSN` is a reasonable choice.

# %% [markdown]
# ### Using a series you already have
#
# `set_proxy` takes the series as arrays, which is convenient for exploration and is how
# this example works. A warning is thrown since a par file cannot store the proxy data,
# so a model built this way **cannot** be written to a par file
# and read back. Writing it out emits `SWPRSOURCE`, and reading that back resolves the
# named series instead, giving a different model. For anything whose results are to be
# kept, name a series in `SWPRSOURCE` or pin a file through `$PINT_SOLAR_INDICES`.
#
# The synthetic series below is a sinusoid at the 11 year cycle period, in the right
# ballpark for F10.7 in sfu.

# %%
np.random.seed(0)  # so the fitted values below are the same on every docs build

proxy_mjd = np.arange(53500.0, 60000.0)
proxy_vals = 100.0 + 60.0 * np.sin(2 * np.pi * (proxy_mjd - 53000) / (11 * 365.25))

with warnings.catch_warnings():
    # The non-serialisable warning is the point of the paragraph above, not news here.
    warnings.simplefilter("ignore", UserWarning)
    proxy_component.set_proxy(proxy_mjd, proxy_vals)

toas = make_fake_toas_uniform(
    54000,
    54000 + 365.25 * 15,
    1500,
    model=model_proxy,
    freq=np.array([430.0, 1400.0]) * u.MHz,
    multi_freqs_in_epoch=True,
    obs="gbt",
    error=1 * u.us,
    add_noise=True,
)

mjds = toas.get_mjds().value
fig, (ax, axdm) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
ax.plot(proxy_mjd, proxy_vals, color="0.7", label="proxy P(t)")
smoothed = proxy_component.smoothed_proxy()
ax.plot(
    smoothed.mjd,
    smoothed.values,
    color="C3",
    label=f"smoothed, SWPRSMOOTH = {model_proxy.SWPRSMOOTH.value:.0f} d",
)
ax.set_xlim(mjds.min(), mjds.max())
ax.set_ylabel("proxy [sfu]")
ax.legend()
axdm.plot(mjds, proxy_component.sw_proxy_ne(toas), ".", label="$n_E(t)$ [cm$^{-3}$]")
axdm.plot(
    mjds,
    1e4 * proxy_component.solar_wind_dm(toas).value,
    ".",
    label="$10^4 \\times \\mathrm{DM}_{\\mathrm{SW}}$ [pc cm$^{-3}$]",
)
axdm.set_xlabel("MJD")
axdm.legend()

# %% [markdown]
# The density follows the cycle while the DM also carries the annual conjunction spikes,
# which is the point: the two parameters set the level and its long-term trend, and the
# geometry does the rest.

# %% [markdown]
# ### Fitting the coefficients
#
# `SWPRBETA0` and `SWPRBETA1` are strongly anticorrelated, so quite different pairs of
# values give very similar `n_E(t)` series. Rather than looking at the values of these
# parameters (which are proxy source dependent as well) it is often more useful to look at
# `get_mean_ne()` and `sw_proxy_ne()`, which have a direct physical interpretation.

# %%
model_fit = get_model(StringIO(proxy_par))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    model_fit.components["SolarWindProxyRegression"].set_proxy(proxy_mjd, proxy_vals)
# start away from the injected values so the fit has something to do
model_fit.SWPRBETA0.value, model_fit.SWPRBETA1.value = 5.0, 0.02
for parname in ("F0", "DM", "SWPRBETA0", "SWPRBETA1"):
    getattr(model_fit, parname).frozen = False

f = Fitter.auto(toas, model_fit)
f.fit_toas()
fit_component = f.model.components["SolarWindProxyRegression"]

print(
    f"SWPRBETA0 = {f.model.SWPRBETA0.quantity:.4f} "
    f"+/- {f.model.SWPRBETA0.uncertainty:.4f}   (true 2.0)"
)
print(
    f"SWPRBETA1 = {f.model.SWPRBETA1.quantity:.5f} "
    f"+/- {f.model.SWPRBETA1.uncertainty:.5f}   (true 0.05)"
)
print(f"mean n_E  = {fit_component.get_mean_ne(toas):.4f}")

cov = f.parameter_covariance_matrix
names = [label[0] for label in cov.labels[0]]
idx = [names.index("SWPRBETA0"), names.index("SWPRBETA1")]
C = cov.matrix[np.ix_(idx, idx)]
print(f"corr(SWPRBETA0, SWPRBETA1) = {C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]):+.3f}")
# the uncertainty on the mean density, propagated through that 2x2 covariance
gradient = np.array([1.0, fit_component.get_proxy_at_toas(toas).mean()])
print(
    f"sigma(mean n_E) = {np.sqrt(gradient @ C @ gradient):.4f}, against "
    f"sigma(SWPRBETA0) = {f.model.SWPRBETA0.uncertainty_value:.4f}"
)

# %% [markdown]
# ### Par files
#
# The par file plus a resolvable series determines $n_E(t)$ exactly. Rebuilding the model
# from the written par file and handing it the same series reproduces the DM exactly:

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    par_text = f.model.as_parfile()
print("\n".join(line for line in par_text.splitlines() if line.startswith("SWPR")))

rebuilt = get_model(StringIO(par_text))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
    rebuilt.components["SolarWindProxyRegression"].set_proxy(proxy_mjd, proxy_vals)
print(
    "max |difference| in solar wind DM: "
    f"{np.abs(fit_component.solar_wind_dm(toas) - rebuilt.solar_wind_dm(toas)).max():.3e}"
)

# %% [markdown]
# With `SWPRSOURCE` naming a downloadable series, the `set_proxy` calls above are not
# needed at all and `get_model(par)` on its own is enough. That is the configuration to
# use for a result anyone else has to reproduce.
