# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PINT Example Session

# %% [markdown]
# The PINT homepage is at:  https://github.com/nanograv/PINT.
#
# The documentation is availble here: https://nanograv-pint.readthedocs.io/en/latest/index.html
#
# PINT can be run via a Python script, in an interactive session with ipython or jupyter, or using one of the command-line tools provided.

# %% [markdown]
# ## Times of Arrival (TOAs)

# %% [markdown]
# The raw data for PINT are TOAs, which can be read in from files in a variety of formats, or constructed programatically. PINT currently can read TEMPO, Tempo2, and ITOA text files, as well as a range of spacecraft FITS format event files (e.g. Fermi "FT1" and NICER .evt files).
#
# Note:  The first time TOAs get read in, lots of processing (can) happen, which can take some time. However, a  "pickle" file can be saved, so the next time the same file is loaded (if nothing has changed), the TOAs will be loaded from the pickle file, which is much faster.

# %%
import tempfile
import astropy.units as u
from pprint import pprint
from glob import glob
import pint.logging

# setup the logging
# let's have it give less detail
pint.logging.setup(level="WARNING")

# %%
# %matplotlib inline
import matplotlib.pyplot as plt

# Turn on quantity support for plotting. This is very helpful!
from astropy.visualization import quantity_support

quantity_support()

# %%
# Here is how to create a single TOA in Python
# The first argument is an MJD(UTC) as a 2-double tuple to allow extended precision
# and the second argument is the TOA uncertainty
# Wherever possible, it is good to use astropy units on the values,
# but there are sensible defaults if you leave them out (us for uncertainty, MHz for freq)
import pint.toa as toa

a = toa.TOA(
    (54567, 0.876876876876876),
    4.5 * u.us,
    freq=1400.0 * u.MHz,
    obs="GBT",
    backend="GUPPI",
    name="guppi_56789.fits",
)
print(a)

# %%
# An example of reading a TOA file
import pint.toa as toa
import pint.config

# maybe we want extra logging info here to see what happens when we load TOAs
pint.logging.setup(level="DEBUG")
t = toa.get_TOAs(pint.config.examplefile("NGC6440E.tim"), ephem="DE440")
# but then turn back to "WARNING" later
pint.logging.setup(level="WARNING")

# %%
#  You can print a summary of the loaded TOAs
t.print_summary()

# %%
# The get_mjds() method returns an array of the MJDs for the TOAs
# Here is the MJD of the first TOA. Notice that is has the units of days
pprint(t.get_mjds())

# %% [markdown]
# TOAs are stored in a [Astropy Table](https://astropy.readthedocs.org/latest/table/)  in an instance of the TOAs class.

# %%
# List the table columns, which include pre-computed TDB times and
# solar system positions and velocities
t.table.colnames

# %% [markdown]
# Lots of cool things that tables can do...

# %%
# This pops open a browser window showing the contents of the table
# t.table.show_in_browser()

# %% [markdown]
# Can do fancy sorting, selecting, re-arranging very easily.

# %%
select = t.get_errors() < 20 * u.us
print(select)

# %%
pprint(t.table["tdb"][select])

# %% [markdown]
# TOAs objects have a select() method to select based on a boolean mask. This selection can be undone later with unselect.

# %%
t.print_summary()
t.select(select)
t.print_summary()
t.unselect()
t.print_summary()

# %% [markdown]
# PINT routines / classes / functions use [Astropy Units](https://astropy.readthedocs.org/latest/units/) internally and externally as much as possible:

# %%
pprint(t.get_errors())

# %% [markdown]
# The times in each row contain (or are derived from) [Astropy Time](https://astropy.readthedocs.org/latest/time/) objects:

# %%
toa0 = t.table["mjd"][0]

# %%
toa0.tai

# %% [markdown]
# But the most useful timescale, TDB is also stored in its own column as a long double numpy array, to maintain precision and keep from having to redo the conversion.
# *Note that is is the TOA time converted to the TDB timescale, but the Solar System delays have not been applied, so this is NOT what people call "barycentered times"*

# %%
pprint(t.table["tdbld"][:3])

# %% [markdown]
# ## Timing Models

# %% [markdown]
# Now let's define and load a timing model

# %%
import pint.models as models

m = models.get_model(pint.config.examplefile("NGC6440E.par"))

# %%
# Printing a model gives the parfile representation
print(m)

# %% [markdown]
# Timing models are composed of "delay" terms and "phase" terms, which are computed by the Components of the model. The delay terms are evaluated in order, going from terms local to the Solar System, which are needed for computing 'barycenter-corrected' TOAs, through terms for the binary system.

# %%
# delay_funcs lists all the delay functions in the model, and the order is important!
m.delay_funcs

# %% [markdown]
# The phase functions include the spindown model and an absolute phase definition (if the TZR parameters are specified).

# %%
# And phase_funcs holds a list of all the phase functions
m.phase_funcs

# %% [markdown]
# You can easily show/compute individual terms...

# %%
ds = m.solar_system_shapiro_delay(t)
pprint(ds)

# %% [markdown]
# The `get_mjds()` method can return the TOA times as either astropy Time objects (for high precision), or as double precisions Quantities (for easy plotting).

# %%
plt.plot(t.get_mjds(high_precision=False), ds.to(u.us), "+")
plt.xlabel("MJD")
plt.ylabel("Solar System Shapiro Delay ($\mu$s)")

# %% [markdown]
# Here are all of the terms added together:

# %%
pprint(m.delay(t))

# %%
pprint(m.phase(t))

# %% [markdown]
# ## Residuals

# %%
import pint.residuals

# %%
rs = pint.residuals.Residuals(t, m)

# %%
# Note that the Residuals object contains a toas member that has the TOAs used to compute
# the residuals, so you can use that to get the MJDs and uncertainties for each TOA
# Also note that plotting astropy Quantities must be enabled using
# astropy quanity_support() first (see beginning of this notebook)
plt.errorbar(
    rs.toas.get_mjds(),
    rs.time_resids.to(u.us),
    yerr=rs.toas.get_errors().to(u.us),
    fmt=".",
)
plt.title(f"{m.PSR.value} Pre-Fit Timing Residuals")
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()

# %% [markdown]
# ## Fitting and Post-Fit residuals

# %% [markdown]
# The fitter is *completely* separate from the model and the TOA code.  So you can use any type of fitter with some easy coding to create a new subclass of `Fitter`.  This example uses PINT's Weighted Least Squares fitter. The return value for this fitter is the chi^2 after the fit.

# %%
import pint.fitter

f = pint.fitter.WLSFitter(t, m)
f.fit_toas()  # fit_toas() returns the final reduced chi squared

# %%
# You can now print a nice human-readable summary of the fit
f.print_summary()


# %%
# Lets plot the post-fit residuals
plt.errorbar(
    t.get_mjds(), f.resids.time_resids.to(u.us), t.get_errors().to(u.us), fmt="x"
)
plt.title(f"{m.PSR.value} Post-Fit Timing Residuals")
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()

# %% [markdown]
# Now let's save (and print) the post-fit par file. We'll request a more TEMPO2-compatible file, though we could have requested a more TEMPO-style file or a more native PINT format. These differ only slightly, just as much as needed to be read by the three pieces of software. PINT can read all three variants.

# %%
f.model.write_parfile("/tmp/output.par", format="tempo2")
print(f.model.as_parfile(format="tempo2"))

# %% [markdown]
# ## Other interesting things

# %% [markdown]
# We can make Barycentered TOAs in a single line, if you have a model and a TOAs object! These are TDB times with the Solar System delays applied (precisely which of the delay components are applied is changeable -- the default applies all delays before the ones associated with the binary system)

# %%
pprint(m.get_barycentric_toas(t))

# %% [markdown]
# Let's export the clock corrections as they currently stand so we can save
# these exact versions for reproducibility purposes.
#
# %%
import pint.observatory.topo_obs

d = tempfile.mkdtemp()
pint.observatory.topo_obs.export_all_clock_files(d)
for f in sorted(glob(f"{d}/*")):
    print(f)
