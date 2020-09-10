---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# PINT Example Session


The PINT homepage is at:  https://github.com/nanograv/PINT.

The documentation is availble here: https://nanograv-pint.readthedocs.io/en/latest/index.html

PINT can be run via a Python script, in an interactive session with ipython or jupyter, or using one of the command-line tools provided.


## Times of Arrival (TOAs)


The raw data for PINT are TOAs, which can be read in from files in a variety of formats, or constructed programatically. PINT currently can read TEMPO, Tempo2, and ITOA text files, as well as a range of spacecraft FITS format event files (e.g. Fermi "FT1" and NICER .evt files).

Note:  The first time TOAs get read in, lots of processing (can) happen, which can take some time. However, a  "pickle" file can be saved, so the next time the same file is loaded (if nothing has changed), the TOAs will be loaded from the pickle file, which is much faster.

```python execution={"iopub.execute_input": "2020-09-10T16:29:39.132757Z", "iopub.status.busy": "2020-09-10T16:29:39.132213Z", "iopub.status.idle": "2020-09-10T16:29:39.423718Z", "shell.execute_reply": "2020-09-10T16:29:39.423106Z"}
from __future__ import print_function, division
import numpy as np
import astropy.units as u
from pprint import pprint
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:39.428349Z", "iopub.status.busy": "2020-09-10T16:29:39.427787Z", "iopub.status.idle": "2020-09-10T16:29:40.002429Z", "shell.execute_reply": "2020-09-10T16:29:40.001957Z"}
%matplotlib inline
import matplotlib.pyplot as plt

# Turn on quantity support for plotting. This is very helpful!
from astropy.visualization import quantity_support

quantity_support()
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:40.006715Z", "iopub.status.busy": "2020-09-10T16:29:40.006154Z", "iopub.status.idle": "2020-09-10T16:29:41.145039Z", "shell.execute_reply": "2020-09-10T16:29:41.145529Z"}
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
)
print(a)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.149363Z", "iopub.status.busy": "2020-09-10T16:29:41.148810Z", "iopub.status.idle": "2020-09-10T16:29:41.579135Z", "shell.execute_reply": "2020-09-10T16:29:41.579710Z"}
# An example of reading a TOA file
import pint.toa as toa

t = toa.get_TOAs("NGC6440E.tim", usepickle=False)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.596914Z", "iopub.status.busy": "2020-09-10T16:29:41.596300Z", "iopub.status.idle": "2020-09-10T16:29:41.600398Z", "shell.execute_reply": "2020-09-10T16:29:41.600851Z"}
#  You can print a summary of the loaded TOAs
t.print_summary()
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.605483Z", "iopub.status.busy": "2020-09-10T16:29:41.604935Z", "iopub.status.idle": "2020-09-10T16:29:41.607757Z", "shell.execute_reply": "2020-09-10T16:29:41.607286Z"}
# The get_mjds() method returns an array of the MJDs for the TOAs
# Here is the MJD of the first TOA. Notice that is has the units of days
pprint(t.get_mjds())
```

TOAs are stored in a [Astropy Table](https://astropy.readthedocs.org/latest/table/)  in an instance of the TOAs class.

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.611991Z", "iopub.status.busy": "2020-09-10T16:29:41.611442Z", "iopub.status.idle": "2020-09-10T16:29:41.614625Z", "shell.execute_reply": "2020-09-10T16:29:41.614163Z"}
# List the table columns, which include pre-computed TDB times and
# solar system positions and velocities
t.table.colnames
```

Lots of cool things that tables can do...

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.617601Z", "iopub.status.busy": "2020-09-10T16:29:41.617068Z", "iopub.status.idle": "2020-09-10T16:29:41.619770Z", "shell.execute_reply": "2020-09-10T16:29:41.619182Z"}
# This pops open a browser window showing the contents of the table
# t.table.show_in_browser()
```

Can do fancy sorting, selecting, re-arranging very easily.

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.623825Z", "iopub.status.busy": "2020-09-10T16:29:41.623278Z", "iopub.status.idle": "2020-09-10T16:29:41.625580Z", "shell.execute_reply": "2020-09-10T16:29:41.626100Z"}
select = t.get_errors() < 20 * u.us
print(select)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.630816Z", "iopub.status.busy": "2020-09-10T16:29:41.630273Z", "iopub.status.idle": "2020-09-10T16:29:41.633189Z", "shell.execute_reply": "2020-09-10T16:29:41.632689Z"}
pprint(t.table["tdb"][select])
```

TOAs objects have a select() method to select based on a boolean mask. This selection can be undone later with unselect.

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.664597Z", "iopub.status.busy": "2020-09-10T16:29:41.647943Z", "iopub.status.idle": "2020-09-10T16:29:41.673985Z", "shell.execute_reply": "2020-09-10T16:29:41.673406Z"}
t.print_summary()
t.select(select)
t.print_summary()
t.unselect()
t.print_summary()
```

PINT routines / classes / functions use [Astropy Units](https://astropy.readthedocs.org/latest/units/) internally and externally as much as possible:

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.678293Z", "iopub.status.busy": "2020-09-10T16:29:41.677752Z", "iopub.status.idle": "2020-09-10T16:29:41.681143Z", "shell.execute_reply": "2020-09-10T16:29:41.680693Z"}
pprint(t.get_errors())
```

The times in each row contain (or are derived from) [Astropy Time](https://astropy.readthedocs.org/latest/time/) objects:

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.684616Z", "iopub.status.busy": "2020-09-10T16:29:41.684084Z", "iopub.status.idle": "2020-09-10T16:29:41.686828Z", "shell.execute_reply": "2020-09-10T16:29:41.686285Z"}
toa0 = t.table["mjd"][0]
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.690991Z", "iopub.status.busy": "2020-09-10T16:29:41.690450Z", "iopub.status.idle": "2020-09-10T16:29:41.693862Z", "shell.execute_reply": "2020-09-10T16:29:41.693304Z"}
toa0.tai
```

But the most useful timescale, TDB is also stored in its own column as a long double numpy array, to maintain precision and keep from having to redo the conversion.
*Note that is is the TOA time converted to the TDB timescale, but the Solar System delays have not been applied, so this is NOT what people call "barycentered times"*

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.697857Z", "iopub.status.busy": "2020-09-10T16:29:41.697295Z", "iopub.status.idle": "2020-09-10T16:29:41.700215Z", "shell.execute_reply": "2020-09-10T16:29:41.699660Z"}
pprint(t.table["tdbld"][:3])
```

## Timing Models


Now let's define and load a timing model

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.703464Z", "iopub.status.busy": "2020-09-10T16:29:41.702914Z", "iopub.status.idle": "2020-09-10T16:29:41.967700Z", "shell.execute_reply": "2020-09-10T16:29:41.968159Z"}
import pint.models as models

m = models.get_model("NGC6440E.par")
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.973653Z", "iopub.status.busy": "2020-09-10T16:29:41.973019Z", "iopub.status.idle": "2020-09-10T16:29:41.975935Z", "shell.execute_reply": "2020-09-10T16:29:41.975460Z"}
# Printing a model gives the parfile representation
print(m)
```

Timing models are composed of "delay" terms and "phase" terms, which are computed by the Components of the model. The delay terms are evaluated in order, going from terms local to the Solar System, which are needed for computing 'barycenter-corrected' TOAs, through terms for the binary system.

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.980773Z", "iopub.status.busy": "2020-09-10T16:29:41.980219Z", "iopub.status.idle": "2020-09-10T16:29:41.983587Z", "shell.execute_reply": "2020-09-10T16:29:41.983140Z"}
# delay_funcs lists all the delay functions in the model, and the order is important!
m.delay_funcs
```

The phase functions include the spindown model and an absolute phase definition (if the TZR parameters are specified).

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.987415Z", "iopub.status.busy": "2020-09-10T16:29:41.986745Z", "iopub.status.idle": "2020-09-10T16:29:41.990377Z", "shell.execute_reply": "2020-09-10T16:29:41.989759Z"}
# And phase_funcs holds a list of all the phase functions
m.phase_funcs
```

You can easily show/compute individual terms...

```python execution={"iopub.execute_input": "2020-09-10T16:29:41.999568Z", "iopub.status.busy": "2020-09-10T16:29:41.999005Z", "iopub.status.idle": "2020-09-10T16:29:42.001706Z", "shell.execute_reply": "2020-09-10T16:29:42.001251Z"}
ds = m.solar_system_shapiro_delay(t)
pprint(ds)
```

The `get_mjds()` method can return the TOA times as either astropy Time objects (for high precision), or as double precisions Quantities (for easy plotting).

```python execution={"iopub.execute_input": "2020-09-10T16:29:42.027533Z", "iopub.status.busy": "2020-09-10T16:29:42.026975Z", "iopub.status.idle": "2020-09-10T16:29:42.409792Z", "shell.execute_reply": "2020-09-10T16:29:42.409192Z"}
plt.plot(t.get_mjds(high_precision=False), ds.to(u.us), "+")
plt.xlabel("MJD")
plt.ylabel("Solar System Shapiro Delay ($\mu$s)")
```

Here are all of the terms added together:

```python execution={"iopub.execute_input": "2020-09-10T16:29:42.431540Z", "iopub.status.busy": "2020-09-10T16:29:42.430971Z", "iopub.status.idle": "2020-09-10T16:29:42.433970Z", "shell.execute_reply": "2020-09-10T16:29:42.433308Z"}
pprint(m.delay(t))
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:42.458375Z", "iopub.status.busy": "2020-09-10T16:29:42.457824Z", "iopub.status.idle": "2020-09-10T16:29:42.460655Z", "shell.execute_reply": "2020-09-10T16:29:42.460149Z"}
pprint(m.phase(t))
```

## Residuals

```python execution={"iopub.execute_input": "2020-09-10T16:29:42.463981Z", "iopub.status.busy": "2020-09-10T16:29:42.463446Z", "iopub.status.idle": "2020-09-10T16:29:42.467428Z", "shell.execute_reply": "2020-09-10T16:29:42.466942Z"}
import pint.residuals
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:42.492576Z", "iopub.status.busy": "2020-09-10T16:29:42.492016Z", "iopub.status.idle": "2020-09-10T16:29:42.494589Z", "shell.execute_reply": "2020-09-10T16:29:42.493984Z"}
rs = pint.residuals.Residuals(t, m)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:42.517012Z", "iopub.status.busy": "2020-09-10T16:29:42.516449Z", "iopub.status.idle": "2020-09-10T16:29:42.689756Z", "shell.execute_reply": "2020-09-10T16:29:42.689161Z"}
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
plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()
```

## Fitting and Post-Fit residuals


The fitter is *completely* separate from the model and the TOA code.  So you can use any type of fitter with some easy coding to create a new subclass of `Fitter`.  This example uses PINT's Weighted Least Squares fitter. The return value for this fitter is the chi^2 after the fit.

```python execution={"iopub.execute_input": "2020-09-10T16:29:42.693910Z", "iopub.status.busy": "2020-09-10T16:29:42.693349Z", "iopub.status.idle": "2020-09-10T16:29:42.890759Z", "shell.execute_reply": "2020-09-10T16:29:42.891296Z"}
import pint.fitter

f = pint.fitter.WLSFitter(t, m)
f.fit_toas()  # fit_toas() returns the final reduced chi squared
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:42.894577Z", "iopub.status.busy": "2020-09-10T16:29:42.893996Z", "iopub.status.idle": "2020-09-10T16:29:42.935805Z", "shell.execute_reply": "2020-09-10T16:29:42.936250Z"}
# You can now print a nice human-readable summary of the fit
f.print_summary()
```


```python execution={"iopub.execute_input": "2020-09-10T16:29:42.956235Z", "iopub.status.busy": "2020-09-10T16:29:42.955676Z", "iopub.status.idle": "2020-09-10T16:29:43.130677Z", "shell.execute_reply": "2020-09-10T16:29:43.130112Z"}
# Lets plot the post-fit residuals
plt.errorbar(
    t.get_mjds(), f.resids.time_resids.to(u.us), t.get_errors().to(u.us), fmt="x"
)
plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()
```

## Other interesting things


We can make Barycentered TOAs in a single line, if you have a model and a TOAs object! These are TDB times with the Solar System delays applied (precisely which of the delay components are applied is changeable -- the default applies all delays before the ones associated with the binary system)

```python execution={"iopub.execute_input": "2020-09-10T16:29:43.152582Z", "iopub.status.busy": "2020-09-10T16:29:43.152023Z", "iopub.status.idle": "2020-09-10T16:29:43.155240Z", "shell.execute_reply": "2020-09-10T16:29:43.154585Z"}
pprint(m.get_barycentric_toas(t))
```

```python

```
