---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# PINT Example Session


The PINT homepage is at:  https://github.com/nanograv/PINT.
There, you can find a Wiki with information on installing PINT
PINT can be run via a script, in an interactive session with ipython or jupyter, or using one of the command-line tools provided.


## Times of Arrival (TOAs)


The raw data for PINT are TOAs, which can be read in from files in a variety of formats, or constructed programatically. PINT currently can read TEMPO, Tempo2, and Fermi "FT1" photon files.

Note:  The first time TOAs get read in, lots of processing (can) happen, which can take some time. However, a  "pickle" file is saved, so the next time the same file is loaded (if nothing has changed), the TOAs will be loaded from the pickle file, which is much faster.

```python
from __future__ import print_function, division
import numpy as np
import astropy.units as u
```

```python
%matplotlib inline
import matplotlib.pyplot as plt
# Turn on quantity support for plotting. This is very helpful!
from astropy.visualization import quantity_support
quantity_support() 
```

```python
# Here is how to create a single TOA in Python
import pint.toa as toa

a = toa.TOA(
    (54567, 0.876876876876876),
    4.5,
    freq=1400.0,
    obs="GBT",
    backend="GUPPI",
    location=None,
)
print(a)
```

```python
# An example of reading a TOA file
import pint.toa as toa

t = toa.get_TOAs("NGC6440E.tim", usepickle=False)
```

```python
#  Here is a summary.
t.print_summary()
```

```python
# Here is the MJD of the first TOA
t.get_mjds()[0]
```

TOAs are stored in a [Astropy Table](https://astropy.readthedocs.org/latest/table/)  in an instance of the TOAs class.

```python
# List the table columns, which include pre-computed TDB times and solar system positions and velocities
t.table.colnames
```

Lots of cool things that tables can do...

```python
# This pops open a browser window showing the contents of the table
tt = t.table
# tt.show_in_browser()
```

Can do fancy sorting, selecting, re-arranging very easily.

```python
select = t.get_errors() < 20 * u.us
print(select)
```

```python
tt["tdb"][select]
```

Many PINT routines / classes / functions use [Astropy Units](https://astropy.readthedocs.org/latest/units/) internally or externally:

```python
t.get_errors()
```

The times in each row contain (or are derived from) [Astropy Time](https://astropy.readthedocs.org/latest/time/) objects:

```python
t0 = tt["mjd"][0]
```

```python
t0.tai
```

But the most useful timescale, TDB is also stored as long double numpy arrays, to maintain precision:

```python
tt["tdbld"][:3]
```

## Timing (or other) Models


Now let's define and load a timing model

```python
import pint.models as models

m = models.get_model("NGC6440E.par")
```

```python
print(m.as_parfile())
```

Timing models are basically composed of "delay" terms and "phase" terms. Currently the delay terms are organized into two 'levels'. L1 are delay terms local to the Solar System, which are needed for computing 'barycenter-corrected' TOAs. L2 are delay terms for the binary system.  (This system may change in the future to accommodate more complicated scenarios)

```python
m.delay_funcs
```

```python
m.phase_funcs
```

Can easily show/compute individual terms...

```python
ds = m.solar_system_shapiro_delay(t)
print(ds)
```

```python
plt.plot(t.get_mjds(high_precision=False), ds * 1e6, "x")
plt.xlabel("MJD")
plt.ylabel("Delay ($\mu$s)")
```

or all of the terms added together:

```python
m.delay(t)
```

```python
m.phase(t)
```

## Residuals

```python
import pint.residuals as r
```

```python
rs = r.Residuals(t, m)
```

```python
# Note that the Residuals object contains a toas member that has the TOAs used to compute
# the residuals, so you can use that to get the MJDs and uncertainties for each TOA
# Also note that plotting astropy Quantities must be enabled using 
# astropy quanity_support() first (see beginning of this notebook)
plt.errorbar(rs.toas.get_mjds(), rs.time_resids.to(u.us), yerr=rs.toas.get_errors().to(u.us), 
             fmt='.')
plt.title("%s Pre-Fit Timing Residuals" % m.PSR.value)
plt.xlabel("MJD")
plt.ylabel("Residual (phase)")
plt.grid()
```


## Fitting and Post-Fit residuals


The fitter is *completely* separate from the model and the TOA code.  So you can use any type of fitter with some easy coding.  This example uses a very simple Powell minimizer from the SciPy optimize module.

```python
import pint.fitter as fit

f = fit.WLSFitter(t, m)
f.fit_toas()
```

```python
print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
print("RMS in phase is", f.resids.phase_resids.std())
print("RMS in time is", f.resids.time_resids.std().to(u.us))
print("\n Best model is:")
print(f.model.as_parfile())
```


```python
plt.errorbar(t.get_mjds(),
             f.resids.time_resids.to(u.us),
             t.get_errors().to(u.us), fmt="x")
plt.title("%s Post-Fit Timing Residuals" % m.PSR.value)
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()
```

## Other interesting things


We can make Barycentered TOAs in a single line!

```python
m.get_barycentric_toas(t)
```

```python

```
