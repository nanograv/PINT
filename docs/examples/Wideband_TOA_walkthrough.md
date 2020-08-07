---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Wideband TOA fitting

```python
import os

from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WidebandTOAFitter
import matplotlib.pyplot as plt
import astropy.units as u
```

## Setup your inputs

```python
model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
toas = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim", ephem="de436")
```

## Setup the fitter like old time

```python
fitter = WidebandTOAFitter(toas, model)
```

## Run your fits like old time

```python
fitter.fit_toas()
```

## What are the difference?


### Concept of fitting different types of data together
#### Residuals are combined with TOA/time residuals and dm residuals

```python
type(fitter.resids)
```

#### If we look into the resids attribute, it has two independent Residual objects.

```python
fitter.resids.residual_objs
```

#### Each of them can be used independently

* Time residual

```python
time_resids = fitter.resids.residual_objs[0].time_resids
plt.errorbar(
    toas.get_mjds().value,
    time_resids.to_value(u.us),
    yerr=toas.get_errors().to_value(u.us),
    fmt="x",
)
plt.ylabel("us")
plt.xlabel("MJD")
```

```python
# Time RMS
print(fitter.resids.residual_objs[0].rms_weighted())
print(fitter.resids.residual_objs[0].chi2)
```

* DM residual

```python
dm_resids = fitter.resids.residual_objs[1].resids
dm_error = fitter.resids.residual_objs[1].data_error
plt.errorbar(toas.get_mjds().value, dm_resids.value, yerr=dm_error.value, fmt="x")
plt.ylabel("pc/cm^3")
plt.xlabel("MJD")
```

```python
# DM RMS
print(fitter.resids.residual_objs[1].rms_weighted())
print(fitter.resids.residual_objs[1].chi2)
```

#### However, in the combined residuals, one can access rms and chi2 as well

```python
print(fitter.resids.rms_weighted())
print(fitter.resids.chi2)
```

#### The initial residuals is also a combined residual object

```python
time_resids = fitter.resids_init.residual_objs[0].time_resids
plt.errorbar(
    toas.get_mjds().value,
    time_resids.to_value(u.us),
    yerr=toas.get_errors().to_value(u.us),
    fmt="x",
)
plt.ylabel("us")
plt.xlabel("MJD")
```

```python
dm_resids = fitter.resids_init.residual_objs[1].resids
dm_error = fitter.resids_init.residual_objs[1].data_error
plt.errorbar(toas.get_mjds().value, dm_resids.value, yerr=dm_error.value, fmt="x")
plt.ylabel("pc/cm^3")
plt.xlabel("MJD")
```

#### Design Matrix are combined

```python
d_matrix = fitter.get_designmatrix()
```

```python
print("Number of TOAs:", toas.ntoas)
print("Number of DM measurments:", len(fitter.resids.residual_objs[1].dm_data))
print("Number of fit params:", len(fitter.get_fitparams()))
print("Shape of design matrix:", d_matrix.shape)
```

#### Covariance Matrix are combined

```python
c_matrix = fitter.get_noise_covariancematrix()
```

```python
print("Shape of covariance matrix:", c_matrix.shape)
```

### NOTE the matrix are PINTMatrix object right now, here are the difference


If you want to access the matrix data

```python
print(d_matrix.matrix)
```

PINT matrix has labels that marks all the element in the matrix. It has the label name, index of range of the matrix, and the unit.

```python
print("labels for dimension 0:", d_matrix.labels[0])
```

```python

```
