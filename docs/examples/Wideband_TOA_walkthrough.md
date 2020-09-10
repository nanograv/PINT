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

# Wideband TOA fitting

```python execution={"iopub.execute_input": "2020-09-10T16:29:20.198689Z", "iopub.status.busy": "2020-09-10T16:29:20.198111Z", "iopub.status.idle": "2020-09-10T16:29:22.547401Z", "shell.execute_reply": "2020-09-10T16:29:22.547856Z"}
import os

from pint.models import get_model
from pint.toa import get_TOAs
from pint.fitter import WidebandTOAFitter
import matplotlib.pyplot as plt
import astropy.units as u
```

## Setup your inputs

```python execution={"iopub.execute_input": "2020-09-10T16:29:22.551487Z", "iopub.status.busy": "2020-09-10T16:29:22.550933Z", "iopub.status.idle": "2020-09-10T16:29:24.214947Z", "shell.execute_reply": "2020-09-10T16:29:24.214428Z"}
model = get_model("J1614-2230_NANOGrav_12yv3.wb.gls.par")
toas = get_TOAs("J1614-2230_NANOGrav_12yv3.wb.tim", ephem="de436")
```

## Setup the fitter like old time

```python execution={"iopub.execute_input": "2020-09-10T16:29:24.231849Z", "iopub.status.busy": "2020-09-10T16:29:24.225575Z", "iopub.status.idle": "2020-09-10T16:29:24.723913Z", "shell.execute_reply": "2020-09-10T16:29:24.723416Z"}
fitter = WidebandTOAFitter(toas, model)
```

## Run your fits like old time

```python execution={"iopub.execute_input": "2020-09-10T16:29:24.760908Z", "iopub.status.busy": "2020-09-10T16:29:24.760345Z", "iopub.status.idle": "2020-09-10T16:29:28.292646Z", "shell.execute_reply": "2020-09-10T16:29:28.292141Z"}
fitter.fit_toas()
```

## What are the difference?


### Concept of fitting different types of data together
#### Residuals are combined with TOA/time residuals and dm residuals

```python execution={"iopub.execute_input": "2020-09-10T16:29:28.296487Z", "iopub.status.busy": "2020-09-10T16:29:28.295938Z", "iopub.status.idle": "2020-09-10T16:29:28.299335Z", "shell.execute_reply": "2020-09-10T16:29:28.298731Z"}
type(fitter.resids)
```

#### If we look into the resids attribute, it has two independent Residual objects.

```python execution={"iopub.execute_input": "2020-09-10T16:29:28.303156Z", "iopub.status.busy": "2020-09-10T16:29:28.302609Z", "iopub.status.idle": "2020-09-10T16:29:28.305446Z", "shell.execute_reply": "2020-09-10T16:29:28.305874Z"}
fitter.resids.residual_objs
```

#### Each of them can be used independently

* Time residual

```python execution={"iopub.execute_input": "2020-09-10T16:29:28.342821Z", "iopub.status.busy": "2020-09-10T16:29:28.330288Z", "iopub.status.idle": "2020-09-10T16:29:28.520180Z", "shell.execute_reply": "2020-09-10T16:29:28.519607Z"}
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

```python execution={"iopub.execute_input": "2020-09-10T16:29:28.525251Z", "iopub.status.busy": "2020-09-10T16:29:28.524698Z", "iopub.status.idle": "2020-09-10T16:29:28.527648Z", "shell.execute_reply": "2020-09-10T16:29:28.527083Z"}
# Time RMS
print(fitter.resids.residual_objs[0].rms_weighted())
print(fitter.resids.residual_objs[0].chi2)
```

* DM residual

```python execution={"iopub.execute_input": "2020-09-10T16:29:28.556722Z", "iopub.status.busy": "2020-09-10T16:29:28.535404Z", "iopub.status.idle": "2020-09-10T16:29:28.698341Z", "shell.execute_reply": "2020-09-10T16:29:28.697831Z"}
dm_resids = fitter.resids.residual_objs[1].resids
dm_error = fitter.resids.residual_objs[1].data_error
plt.errorbar(toas.get_mjds().value, dm_resids.value, yerr=dm_error.value, fmt="x")
plt.ylabel("pc/cm^3")
plt.xlabel("MJD")
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:28.703699Z", "iopub.status.busy": "2020-09-10T16:29:28.703155Z", "iopub.status.idle": "2020-09-10T16:29:28.705717Z", "shell.execute_reply": "2020-09-10T16:29:28.706203Z"}
# DM RMS
print(fitter.resids.residual_objs[1].rms_weighted())
print(fitter.resids.residual_objs[1].chi2)
```

#### However, in the combined residuals, one can access rms and chi2 as well

```python execution={"iopub.execute_input": "2020-09-10T16:29:28.710647Z", "iopub.status.busy": "2020-09-10T16:29:28.710098Z", "iopub.status.idle": "2020-09-10T16:29:28.713817Z", "shell.execute_reply": "2020-09-10T16:29:28.713259Z"}
print(fitter.resids.rms_weighted())
print(fitter.resids.chi2)
```

#### The initial residuals is also a combined residual object

```python execution={"iopub.execute_input": "2020-09-10T16:29:28.792714Z", "iopub.status.busy": "2020-09-10T16:29:28.779022Z", "iopub.status.idle": "2020-09-10T16:29:28.937303Z", "shell.execute_reply": "2020-09-10T16:29:28.936720Z"}
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

```python execution={"iopub.execute_input": "2020-09-10T16:29:28.957488Z", "iopub.status.busy": "2020-09-10T16:29:28.956731Z", "iopub.status.idle": "2020-09-10T16:29:29.107675Z", "shell.execute_reply": "2020-09-10T16:29:29.107097Z"}
dm_resids = fitter.resids_init.residual_objs[1].resids
dm_error = fitter.resids_init.residual_objs[1].data_error
plt.errorbar(toas.get_mjds().value, dm_resids.value, yerr=dm_error.value, fmt="x")
plt.ylabel("pc/cm^3")
plt.xlabel("MJD")
```

#### Design Matrix are combined

```python execution={"iopub.execute_input": "2020-09-10T16:29:29.121833Z", "iopub.status.busy": "2020-09-10T16:29:29.115596Z", "iopub.status.idle": "2020-09-10T16:29:32.307439Z", "shell.execute_reply": "2020-09-10T16:29:32.307892Z"}
d_matrix = fitter.get_designmatrix()
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:32.318273Z", "iopub.status.busy": "2020-09-10T16:29:32.311723Z", "iopub.status.idle": "2020-09-10T16:29:32.327689Z", "shell.execute_reply": "2020-09-10T16:29:32.327089Z"}
print("Number of TOAs:", toas.ntoas)
print("Number of DM measurments:", len(fitter.resids.residual_objs[1].dm_data))
print("Number of fit params:", len(fitter.get_fitparams()))
print("Shape of design matrix:", d_matrix.shape)
```

#### Covariance Matrix are combined

```python execution={"iopub.execute_input": "2020-09-10T16:29:32.339963Z", "iopub.status.busy": "2020-09-10T16:29:32.339244Z", "iopub.status.idle": "2020-09-10T16:29:37.638810Z", "shell.execute_reply": "2020-09-10T16:29:37.638235Z"}
c_matrix = fitter.get_noise_covariancematrix()
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:37.642426Z", "iopub.status.busy": "2020-09-10T16:29:37.641863Z", "iopub.status.idle": "2020-09-10T16:29:37.645045Z", "shell.execute_reply": "2020-09-10T16:29:37.644512Z"}
print("Shape of covariance matrix:", c_matrix.shape)
```

### NOTE the matrix are PINTMatrix object right now, here are the difference


If you want to access the matrix data

```python execution={"iopub.execute_input": "2020-09-10T16:29:37.648769Z", "iopub.status.busy": "2020-09-10T16:29:37.648232Z", "iopub.status.idle": "2020-09-10T16:29:37.651476Z", "shell.execute_reply": "2020-09-10T16:29:37.650922Z"}
print(d_matrix.matrix)
```

PINT matrix has labels that marks all the element in the matrix. It has the label name, index of range of the matrix, and the unit.

```python execution={"iopub.execute_input": "2020-09-10T16:29:37.654999Z", "iopub.status.busy": "2020-09-10T16:29:37.654448Z", "iopub.status.idle": "2020-09-10T16:29:37.657207Z", "shell.execute_reply": "2020-09-10T16:29:37.656755Z"}
print("labels for dimension 0:", d_matrix.labels[0])
```

```python

```
