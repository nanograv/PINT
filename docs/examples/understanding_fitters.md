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

# Understanding Fitters



```python execution={"iopub.execute_input": "2020-09-10T16:29:46.396063Z", "iopub.status.busy": "2020-09-10T16:29:46.395515Z", "iopub.status.idle": "2020-09-10T16:29:46.680836Z", "shell.execute_reply": "2020-09-10T16:29:46.680224Z"}
from __future__ import print_function, division
import numpy as np
import astropy.units as u
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:46.684438Z", "iopub.status.busy": "2020-09-10T16:29:46.683898Z", "iopub.status.idle": "2020-09-10T16:29:48.340734Z", "shell.execute_reply": "2020-09-10T16:29:48.341188Z"}
import pint.toa
import pint.models
import pint.fitter
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:48.345797Z", "iopub.status.busy": "2020-09-10T16:29:48.345215Z", "iopub.status.idle": "2020-09-10T16:29:48.636498Z", "shell.execute_reply": "2020-09-10T16:29:48.635991Z"}
%matplotlib inline
import matplotlib.pyplot as plt

# Turn on quantity support for plotting. This is very helpful!
from astropy.visualization import quantity_support

quantity_support()
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:48.640228Z", "iopub.status.busy": "2020-09-10T16:29:48.639664Z", "iopub.status.idle": "2020-09-10T16:29:49.131499Z", "shell.execute_reply": "2020-09-10T16:29:49.131925Z"}
# Load some TOAs and a model to fit
t = pint.toa.get_TOAs("NGC6440E.tim", usepickle=False)
m = pint.models.get_model("NGC6440E.par")
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:49.135573Z", "iopub.status.busy": "2020-09-10T16:29:49.134964Z", "iopub.status.idle": "2020-09-10T16:29:49.138294Z", "shell.execute_reply": "2020-09-10T16:29:49.137746Z"}
# You can check if a model includes a noise model with correlated errors (e.g. ECORR or TNRED) by checking the has_correlated_errors property
m.has_correlated_errors
```

There are several fitters in PINT, each of which is a subclass of `Fitter`

* `WLSFitter` - PINT's workhorse fitter, which does a basic weighted least-squares minimization of the residuals.
* `GLSFitter` - A generalized least squares fitter, like "tempo -G", that can handle noise processes like ECORR and red noise that are specified by their correlation function properties.
* `PowellFitter` - A very simple example fitter that uses the Powell method implemented in scipy. One notable feature is that it does not require evaluating derivatives w.r.t the model parameters.
* `MCMCFitter` - A fitter that does an MCMC fit using the [emcee](https://emcee.readthedocs.io/en/stable/) package. This can be very slow, but accomodates Priors on the parameter values and can produce corner plots and other analyses of the posterior distributions of the parameters.




## Weighted Least Squares Fitter

```python execution={"iopub.execute_input": "2020-09-10T16:29:49.179940Z", "iopub.status.busy": "2020-09-10T16:29:49.169313Z", "iopub.status.idle": "2020-09-10T16:29:49.211172Z", "shell.execute_reply": "2020-09-10T16:29:49.210676Z"}
# Instantiate a fitter
wlsfit = pint.fitter.WLSFitter(toas=t, model=m)
```

A fit is performed by calling `fit_toas()`

For most fitters, multiple iterations can be done by setting the `maxiter` keyword argument

The return value of most fitters is the final chi^2 value

```python execution={"iopub.execute_input": "2020-09-10T16:29:49.333439Z", "iopub.status.busy": "2020-09-10T16:29:49.235196Z", "iopub.status.idle": "2020-09-10T16:29:49.337152Z", "shell.execute_reply": "2020-09-10T16:29:49.336573Z"}
wlsfit.fit_toas(maxiter=1)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:49.340545Z", "iopub.status.busy": "2020-09-10T16:29:49.339987Z", "iopub.status.idle": "2020-09-10T16:29:49.423914Z", "shell.execute_reply": "2020-09-10T16:29:49.423449Z"}
# A summary of the fit and resulting model parameters can easily be printed
# Only free parameters will have values and uncertainties in the Postfit column
wlsfit.print_summary()
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:49.427735Z", "iopub.status.busy": "2020-09-10T16:29:49.427187Z", "iopub.status.idle": "2020-09-10T16:29:49.430728Z", "shell.execute_reply": "2020-09-10T16:29:49.430162Z"}
# The WLS fitter doesn't handle correlated errors
wlsfit.resids.model.has_correlated_errors
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:49.434815Z", "iopub.status.busy": "2020-09-10T16:29:49.434228Z", "iopub.status.idle": "2020-09-10T16:29:49.437688Z", "shell.execute_reply": "2020-09-10T16:29:49.437217Z"}
# You can request a pretty-printed covariance matrix
cov = wlsfit.get_covariance_matrix(pretty_print=True)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:49.455410Z", "iopub.status.busy": "2020-09-10T16:29:49.454785Z", "iopub.status.idle": "2020-09-10T16:29:49.700825Z", "shell.execute_reply": "2020-09-10T16:29:49.700298Z"}
# plot() will make a plot of the post-fit residuals
wlsfit.plot()
```

## Powell fitter

The Powell fitter takes much longer to run! It also doesn't find quite as good of a minimum as the WLS fitter.

This uses scipy's modification of Powell’s method, which is a conjugate direction method. It performs sequential one-dimensional minimizations along each vector of the directions, which is updated at each iteration of the main minimization loop. The function need not be differentiable, and no derivatives are taken.

The default number of iterations is 20, but this can be changed with the `maxiter` parameter

```python execution={"iopub.execute_input": "2020-09-10T16:29:49.770034Z", "iopub.status.busy": "2020-09-10T16:29:49.732333Z", "iopub.status.idle": "2020-09-10T16:29:49.772781Z", "shell.execute_reply": "2020-09-10T16:29:49.772199Z"}
powfit = pint.fitter.PowellFitter(toas=t, model=m)
```

```python execution={"iopub.execute_input": "2020-09-10T16:29:50.271975Z", "iopub.status.busy": "2020-09-10T16:29:49.922152Z", "iopub.status.idle": "2020-09-10T16:30:04.467925Z", "shell.execute_reply": "2020-09-10T16:30:04.468442Z"}
powfit.fit_toas()
```

```python execution={"iopub.execute_input": "2020-09-10T16:30:04.472612Z", "iopub.status.busy": "2020-09-10T16:30:04.471972Z", "iopub.status.idle": "2020-09-10T16:30:04.484548Z", "shell.execute_reply": "2020-09-10T16:30:04.483972Z"}
powfit.print_summary()
```

***!!! Note that the Powell fitter does not produce a covariance matrix or estimates of the uncertainties. !!!***

## Comparing models

There also a convenience function for pretty printing a comparison of two models with the differences measured in sigma.

```python execution={"iopub.execute_input": "2020-09-10T16:30:04.490927Z", "iopub.status.busy": "2020-09-10T16:30:04.490380Z", "iopub.status.idle": "2020-09-10T16:30:04.492804Z", "shell.execute_reply": "2020-09-10T16:30:04.493406Z"}
print(wlsfit.model.compare(powfit.model))
```

## Generalized Least Squares fitter

The GLS fitter is capable of handling correlated noise models.

It has some more complex options using the `maxiter`, `threshold`, and `full_cov` keyword arguments to `fit_toas()`.

If `maxiter` is less than one, **no fitting is done**, just the
chi-squared computation. In this case, you must provide the `residuals`
argument.

If `maxiter` is one or more, so fitting is actually done, the
chi-squared value returned is only approximately the chi-squared
of the improved(?) model. In fact it is the chi-squared of the
solution to the linear fitting problem, and the full non-linear
model should be evaluated and new residuals produced if an accurate
chi-squared is desired.

A first attempt is made to solve the fitting problem by Cholesky
decomposition, but if this fails singular value decomposition is
used instead. In this case singular values below threshold are removed.

`full_cov` determines which calculation is used. If True, the full
covariance matrix is constructed and the calculation is relatively
straightforward but the full covariance matrix may be enormous.
If False, an algorithm is used that takes advantage of the structure
of the covariance matrix, based on information provided by the noise
model. The two algorithms should give the same result to numerical
accuracy where they both can be applied.


To test this fitter properly, we need a model that includes correlated noise components, so we will load one from NANOGrav 9yr data release.

```python execution={"iopub.execute_input": "2020-09-10T16:30:04.497034Z", "iopub.status.busy": "2020-09-10T16:30:04.496484Z", "iopub.status.idle": "2020-09-10T16:30:04.870981Z", "shell.execute_reply": "2020-09-10T16:30:04.871526Z"}
m1855 = pint.models.get_model("B1855+09_NANOGrav_9yv1.gls.par")
```

```python execution={"iopub.execute_input": "2020-09-10T16:30:04.875258Z", "iopub.status.busy": "2020-09-10T16:30:04.874699Z", "iopub.status.idle": "2020-09-10T16:30:04.878088Z", "shell.execute_reply": "2020-09-10T16:30:04.877510Z"}
# You can check if a model includes a noise model with correlated errors (e.g. ECORR or TNRED) by checking the has_correlated_errors property
m1855.has_correlated_errors
```

```python execution={"iopub.execute_input": "2020-09-10T16:30:04.914831Z", "iopub.status.busy": "2020-09-10T16:30:04.914208Z", "iopub.status.idle": "2020-09-10T16:30:04.916868Z", "shell.execute_reply": "2020-09-10T16:30:04.917319Z"}
print(m1855)
```

```python execution={"iopub.execute_input": "2020-09-10T16:30:04.920970Z", "iopub.status.busy": "2020-09-10T16:30:04.920362Z", "iopub.status.idle": "2020-09-10T16:30:13.987324Z", "shell.execute_reply": "2020-09-10T16:30:13.986743Z"}
ts1855 = pint.toa.get_TOAs("B1855+09_NANOGrav_9yv1.tim")
ts1855.print_summary()
```

```python execution={"iopub.execute_input": "2020-09-10T16:30:13.991886Z", "iopub.status.busy": "2020-09-10T16:30:13.991313Z", "iopub.status.idle": "2020-09-10T16:30:16.407409Z", "shell.execute_reply": "2020-09-10T16:30:16.407866Z"}
glsfit = pint.fitter.GLSFitter(toas=ts1855, model=m1855)
```

```python execution={"iopub.execute_input": "2020-09-10T16:30:16.427618Z", "iopub.status.busy": "2020-09-10T16:30:16.421670Z", "iopub.status.idle": "2020-09-10T16:30:26.038804Z", "shell.execute_reply": "2020-09-10T16:30:26.038179Z"}
glsfit.fit_toas(maxiter=1)
```

```python execution={"iopub.execute_input": "2020-09-10T16:30:26.042140Z", "iopub.status.busy": "2020-09-10T16:30:26.041512Z", "iopub.status.idle": "2020-09-10T16:30:26.043906Z", "shell.execute_reply": "2020-09-10T16:30:26.043397Z"}
# Not sure how to do this properly yet.
# glsfit2 = pint.fitter.GLSFitter(toas=t, model=glsfit.model, residuals=glsfit.resids)
# glsfit2.fit_toas(maxiter=0)
```

```python execution={"iopub.execute_input": "2020-09-10T16:30:26.055501Z", "iopub.status.busy": "2020-09-10T16:30:26.054912Z", "iopub.status.idle": "2020-09-10T16:30:26.156579Z", "shell.execute_reply": "2020-09-10T16:30:26.156000Z"}
glsfit.print_summary()
```

The GLS fitter produces two types of residuals, the normal residuals to the deterministic model and those from the noise model.

```python execution={"iopub.execute_input": "2020-09-10T16:30:26.161363Z", "iopub.status.busy": "2020-09-10T16:30:26.160747Z", "iopub.status.idle": "2020-09-10T16:30:26.163858Z", "shell.execute_reply": "2020-09-10T16:30:26.164305Z"}
glsfit.resids.time_resids
```

```python execution={"iopub.execute_input": "2020-09-10T16:30:26.168927Z", "iopub.status.busy": "2020-09-10T16:30:26.168368Z", "iopub.status.idle": "2020-09-10T16:30:26.171958Z", "shell.execute_reply": "2020-09-10T16:30:26.171319Z"}
glsfit.resids.noise_resids
```

```python execution={"iopub.execute_input": "2020-09-10T16:30:26.196241Z", "iopub.status.busy": "2020-09-10T16:30:26.195623Z", "iopub.status.idle": "2020-09-10T16:30:26.658631Z", "shell.execute_reply": "2020-09-10T16:30:26.658052Z"}
# Here we can plot both the residuals to the deterministic model as well as the realization of the noise model residuals
# The difference will be the "whitened" residuals
fig, ax = plt.subplots(figsize=(16, 9))
mjds = glsfit.toas.get_mjds()
ax.plot(mjds, glsfit.resids.time_resids, ".")
ax.plot(mjds, glsfit.resids.noise_resids["pl_red_noise"], ".")
```

The MCMC fitter is considerably more complicated, so it has its own dedicated walkthroughs in `MCMC_walkthrough.ipynb` (for photon data) and `examples/fit_NGC6440E_MCMC.py` (for fitting TOAs).

```python

```
