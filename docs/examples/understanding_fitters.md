---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Understanding Fitters



```python
from __future__ import print_function, division
import numpy as np
import astropy.units as u
```

```python
import pint.toa
import pint.models
import pint.fitter
```

```python
%matplotlib inline
import matplotlib.pyplot as plt

# Turn on quantity support for plotting. This is very helpful!
from astropy.visualization import quantity_support

quantity_support()
```

```python
# Load some TOAs and a model to fit
t = pint.toa.get_TOAs("NGC6440E.tim", usepickle=False)
m = pint.models.get_model("NGC6440E.par")
```

```python
# You can check if a model includes a noise model with correlated errors (e.g. ECORR or TNRED) by checking the has_correlated_errors property
m.has_correlated_errors
```

There are several fitters in PINT, each of which is a subclass of `Fitter`

* `WLSFitter` - PINT's workhorse fitter, which does a basic weighted least-squares minimization of the residuals.
* `GLSFitter` - A generalized least squares fitter, like "tempo -G", that can handle noise processes like ECORR and red noise that are specified by their correlation function properties.
* `PowellFitter` - A very simple example fitter that uses the Powell method implemented in scipy. One notable feature is that it does not require evaluating derivatives w.r.t the model parameters.
* `MCMCFitter` - A fitter that does an MCMC fit using the [emcee](https://emcee.readthedocs.io/en/stable/) package. This can be very slow, but accomodates Priors on the parameter values and can produce corner plots and other analyses of the posterior distributions of the parameters.




## Weighted Least Squares Fitter

```python
# Instantiate a fitter
wlsfit = pint.fitter.WLSFitter(toas=t, model=m)
```

A fit is performed by calling `fit_toas()`

For most fitters, multiple iterations can be done by setting the `maxiter` keyword argument

The return value of most fitters is the final chi^2 value

```python
wlsfit.fit_toas(maxiter=1)
```

```python
# A summary of the fit and resulting model parameters can easily be printed
# Only free parameters will have values and uncertainties in the Postfit column
wlsfit.print_summary()
```

```python
# The WLS fitter doesn't handle correlated errors
wlsfit.resids.model.has_correlated_errors
```

```python
# You can request a pretty-printed covariance matrix
cov = wlsfit.get_covariance_matrix(pretty_print=True)
```

```python
# plot() will make a plot of the post-fit residuals
wlsfit.plot()
```

## Powell fitter

The Powell fitter takes much longer to run! It also doesn't find quite as good of a minimum as the WLS fitter.

This uses scipy's modification of Powell’s method, which is a conjugate direction method. It performs sequential one-dimensional minimizations along each vector of the directions, which is updated at each iteration of the main minimization loop. The function need not be differentiable, and no derivatives are taken.

The default number of iterations is 20, but this can be changed with the `maxiter` parameter

```python
powfit = pint.fitter.PowellFitter(toas=t, model=m)
```

```python
powfit.fit_toas()
```

```python
powfit.print_summary()
```

***!!! Note that the Powell fitter does not produce a covariance matrix or estimates of the uncertainties. !!!***

## Comparing models

There also a convenience function for pretty printing a comparison of two models with the differences measured in sigma.

```python
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

```python
m1855 = pint.models.get_model("B1855+09_NANOGrav_9yv1.gls.par")
```

```python
# You can check if a model includes a noise model with correlated errors (e.g. ECORR or TNRED) by checking the has_correlated_errors property
m1855.has_correlated_errors
```

```python
print(m1855)
```

```python
ts1855 = pint.toa.get_TOAs("B1855+09_NANOGrav_9yv1.tim")
ts1855.print_summary()
```

```python
glsfit = pint.fitter.GLSFitter(toas=ts1855, model=m1855)
```

```python
glsfit.fit_toas(maxiter=1)
```

```python
# Not sure how to do this properly yet.
# glsfit2 = pint.fitter.GLSFitter(toas=t, model=glsfit.model, residuals=glsfit.resids)
# glsfit2.fit_toas(maxiter=0)
```

```python
glsfit.print_summary()
```

The GLS fitter produces two types of residuals, the normal residuals to the deterministic model and those from the noise model.

```python
glsfit.resids.time_resids
```

```python
glsfit.resids.noise_resids
```

```python
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
