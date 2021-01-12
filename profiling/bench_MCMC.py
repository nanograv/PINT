#! /usr/bin/env python
"""Demonstrate fitting from a script."""
import os
import pint.toa
import pint.models
import pint.mcmc_fitter
import pint.sampler
import pint.residuals
import pint.models.model_builder as mb
import astropy.units as u
from scipy.stats import norm, uniform

datadir = "."
parfile = os.path.join(datadir, "NGC6440E.par")
timfile = os.path.join(datadir, "NGC6440E.tim")
print(parfile)
print(timfile)
nwalkers = 25
nsteps = 20

# Define the timing model
m = mb.get_model(parfile)

# Read in the TOAs
t = pint.toa.get_TOAs(timfile)

# Print a summary of the TOAs that we have
t.print_summary()

# These are pre-fit residuals
rs = pint.residuals.Residuals(t, m).phase_resids
xt = t.get_mjds()

# Now do the fit
print("Fitting...")
sampler = pint.sampler.EmceeSampler(nwalkers)
f = pint.mcmc_fitter.MCMCFitter(
    t,
    m,
    sampler,
    resids=True,
    phs=0.50,
    phserr=0.01,
    lnlike=pint.mcmc_fitter.lnlikelihood_chi2,
)

# Now deal with priors
pint.mcmc_fitter.set_priors_basic(f)
print("Gaussian priors set")

# Examples of custom position priors:

# f.model.RAJ.prior = Prior(normal(0.001,1e5))

# Do the fit
print(f.fit_toas(nsteps))

# plotting the chains
chains = sampler.chains_to_dict(f.fitkeys)

# Print some basic params
print("Best fit has reduced chi^2 of", f.resids.chi2_reduced)
print("RMS in phase is", f.resids.phase_resids.std())
print("RMS in time is", f.resids.time_resids.std().to(u.us))
print("\n Best model is:")
print(f.model.as_parfile())
