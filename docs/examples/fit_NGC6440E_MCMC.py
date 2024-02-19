# %% [markdown]
# # Demonstrate fitting from a script."""

# %%
import pint.toa
import pint.models
import pint.mcmc_fitter
import pint.sampler
import pint.residuals
import matplotlib.pyplot as plt
import astropy.units as u
import pint.logging
import contextlib
import pint.config
import numpy as np

# %%
# setup logging
pint.logging.setup(level="INFO")


# %%
def plot_chains(chain_dict, file=False):
    npts = len(chain_dict)
    fig, axes = plt.subplots(npts, 1, sharex=True, figsize=(8, 9))
    for ii, name in enumerate(chain_dict.keys()):
        axes[ii].plot(chain_dict[name], color="k", alpha=0.3)
        axes[ii].set_ylabel(name)
    axes[npts - 1].set_xlabel("Step Number")
    fig.tight_layout()
    if file:
        fig.savefig(file)
    else:
        plt.show()

    plt.close()


# %%
parfile = pint.config.examplefile("NGC6440E.par.good")
timfile = pint.config.examplefile("NGC6440E.tim")
print(parfile)
print(timfile)

# %%
# Load the timing model and the TOAs
m, t = pint.models.get_model_and_toas(parfile, timfile)

# Print a summary of the TOAs that we have
t.print_summary()

# %%
# These are pre-fit residuals
rs = pint.residuals.Residuals(t, m).phase_resids
xt = t.get_mjds()
plt.plot(xt, rs, "x")
plt.title(f"{m.PSR.value} Pre-Fit Timing Residuals")
plt.xlabel("MJD")
plt.ylabel("Residual (phase)")
plt.grid()
plt.show()

# %%
nwalkers = 50
nsteps = 2000

# %%
# Now create the fitter.
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

# %%
# Now deal with priors
pint.mcmc_fitter.set_priors_basic(f)
print("Gaussian priors set")

# Examples of custom position priors:
# f.model.RAJ.prior = Prior(normal(0.001,1e5))

# %%
# Do the fit
print("Fitting...")
print(f.fit_toas(nsteps))

# %%
# plotting the chains
chains = sampler.chains_to_dict(f.fitkeys)
plot_chains(chains, file=f"{f.model.PSR.value}_chains.png")

# %%
# triangle plot
# this doesn't include burn-in because we're not using it here, otherwise set get_chain(discard=burnin)
ndim = len(m.free_params)
samples = np.transpose(sampler.get_chain(), (1, 0, 2)).reshape((-1, ndim))
with contextlib.suppress(ImportError):
    import corner

    fig = corner.corner(
        samples, labels=f.fitkeys, bins=50, truths=f.maxpost_fitvals, plot_contours=True
    )
    fig.savefig(f"{f.model.PSR.value}_triangle.png")
    plt.show()

# %%
# Print some basic params
print("Best fit has reduced chi^2 of", f.resids.reduced_chi2)
print("RMS in phase is", f.resids.phase_resids.std())
print("RMS in time is", f.resids.time_resids.std().to(u.us))
print("\n Best model is:")
print(f.model.as_parfile())

plt.errorbar(
    xt.value,
    f.resids.time_resids.to(u.us).value,
    t.get_errors().to(u.us).value,
    fmt="x",
)
plt.title(f"{m.PSR.value} Post-Fit Timing Residuals")
plt.xlabel("MJD")
plt.ylabel("Residual (us)")
plt.grid()
plt.show()
