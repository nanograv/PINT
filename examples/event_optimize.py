import numpy as np
import pint.toa as toa
import pint.models
import pint.fitter as fitter
import pint.fermi_toas as fermi
import matplotlib.pyplot as plt
import astropy.table
import astropy.units as u
import psr_utils as pu
import scipy.optimize as op
import sys, os, copy, fftfit

if len(sys.argv[1:])==4:
    eventfile, parfile, gaussianfile, weightcol = sys.argv[1:]
elif len(sys.argv[1:])==3:
    eventfile, parfile, gaussianfile = sys.argv[1:]
    weightcol=None
else:
    print "usage:  python event_optimize.py eventfile parfile gaussianfile [weightcol]"
    sys.exit()

# Params you might want to edit
nwalkers = 100
nsteps = 1500
nbins = 256
maxMJD = 57210.0 # latest MJD to use (limited by IERS file usually)
minWeight = 0.1  # if using weights, this is the minimum to include

# initialization values
maxlike = -9e99
numcalls = 0

def measure_phase(profile, template, rotate_prof=True):
    """
    measure_phase(profile, template):
        Call FFTFIT on the profile and template to determine the
            following parameters: shift,eshift,snr,esnr,b,errb,ngood
            (returned as a tuple).  These are defined as in Taylor's
            talk at the Royal Society.
    """
    c,amp,pha = fftfit.cprof(template)
    pha1 = pha[0]
    if (rotate_prof):
        pha = np.fmod(pha-np.arange(1,len(pha)+1)*pha1,2.0*np.pi)
    shift,eshift,snr,esnr,b,errb,ngood = fftfit.fftfit(profile,amp,pha)
    return shift,eshift,snr,esnr,b,errb,ngood

def profile_likelihood(phs, *otherargs):
    """
    A single likelihood calc for matching phases to a template.
    Likelihood is calculated as per eqn 2 in Pletsch & Clark 2015.
    """
    xvals, phases, template, weights = otherargs
    phss = phases.astype(np.float64) + phs
    # ensure that all the phases are within 0-1
    phss[phss > 1.0] -= 1.0
    phss[phss < 0.0] += 1.0
    probs = np.interp(phss, xvals, template, right=template[0])
    if weights is None:
        return -np.log(probs).sum()
    else:
        return -np.log(weights*probs + 1.0-weights).sum()

def marginalize_over_phase(phases, template, weights=None, resolution=1.0/1024,
    minimize=True, fftfit=False, showplot=False, lophs=0.0, hiphs=1.0):
    """
    def marginalize_over_phase(phases, template, weights=None, resolution=1.0/1024,
        minimize=True, fftfit=False, showplot=False, lophs=0.0, hiphs=1.0):
            a pulse profile comprised of combined photon phases.  A maximum
            likelood technique is used.  The shift and the max log likehood
            are returned.  You probably want to use "minimize" rathre than
            "fftfit" unless you are only sampling very close to your known min.
    """
    ltemp = len(template)
    xtemp = np.arange(ltemp) * 1.0/ltemp
    if minimize:
        phs, like = marginalize_over_phase(phases, template, weights,
            resolution=1.0/64, minimize=False, showplot=showplot)
        phs = 1.0 - phs / ltemp
        hwidth = 0.03
        lophs, hiphs = phs - hwidth, phs + hwidth
        result = op.minimize(profile_likelihood, [phs],
            args=(xtemp, phases, template, weights), bounds=[[lophs, hiphs]])
        return ltemp - result['x'] * ltemp, -result['fun']
    if fftfit:
        deltabin = 3
        h, x = np.histogram(phases.astype(np.float64), ltemp, range=[0.0, 1.0],
            weights=weights)
        s,es,snr,esnr,b,errb,ngood = measure_phase(h, template,
            rotate_prof=False)
        # s is in bins based on the template size
        lophs = (ltemp - s - deltabin) / float(ltemp)  # bins below
        if lophs < 0.0:  lophs += 1.0
        hiphs = lophs + 2.0 * deltabin / float(ltemp)  # bins above
    dphss = np.arange(lophs, hiphs, resolution)
    trials = phases.astype(np.float64) + dphss[:,np.newaxis]
    # ensure that all the phases are within 0-1
    trials[trials > 1.0] -= 1.0
    probs = np.interp(trials, xtemp, template, right=template[0])
    if weights is None:
        lnlikes = (np.log(probs)).sum(axis=1)
    else:
        lnlikes = (np.log(weights*probs + 1.0-weights)).sum(axis=1)
    if showplot:
        plt.plot(dphss, lnlikes)
        plt.xlabel("Pulse Phase")
        plt.ylabel("Log likelihood")
        plt.show()
    return ltemp - dphss[lnlikes.argmax()]*ltemp, lnlikes.max()

class emcee_fitter(fitter.fitter):

    def __init__(self, toas=None, model=None, template=None, weights=None):
        self.toas = toas
        self.model_init = model
        self.reset_model()
        self.template = template
        self.weights = weights
        self.fitkeys, self.fitvals, self.fiterrs = self.get_lnprior_vals()
        self.n_fit_params = len(self.fitvals)

    def get_event_phases(self):
        """
        Return pulse phases based on the current model
        """
        phss = self.model.phase(self.toas.table)[1]
        # ensure all postive
        return np.where(phss < 0.0, phss + 1.0, phss)

    def get_lnprior_vals(self, errfact=1.5):
        """
        By default use Gaussian priors on fit params of errfact * TEMPO errors
        """
        fitkeys = [p for p in self.model.params if not
            getattr(self.model,p).frozen]
        fitvals = []
        fiterrs = []
        for p in fitkeys:
            fitvals.append(getattr(self.model, p).value)
            fiterrs.append(getattr(self.model, p).uncertainty * errfact)
            if p in ["RAJ", "DECJ", "T0"]:
                fitvals[-1] = fitvals[-1].value
                if p != "T0":
                    fiterrs[-1] = fiterrs[-1].value
        return fitkeys, np.asarray(fitvals), np.asarray(fiterrs)

    def lnprior(self, theta):
        """
        The log prior (in this case, gaussian based on initial param errors)
        """
        lnsum = 0.0
        for val, mn, sig in zip(theta, self.fitvals, self.fiterrs):
            lnsum += (-np.log(sig * np.sqrt(2.0 * np.pi)) -
                (val-mn)**2.0/(2.0*sig**2.0))
        return lnsum

    def lnposterior(self, theta):
        """
        The log posterior (priors * likelihood)
        """
        global maxlike, numcalls
        self.set_params(dict(zip(self.fitkeys, theta)))
        # Make sure parallax is positive if we are fitting for it
        if 'PX' in self.fitkeys and self.model.PX.value < 0.0:
            return -np.inf
        phases = self.get_event_phases()
        lnlikelihood = marginalize_over_phase(phases, self.template,
            weights=self.weights)[1]
        numcalls += 1
        if lnlikelihood > maxlike:
            print "New max: ", lnlikelihood
            for name, val in zip(ftr.fitkeys, theta):
                    print "  %8s: %25.15g" % (name, val)
            maxlike = lnlikelihood
            self.maxlike_fitvals = theta
        if numcalls % (nwalkers * nsteps / 100) == 0:
            print "~%d%% complete" % (numcalls / (nwalkers * nsteps / 100))
        return self.lnprior(theta) + lnlikelihood

    def minimize_func(self, theta):
        """
        Returns -log(likelihood) so that we can use scipy.optimize.minimize
        """
        # first scale the params based on the errors
        ntheta = (theta * self.fiterrs) + self.fitvals
        self.set_params(dict(zip(self.fitkeys, ntheta)))
        if 'PX' in self.fitkeys and self.model.PX.value < 0.0:
            return np.inf
        phases = self.get_event_phases()
        lnlikelihood = marginalize_over_phase(phases, self.template,
            weights=self.weights)[1]
        print lnlikelihood, ntheta
        return -lnlikelihood

    def phaseogram(self, weights=None, bins=100, rotate=0.0, size=5,
        alpha=0.25, file=False):
        """
        Make a nice 2-panel phaseogram for the current model
        """
        mjds = self.toas.table['tdbld'].astype(np.float64)
        phss = self.get_event_phases()
        fermi.phaseogram(mjds, phss, weights=self.weights, bins=bins,
            rotate=rotate, size=size, alpha=alpha, file=file)

# TODO: make this properly handle long double
if 1 or not (os.path.isfile(eventfile+".pickle") or
    os.path.isfile(eventfile+".pickle.gz")):
    # Read event file and return list of TOA objects
    tl = fermi.load_Fermi_TOAs(eventfile, weightcolumn=weightcol)
    # Limit the TOAs to ones where we have IERS corrections for
    tl = [tl[ii] for ii in range(len(tl)) if tl[ii].mjd.value < maxMJD and
        tl[ii].flags['weight'] > minWeight]
    print "There are %d events we will use" % len(tl)
    # Now convert to TOAs object and compute TDBs and posvels
    ts = toa.TOAs(toalist=tl)
    ts.filename = eventfile
    ts.compute_TDBs()
    ts.compute_posvels(ephem="DE421", planets=False)
    if weightcol is not None:
        weights = np.asarray([x['weight'] for x in ts.table['flags']])
    else:
        weights = None

# Read in initial model
modelin = pint.models.get_model(parfile)
# Remove the dispersion delay as it is unnecessary
modelin.delay_funcs.remove(modelin.dispersion_delay)

# Now load in the gaussian template and normalize it
gtemplate = pu.read_gaussfitfile(gaussianfile, nbins)
gtemplate /= gtemplate.sum()

# Now define the requirements for emcee
ftr = emcee_fitter(ts, modelin, gtemplate, weights)

# Now compute the photon phases and see if we see a pulse
phss = ftr.get_event_phases()
maxbin, like_start = marginalize_over_phase(phss, gtemplate,
    weights=ftr.weights, minimize=True, showplot=False)
print "Starting pulse likelihood:", like_start
ftr.phaseogram(file=ftr.model.PSR.value+"_pre.png")
plt.close()
#ftr.phaseogram()

# Try normal optimization first to see how it goes
result = op.minimize(ftr.minimize_func, np.zeros_like(ftr.fitvals))
newfitvals = np.asarray(result['x']) * ftr.fiterrs + ftr.fitvals
like_optmin = -result['fun']
print "Optimization likelihood:", like_optmin
ftr.set_params(dict(zip(ftr.fitkeys, newfitvals)))
#ftr.phaseogram()

# Set up the initial conditions for the emcee walkers.  Use the
# scipy.optimize newfitvals instead if they are better
ndim = ftr.n_fit_params
if like_start > like_optmin:
    pos = [ftr.fitvals + ftr.fiterrs * np.random.randn(ndim)
        for i in range(nwalkers)]
else:
    pos = [newfitvals + ftr.fiterrs*np.random.randn(ndim)
        for i in range(nwalkers)]

# If we are fitting for PX, make sure that the initial conditions
# are positive for all walkers, since we heavily penalize PX < 0
if 'PX' in ftr.fitkeys:
    idx = ftr.fitkeys.index('PX')
    for xs in pos:
        if xs[idx] < 0.0:
            xs[idx] = np.fabs(xs[idx])

import emcee
sampler = emcee.EnsembleSampler(nwalkers, ndim, ftr.lnposterior)
# The number is the number of points in the chain
sampler.run_mcmc(pos, nsteps)

def chains_to_dict(names, sampler):
    chains = [sampler.chain[:,:,ii].T for ii in range(len(names))]
    return dict(zip(names,chains))

def plot_chains(chain_dict, file=False):
    np = len(chain_dict)
    fig, axes = plt.subplots(np, 1, sharex=True, figsize=(8, 9))
    for ii, name in enumerate(chain_dict.keys()):
        axes[ii].plot(chain_dict[name], color="k", alpha=0.3)
        axes[ii].set_ylabel(name)
    axes[np-1].set_xlabel("Step Number")
    fig.tight_layout()
    if file:
        fig.savefig(file)
        plt.close()
    else:
        plt.show()
        plt.close()

chains = chains_to_dict(ftr.fitkeys, sampler)
plot_chains(chains, file=ftr.model.PSR.value+"_chains.png")

# Make the triangle plot.
import triangle
burnin = 200
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
fig = triangle.corner(samples, labels=ftr.fitkeys)
fig.savefig(ftr.model.PSR.value+"_triangle.png")
plt.close()

# Make a phaseogram with the 50th percentile values
#ftr.set_params(dict(zip(ftr.fitkeys, np.percentile(samples, 50, axis=0))))
# Make a phaseogram with the best MCMC result
ftr.set_params(dict(zip(ftr.fitkeys, ftr.maxlike_fitvals)))
ftr.phaseogram(file=ftr.model.PSR.value+"_post.png")
plt.close()

# Print the best MCMC values and ranges
ranges = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print "Post-MCMC values (50th percentile +/- (16th/84th percentile):"
for name, vals in zip(ftr.fitkeys, ranges):
    print "%8s:"%name, "%25.15g (+ %12.5g  / - %12.5g)"%vals

import cPickle
cPickle.dump(samples, open(ftr.model.PSR.value+"_samples.pickle", "wb"))
