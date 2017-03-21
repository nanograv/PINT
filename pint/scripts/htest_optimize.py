#!/usr/bin/env python

# At the moment, this code is kind of an unsupported hack.
# It will not be installed by setup.py and not tested by nosetests.

from __future__ import print_function
import numpy as np
import pint.toa as toa
import pint.models
from pint.fitter import Fitter
import pint.fermi_toas as fermi
from pint.eventstats import hmw, hm, sf_hm
import matplotlib.pyplot as plt
import astropy.table
import pint.plot_utils
import astropy.units as u
import psr_utils as pu
import scipy.optimize as op
import sys, os, copy, fftfit
from astropy.coordinates import SkyCoord


# Params you might want to edit
nwalkers = 200
burnin = 100
nsteps = 1000
nbins = 256 # For likelihood calculation based on gaussians file
outprof_nbins = 256 # in the text file, for pygaussfit.py, for instance
minMJD = 54680.0 # Earliest MJD (limited by LAT data)
maxMJD = 57250.0 # latest MJD to use (limited by IERS file usually)
# Set minWeight to 0.0 to get plots about how significant the
# pulsations are before doing an expensive MCMC.  This allows
# you to set minWeight intelligently.
minWeight = 0.01 # if using weights, this is the minimum to include
errfact = 10.0 # multiplier for gaussian priors based TEMPO errors
do_opt_first = True
# Raise the calculated weights to this power
wgtexp = 0.5

# initialization values
maxpost = -9e99
numcalls = 0


class emcee_fitter(Fitter):

    def __init__(self, toas=None, model=None, weights=None):
        self.toas = toas
        self.model_init = model
        self.reset_model()
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

    def get_lnprior_vals(self, errfact=errfact):
        """
        By default use Gaussian priors on fit params of errfact * TEMPO errors
        """
        fitkeys = [p for p in self.model.params if not
            getattr(self.model,p).frozen]
        fitvals = []
        fiterrs = []
        for p in fitkeys:
            fitvals.append(getattr(self.model, p).num_value)
            fiterrs.append(getattr(self.model, p).num_uncertainty * errfact)
            # I think the following is just to strip off units from those params
            if p in ["RAJ", "DECJ", "T0", "GLEP_1"]:
                fitvals[-1] = fitvals[-1].value
                if p not in ["T0", "GLEP_1"]:
                    fiterrs[-1] = fiterrs[-1].value
        return fitkeys, np.asarray(fitvals), np.asarray(fiterrs)

    def lnprior(self, theta):
        """
        The log prior
        """
        lnsum = 0.0
        for val, mn, sig, key in \
            zip(theta, self.fitvals, self.fiterrs, self.fitkeys):
            # Do uniform priors first
            if key=="GLEP_1":
                # Don't allow a glitch within the first/last 100 days
                lnsum += 0.0 if 54680.0+100 < val < maxMJD-100 else -np.inf
            elif key=="GLPH_1":
                lnsum += 0.0 if -0.5 < val < 0.5 else -np.inf
            elif key=="SINI":
                lnsum += 0.0 if 0.0 < val < 1.0 else -np.inf
            elif key=="M2":
                lnsum += 0.0 if 0.1 < val < 0.6 else -np.inf
            else:  # gaussian prior based on initial param errors
                lnsum += (-np.log(sig * np.sqrt(2.0 * np.pi)) -
                          (val-mn)**2.0/(2.0*sig**2.0))
        return lnsum

    def lnposterior(self, theta):
        """
        The log posterior (priors * likelihood)
        """
        global maxpost, numcalls
        self.set_params(dict(zip(self.fitkeys, theta)))
        # Make sure parallax is positive if we are fitting for it
        if 'PX' in self.fitkeys and self.model.PX.value < 0.0:
            return -np.inf
        if 'SINI' in self.fitkeys and (self.model.SINI.value > 1.0 or self.model.SINI.value < 0.0):
            return -np.inf
        # Do we really need to check both E and ECC or can the model param alias handle that?
        if 'E' in self.fitkeys and (self.model.E.value < 0.0 or self.model.E.value>=1.0):
            return -np.inf
        if 'ECC' in self.fitkeys and (self.model.ECC.value < 0.0 or self.model.ECC.value>=1.0):
            return -np.inf
        phases = self.get_event_phases()
        # Here, I need to negate the survival function of H, so I am looking
        # for the maximum
        lnlikelihood = -1.0*sf_hm(hmw(phases,weights=self.weights),logprob=True)
        numcalls += 1
        if numcalls % (nwalkers * nsteps / 100) == 0:
            print("~%d%% complete" % (numcalls / (nwalkers * nsteps / 100)))
        lnpost = self.lnprior(theta) + lnlikelihood
        if lnpost > maxpost:
            print("New max: ", lnpost)
            for name, val in zip(ftr.fitkeys, theta):
                print("  %8s: %25.15g" % (name, val))
            maxpost = lnpost
            self.maxpost_fitvals = theta
        return lnpost

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
        # Here I'm using H-test and computing the log of the probability
        # of getting that value or higher. So this is already a negative
        # log likelihood, and should be minimized.
        lnlikelihood = sf_hm(hmw(phases, self.weights),logprob=True)
        print(lnlikelihood, ntheta)
        return lnlikelihood

    def phaseogram(self, bins=100, rotate=0.0, size=5,
        alpha=0.25, file=False):
        """
        Make a nice 2-panel phaseogram for the current model
        """
        mjds = self.toas.table['tdbld'].astype(np.float64)
        phss = self.get_event_phases()
        plot_utils.phaseogram(mjds, phss, weights=self.weights, bins=bins,
            rotate=rotate, size=size, alpha=alpha, file=file)

    def prof_vs_weights(self, nbins=50, use_weights=False):
        """
        Show binned profiles (and H-test values) as a function
        of the minimum weight used. nbins is only for the plots.
        """
        f, ax = plt.subplots(3, 3, sharex=True)
        phss = ftr.get_event_phases()
        htests = []
        weights = np.linspace(0.0, 0.95, 20)
        for ii, minwgt in enumerate(weights):
            good = ftr.weights > minwgt
            nphotons = np.sum(good)
            wgts = ftr.weights[good] if use_weights else None
            if nphotons <= 0:
                hval = 0
            else:
                if use_weights:
                    hval = hmw(phss[good], weights=wgts)
                else:
                    hval = hm(phss[good])
            htests.append(hval)
            if ii > 0 and ii%2==0 and ii<20:
                r, c = ((ii-2)/2)/3, ((ii-2)/2)%3
                ax[r][c].hist(phss[good], nbins, range=[0,1],
                              weights=wgts, color='k',
                              histtype='step')
                ax[r][c].set_title("%.1f / %.1f / %.0f" %
                                   (minwgt, hval, nphotons),
                                   fontsize=11)
                if c==0: ax[r][c].set_ylabel("Htest")
                if r==2: ax[r][c].set_xlabel("Phase")
                f.suptitle("%s:  Minwgt / H-test / Approx # events" %
                           self.model.PSR.value, fontweight='bold')
        if use_weights:
            plt.savefig(ftr.model.PSR.value+"_profs_v_wgtcut.png")
        else:
            plt.savefig(ftr.model.PSR.value+"_profs_v_wgtcut_unweighted.png")
        plt.close()
        plt.plot(weights, htests, 'k')
        plt.xlabel("Min Weight")
        plt.ylabel("H-test")
        plt.title(self.model.PSR.value)
        if use_weights:
            plt.savefig(ftr.model.PSR.value+"_htest_v_wgtcut.png")
        else:
            plt.savefig(ftr.model.PSR.value+"_htest_v_wgtcut_unweighted.png")
        plt.close()

def main(argv=None):

    if len(argv)==3:
        eventfile, parfile, weightcol = sys.argv[1:]
    elif len(argv)==2:
        eventfile, parfile = sys.argv[1:]
        weightcol=None
    else:
        print("usage: htest_optimize eventfile parfile [weightcol]")
        sys.exit()

    # Read in initial model
    modelin = pint.models.get_model(parfile)
    # Remove the dispersion delay as it is unnecessary
    modelin.delay_funcs['L1'].remove(modelin.dispersion_delay)
    # Set the target coords for automatic weighting if necessary
    if 'ELONG' in modelin.params:
        tc = SkyCoord(modelin.ELONG.quantity,modelin.ELAT.quantity,
            frame='barycentrictrueecliptic')
    else:
        tc = SkyCoord(modelin.RAJ.quantity,modelin.DECJ.quantity,frame='icrs')

    target = tc if weightcol=='CALC' else None

    # TODO: make this properly handle long double
    if not (os.path.isfile(eventfile+".pickle") or
        os.path.isfile(eventfile+".pickle.gz")):
        # Read event file and return list of TOA objects
        tl = fermi.load_Fermi_TOAs(eventfile, weightcolumn=weightcol,
                                   targetcoord=target, minweight=minWeight)
        # Limit the TOAs to ones where we have IERS corrections for
        tl = [tl[ii] for ii in range(len(tl)) if (tl[ii].mjd.value < maxMJD
            and (weightcol is None or tl[ii].flags['weight'] > minWeight))]
        print("There are %d events we will use" % len(tl))
        # Now convert to TOAs object and compute TDBs and posvels
        ts = toa.TOAs(toalist=tl)
        ts.filename = eventfile
        ts.compute_TDBs()
        ts.compute_posvels(ephem="DE421", planets=False)
        ts.pickle()
    else:  # read the events in as a pickle file
        picklefile = toa._check_pickle(eventfile)
        if not picklefile:
            picklefile = eventfile
        ts = toa.TOAs(picklefile)

    if weightcol is not None:
        if weightcol=='CALC':
            weights = np.asarray([x['weight'] for x in ts.table['flags']])
            print("Original weights have min / max weights %.3f / %.3f" % \
                (weights.min(), weights.max()))
            weights **= wgtexp
            wmx, wmn = weights.max(), weights.min()
                # make the highest weight = 1, but keep min weight the same
            weights = wmn + ((weights - wmn) * (1.0 - wmn) / (wmx - wmn))
            for ii, x in enumerate(ts.table['flags']):
                x['weight'] = weights[ii]
        weights = np.asarray([x['weight'] for x in ts.table['flags']])
        print("There are %d events, with min / max weights %.3f / %.3f" % \
            (len(weights), weights.min(), weights.max()))
    else:
        weights = None
        print("There are %d events, no weights are being used." % (len(weights)))

    # Now define the requirements for emcee
    ftr = emcee_fitter(ts, modelin, weights)

    # Use this if you want to see the effect of setting minWeight
    if minWeight == 0.0:
        print("Checking h-test vs weights")
        ftr.prof_vs_weights(use_weights=True)
        ftr.prof_vs_weights(use_weights=False)
        sys.exit()

    # Now compute the photon phases and see if we see a pulse
    phss = ftr.get_event_phases()
    like_start = -1.0*sf_hm(hmw(phss,weights=ftr.weights),logprob=True)
    print("Starting pulse likelihood:", like_start)
    ftr.phaseogram(file=ftr.model.PSR.value+"_pre.png")
    plt.close()
    ftr.phaseogram()

    # Write out the starting pulse profile
    vs, xs = np.histogram(ftr.get_event_phases(), outprof_nbins, \
        range=[0,1], weights=ftr.weights)
    f = open(ftr.model.PSR.value+"_prof_pre.txt", 'w')
    for x, v in zip(xs, vs):
        f.write("%.5f  %12.5f\n" % (x, v))
    f.close()

    # Try normal optimization first to see how it goes
    if do_opt_first:
        result = op.minimize(ftr.minimize_func, np.zeros_like(ftr.fitvals))
        newfitvals = np.asarray(result['x']) * ftr.fiterrs + ftr.fitvals
        like_optmin = -result['fun']
        print("Optimization likelihood:", like_optmin)
        ftr.set_params(dict(zip(ftr.fitkeys, newfitvals)))
        ftr.phaseogram()
    else:
        like_optmin = -np.inf

    # Set up the initial conditions for the emcee walkers.  Use the
    # scipy.optimize newfitvals instead if they are better
    ndim = ftr.n_fit_params
    if like_start > like_optmin:
        # Keep the starting deviations small...
        pos = [ftr.fitvals + ftr.fiterrs/errfact * np.random.randn(ndim)
            for ii in range(nwalkers)]
        # Set starting params with uniform priors to uniform in the prior
        for param in ["GLPH_1", "GLEP_1", "SINI", "M2", "E", "ECC", "PX", "A1"]:
            if param in ftr.fitkeys:
                idx = ftr.fitkeys.index(param)
                if param=="GLPH_1":
                    svals = np.random.uniform(-0.5, 0.5, nwalkers)
                elif param=="GLEP_1":
                    svals = np.random.uniform(minMJD+100, maxMJD-100, nwalkers)
                    #svals = 55422.0 + np.random.randn(nwalkers)
                elif param=="SINI":
                    svals = np.random.uniform(0.0, 1.0, nwalkers)
                elif param=="M2":
                    svals = np.random.uniform(0.1, 0.6, nwalkers)
                elif param in ["E", "ECC", "PX", "A1"]:
                    # Ensure all positive
                    svals = np.fabs(ftr.fitvals[idx] + ftr.fiterrs[idx] *
                                    np.random.randn(nwalkers))
                    if param in ["E", "ECC"]:
                        svals[svals>1.0] = 1.0 - (svals[svals>1.0] - 1.0)
                for ii in range(nwalkers):
                    pos[ii][idx] = svals[ii]
    else:
        pos = [newfitvals + ftr.fiterrs/errfact*np.random.randn(ndim)
            for i in range(nwalkers)]
    # Set the 0th walker to have the initial pre-fit solution
    # This way, one walker should always be in a good position
    pos[0] = ftr.fitvals

    import emcee
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, ftr.lnposterior, threads=10)
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
    try:
        import corner
        samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
        fig = corner.corner(samples, labels=ftr.fitkeys, bins=50)
        fig.savefig(ftr.model.PSR.value+"_triangle.png")
        plt.close()
    except:
        pass

    # Make a phaseogram with the 50th percentile values
    #ftr.set_params(dict(zip(ftr.fitkeys, np.percentile(samples, 50, axis=0))))
    # Make a phaseogram with the best MCMC result
    ftr.set_params(dict(zip(ftr.fitkeys, ftr.maxpost_fitvals)))
    ftr.phaseogram(file=ftr.model.PSR.value+"_post.png")
    plt.close()


    # Write out the output pulse profile
    vs, xs = np.histogram(ftr.get_event_phases(), outprof_nbins, \
        range=[0,1], weights=ftr.weights)
    f = open(ftr.model.PSR.value+"_prof_post.txt", 'w')
    for x, v in zip(xs, vs):
        f.write("%.5f  %12.5f\n" % (x, v))
    f.close()

    # Write out the par file for the best MCMC parameter est
    f = open(ftr.model.PSR.value+"_post.par", 'w')
    f.write(ftr.model.as_parfile())
    f.close()

    # Print the best MCMC values and ranges
    ranges = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    print("Post-MCMC values (50th percentile +/- (16th/84th percentile):")
    for name, vals in zip(ftr.fitkeys, ranges):
        print("%8s:"%name, "%25.15g (+ %12.5g  / - %12.5g)"%vals)

    # Put the same stuff in a file
    f = open(ftr.model.PSR.value+"_results.txt", 'w')

    f.write("Post-MCMC values (50th percentile +/- (16th/84th percentile):\n")
    for name, vals in zip(ftr.fitkeys, ranges):
        f.write("%8s:"%name + " %25.15g (+ %12.5g  / - %12.5g)\n"%vals)

    f.write("\nMaximum likelihood par file:\n")
    f.write(ftr.model.as_parfile())
    f.close()

    import cPickle
    cPickle.dump(samples, open(ftr.model.PSR.value+"_samples.pickle", "wb"))
