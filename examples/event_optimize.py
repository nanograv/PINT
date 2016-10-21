#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
import numpy as np
import pint.toa as toa
import pint.models
import pint.fitter as fitter
import pint.fermi_toas as fermi
from pint.eventstats import hmw, hm
from pint.models.priors import Prior, UniformRV, UniformBoundedRV, GaussianBoundedRV
from scipy.stats import norm
import matplotlib.pyplot as plt
import astropy.table
import astropy.units as u
import psr_utils as pu
import scipy.optimize as op
import sys, os, copy, fftfit
from astropy.coordinates import SkyCoord
from astropy import log
import argparse

#log.setLevel('DEBUG')
#np.seterr(all='raise')
        
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

def get_fit_keyvals(model):
    """Read the model to determine fitted keys and their values and errors from the par file
    """
    fitkeys = [p for p in model.params if not
        getattr(model,p).frozen]
    fitvals = []
    fiterrs = []
    for p in fitkeys:
        fitvals.append(getattr(model, p).value)
        fiterrs.append(getattr(model, p).uncertainty_value)
    return fitkeys, np.asarray(fitvals), np.asarray(fiterrs)

class emcee_fitter(fitter.fitter):

    def __init__(self, toas=None, model=None, template=None, weights=None):
        self.toas = toas
        self.model_init = model
        self.reset_model()
        self.template = template
        self.weights = weights
        self.fitkeys, self.fitvals, self.fiterrs = get_fit_keyvals(self.model)
        self.n_fit_params = len(self.fitvals)

    def get_event_phases(self):
        """
        Return pulse phases based on the current model
        """
        phss = self.model.phase(self.toas.table)[1]
        # ensure all postive
        return np.where(phss < 0.0, phss + 1.0, phss)

      
    def lnprior(self, theta):
        """
        The log prior evaulated at the parameter values specified
        """
        lnsum = 0.0
        for val, key in zip(theta, self.fitkeys):
            lnsum += getattr(self.model,key).prior_pdf(val,logpdf=True)
        return lnsum

    def lnposterior(self, theta):
        """
        The log posterior (priors * likelihood)
        """
        global maxpost, numcalls
        self.set_params(dict(zip(self.fitkeys, theta)))

        numcalls += 1
        if numcalls % (nwalkers * nsteps / 100) == 0:
            print "~%d%% complete" % (numcalls / (nwalkers * nsteps / 100))

        # Evaluate the prior FIRST, then don't even both computing
        # the posterior if the prior is not finite
        lnprior = self.lnprior(theta)
        if not np.isfinite(lnprior):
            return -np.inf

        phases = self.get_event_phases()
        lnlikelihood = marginalize_over_phase(phases, self.template,
            weights=self.weights)[1]
        lnpost = lnprior + lnlikelihood
        if lnpost > maxpost:
            print "New max: ", lnpost
            for name, val in zip(ftr.fitkeys, theta):
                    print "  %8s: %25.15g" % (name, val)
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
        if not np.isfinite(self.lnprior(ntheta)):
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PINT tool for MCMC optimization of timing models using event data.")
    
    parser.add_argument("eventfile",help="event file to use")
    parser.add_argument("parfile",help="par file to read model from")
    parser.add_argument("gaussianfile",help="gaussian file that defines template")
    parser.add_argument("--weightcol",help="name of weight column (or 'CALC' to have them computed",default=None)
    parser.add_argument("--nwalkers",help="Number of MCMC walkers (def 200)",type=int,
        default=200)
    parser.add_argument("--burnin",help="Number of MCMC steps for burn in (def 100)",
        type=int, default=100)
    parser.add_argument("--nsteps",help="Number of MCMC steps to compute (def 1000)",
        type=int, default=1000)
    parser.add_argument("--minMJD",help="Earliest MJD to use (def 54680)",type=float,
        default=54680.0)
    parser.add_argument("--maxMJD",help="Latest MJD to use (def 57250)",type=float,
        default=57250.0)
    parser.add_argument("--minWeight",help="Minimum weight to include (def 0.05)",
        type=float,default=0.05)
    parser.add_argument("--wgtexp", 
        help="Raise computed weights to this power (or 0.0 to disable any rescaling of weights)", 
        type=float, default=0.0)
    parser.add_argument("--testWeights",help="Make plots to evalute weight cuts?",
        default=False,action="store_true")
    parser.add_argument("--doOpt",help="Run initial scipy opt before MCMC?",
        default=False,action="store_true")
    parser.add_argument("--initerrfact",help="Multiply par file errors by this factor when initializing walker starting values",type=float,default=0.1)
    parser.add_argument("--priorerrfact",help="Multiple par file errors by this factor when setting gaussian prior widths",type=float,default=10.0)
    parser.add_argument("--usepickle",help="Read events from pickle file, if available?",
        default=False,action="store_true")
   
    args = parser.parse_args()

    eventfile = args.eventfile
    parfile = args.parfile
    gaussianfile = args.gaussianfile
    weightcol = args.weightcol

    nwalkers = args.nwalkers
    burnin = args.burnin
    nsteps = args.nsteps
    if burnin >= nsteps:
        log.error('burnin must be < nsteps')
        sys.exit(1)
    nbins = 256 # For likelihood calculation based on gaussians file
    outprof_nbins = 256 # in the text file, for pygaussfit.py, for instance
    minMJD = args.minMJD
    maxMJD = args.maxMJD # Usually set by coverage of IERS file

    minWeight = args.minWeight
    do_opt_first = args.doOpt
    wgtexp = args.wgtexp

    # initialization values
    # Should probably figure a way to make these not global variables
    maxpost = -9e99
    numcalls = 0
    
    # Read in initial model
    modelin = pint.models.get_model(parfile)
    # Remove the dispersion delay as it is unnecessary
    modelin.delay_funcs['L1'].remove(modelin.dispersion_delay)
    # Set the target coords for automatic weighting if necessary
    target = SkyCoord(modelin.RAJ.value, modelin.DECJ.value, \
        frame='icrs') if weightcol=='CALC' else None

    # TODO: make this properly handle long double
    if not args.usepickle or (not (os.path.isfile(eventfile+".pickle") or
        os.path.isfile(eventfile+".pickle.gz"))):
        # Read event file and return list of TOA objects
        tl = fermi.load_Fermi_TOAs(eventfile, weightcolumn=weightcol,
                                   targetcoord=target, minweight=minWeight)
        # Limit the TOAs to ones in selected MJD range and above minWeight
        tl = [tl[ii] for ii in range(len(tl)) if (tl[ii].mjd.value > minMJD and tl[ii].mjd.value < maxMJD
            and (weightcol is None or tl[ii].flags['weight'] > minWeight))]
        print "There are %d events we will use" % len(tl)
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
            print "Original weights have min / max weights %.3f / %.3f" % \
                (weights.min(), weights.max())
            # Rescale the weights, if requested (by having wgtexp != 0.0)
            if wgtexp != 0.0:
                weights **= wgtexp
                wmx, wmn = weights.max(), weights.min()
                # make the highest weight = 1, but keep min weight the same
                weights = wmn + ((weights - wmn) * (1.0 - wmn) / (wmx - wmn))
            for ii, x in enumerate(ts.table['flags']):
                x['weight'] = weights[ii]
        weights = np.asarray([x['weight'] for x in ts.table['flags']])
        print "There are %d events, with min / max weights %.3f / %.3f" % \
            (len(weights), weights.min(), weights.max())
    else:
        weights = None
        print "There are %d events, no weights are being used." % (len(weights))

    # Now load in the gaussian template and normalize it
    gtemplate = pu.read_gaussfitfile(gaussianfile, nbins)
    gtemplate /= gtemplate.sum()


    # Set the priors on the parameters in the model, before
    # instantiating the emcee_fitter
    # Currently, this adds a gaussian prior on each parameter
    # with width equal to the par file uncertainty * priorerrfact,
    # and then puts in some special cases.
    # *** This should be replaced/supplemented with a way to specify
    # more general priors on parameters that need certain bounds
    fitkeys, fitvals, fiterrs = get_fit_keyvals(modelin)

    for key, v, e in zip(fitkeys,fitvals,fiterrs):
        if key == 'SINI' or key == 'E' or key == 'ECC':
            getattr(modelin,key).prior = Prior(UniformBoundedRV(0.0,1.0))
        elif key == 'PX':
            getattr(modelin,key).prior = Prior(UniformBoundedRV(0.0,10.0))
        elif key.startswith('GLPH'):
            getattr(modelin,key).prior = Prior(UniformBoundedRV(-0.5,0.5))
        else:
            getattr(modelin,key).prior = Prior(norm(loc=float(v),scale=float(e*args.priorerrfact)))

    # Now define the requirements for emcee
    ftr = emcee_fitter(ts, modelin, gtemplate, weights)

    # Use this if you want to see the effect of setting minWeight
    if args.testWeights:
        log.info("Checking H-test vs weights")
        ftr.prof_vs_weights(use_weights=True)
        ftr.prof_vs_weights(use_weights=False)
        sys.exit()

    # Now compute the photon phases and see if we see a pulse
    phss = ftr.get_event_phases()
    maxbin, like_start = marginalize_over_phase(phss, gtemplate,
        weights=ftr.weights, minimize=True, showplot=False)
    print "Starting pulse likelihood:", like_start
    ftr.phaseogram(file=ftr.model.PSR.value+"_pre.png")
    plt.close()
    #ftr.phaseogram()

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
        print "Optimization likelihood:", like_optmin
        ftr.set_params(dict(zip(ftr.fitkeys, newfitvals)))
        ftr.phaseogram()
    else:
        like_optmin = -np.inf

    # Set up the initial conditions for the emcee walkers.  Use the
    # scipy.optimize newfitvals instead if they are better
    ndim = ftr.n_fit_params
    if like_start > like_optmin:
        # Keep the starting deviations small...
        pos = [ftr.fitvals + ftr.fiterrs/args.initerrfact * np.random.randn(ndim)
            for ii in range(nwalkers)]
        # Set starting params
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
        pos = [newfitvals + ftr.fiterrs/args.initerrfact*np.random.randn(ndim)
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
        npts = len(chain_dict)
        fig, axes = plt.subplots(npts, 1, sharex=True, figsize=(8, 9))
        for ii, name in enumerate(chain_dict.keys()):
            axes[ii].plot(chain_dict[name], color="k", alpha=0.3)
            axes[ii].set_ylabel(name)
        axes[npts-1].set_xlabel("Step Number")
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
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    try:
        import corner
        # Note, I had to turn off plot_contours because I kept getting
        # errors about how contour levels must be increasing.
        fig = corner.corner(samples, labels=ftr.fitkeys, bins=20,
            truths=ftr.maxpost_fitvals, plot_contours=False)
        fig.savefig(ftr.model.PSR.value+"_triangle.png")
        plt.close()
    except:
        log.warning("Corner plot failed")

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
    print "Post-MCMC values (50th percentile +/- (16th/84th percentile):"
    for name, vals in zip(ftr.fitkeys, ranges):
        print "%8s:"%name, "%25.15g (+ %12.5g  / - %12.5g)"%vals

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
