import numpy as np
import astropy.units as u
import scipy.optimize as opt, scipy.linalg as sl
import matplotlib.pyplot as plt
import pint.plot_utils as plot_utils
from .residuals import resids
from pint.fitter import Fitter
from pint.scripts.event_optimize import profile_likelihood
from pint.models.priors import Prior
from scipy.stats import norm, uniform 
from astropy import log

class MCMCFitter(Fitter):
    """A class for Markov-Chain Monte Carlo optimization style-fitting,
        similar to that implemented in event_optimize.py

        toas
        model
        sampler - A subclass of pint.sampler.MCMCSampler
        template - A template profile, for example, of a gaussian pulse 
        weights - Weights for likelihood calculations
        phs - Pulse phase - to be added to the model (remove when phs is part of par files)
        phserr - Error associated with pulse phase
        minMJD - Minimium MJD in dataset (used sometimes for get_initial_pos)
        maxMJD - Maximum MJD in dataset (used sometimes for get_initial_pos)
        Default values for arguments were taken from event_optimize.py
    """
    def __init__(self, toas, model, sampler, template=None, weights=None, 
        phs=0.5, phserr=0.03, minMJD=54680, maxMJD=57250):
        super(MCMCFitter, self).__init__(toas, model)
        self.method = 'MCMC'
        self.sampler = sampler
        self.template = template
        if template is not None:
            self.ltemp = len(template)
            self.xtemp = np.arange(self.ltemp) * 1.0 / self.ltemp
        self.weights = weights
        self.fitkeys, self.fitvals, self.fiterrs = \
            self.generate_fit_keyvals(phs, phserr)
        log.info('Fit Keys:\t%s' % (self.fitkeys))
        log.info('Fit Vals:\t%s' % (self.fitvals))
        self.minMJD = minMJD
        self.maxMJD = maxMJD
        self.n_fit_params = len(self.fitvals)
        self.numcalls = 0
        self.nsteps = 1
        self.maxpost = -np.inf

    def set_template(self, template):
        """
        Sets template and template metadata
        """
        self.template = template
        self.ltemp = len(template)
        self.xtemp = np.arange(self.ltemp) * 1.0 / self.ltemp

    def get_fit_keyvals(self):
        """
        Basic getter, useful in event_optimize script
        """
        return self.fitkeys, self.fitvals, self.fiterrs

    def generate_fit_keyvals(self, phs, phserr):
        """Read the model to determine fitted keys and their values and errors 
            from the par file
        """
        fitkeys = [p for p in self.model.params if not 
            getattr(self.model,p).frozen]
        fitvals = []
        fiterrs = []
        for p in fitkeys:
            fitvals.append(getattr(self.model, p).value)
            fiterrs.append(getattr(self.model, p).uncertainty_value)
        #Last entry in the fit lists is the absolute PHASE term
        #Should be removed if PHASE is made a model param
        fitkeys.append("PHASE")
        fitvals.append(phs)
        fiterrs.append(phserr)
        return fitkeys, np.asarray(fitvals), np.asarray(fiterrs)

    def set_priors(self, priorerrfact=10.0):
        """
        Set priors on parameters in the model. This adds a gaussian prior
        on each parameter with width equal to the par file uncertainty * priorerrfact,
        and then puts in some special cases.
        """
        for key, v, e in zip(self.fitkeys[:-1], self.fitvals[:-1], self.fiterrs[:-1]):
            if key == 'SINI' or key == 'E' or key == 'ECC':
                getattr(self.model,key).prior = Prior(uniform(0.0, 1.0))
            elif key == 'PX':
                getattr(self.model,key).prior = Prior(uniform(0.0, 10.0))
            elif key.startswith('GLPH'):
                getattr(self.model,key).prior = Prior(uniform(-0.4, 1.0))
            else:
                getattr(self.model,key).prior = Prior(norm(loc=float(v),
                                                        scale=float(e*priorerrfact)))

    def get_event_phases(self):
        """
        Return pulse phases based on the current model
        """
        phases = self.model.phase(self.toas.table)[1]
        # ensure all positive
        return np.where(phases < 0.0 * u.cycle, phases + 1.0 * u.cycle, phases)

    def lnprior(self, theta):
        """
        The log prior evaluated at the parameter values specified
        """
        lnsum = 0.0
        for val, key in zip(theta[:-1], self.fitkeys[:-1]):
            lnsum += getattr(self.model,key).prior_pdf(val,logpdf=True)
        #Add phase term
        if theta[-1] > 1.0 or theta[-1] < 0.0:
            return np.inf
        return lnsum

    def lnposterior(self, theta):
        """
        The log posterior (priors * likelihood)
        """
        self.set_params(dict(zip(self.fitkeys[:-1], theta[:-1])))

        self.numcalls += 1

        #Evaluate prior first. Don't compute posterior if prior is not finite
        lnprior = self.lnprior(theta)
        if not np.isfinite(lnprior):
            return -np.inf

        #Compute phases using PINT
        phases = self.get_event_phases()
        lnlikelihood = profile_likelihood(theta[-1], self.xtemp,
                                          phases, self.template, self.weights)
        lnpost = lnprior + lnlikelihood
        if lnpost > self.maxpost:
            log.info("New max: %f" % lnpost)
            for name, val in zip(self.fitkeys, theta):
                log.info("\t%8s: %25.15g" % (name, val))
            self.maxpost = lnpost
            self.maxpost_fitvals = theta
        return lnpost

    def minimize_func(self, theta):
        """
        Override superclass minimize_func to make compatible with scipy.optimize
        """
        #Scale params based on errors
        ntheta = (theta[:-1] * self.fiterrs[:-1]) + self.fitvals[:-1]
        self.set_params(dict(zip(self.fitkeys[:-1], ntheta)))
        if not np.isfinite(self.lnprior(ntheta)):
            return np.inf
        phases = self.get_event_phases()
        lnlikelihood = profile_likelihood(theta[-1], self.xtemp,
                                          phases, self.template, self.weights)
        return -lnlikelihood
    
    def fit_toas(self, maxiter=100, pos=None, errfact=0.1, priorerrfact=10.0):
        """
        Fitting function - calls sampler.run_mcmc to converge using MCMC approach

            maxiter - The number of iterations to run_mcmc for
            pos - The intiial position of the sampler. Default behavior calls
                    sampler.get_initial_pos()
            errfact - Multiplicative factor for errors in get_intial_pos
            priorerrfact - Error factor in setting prior widths
        """
        self.set_priors(priorerrfact)
        if pos is None:
            pos = self.sampler.get_initial_pos(self.fitkeys, self.fitvals, self.fiterrs, 
                errfact, minMJD=self.minMJD, maxMJD=self.maxMJD)

        #Initialize sampler
        self.sampler.initialize_sampler(self.lnposterior, self.n_fit_params)

        #Run sampler for some number of iterations
        self.sampler.run_mcmc(pos, maxiter)

        #Process results and get chi2 for new parameters
        self.set_params(dict(zip(self.fitkeys[:-1], self.maxpost_fitvals[:-1])))
        self.resids.update()
        return self.lnposterior(self.maxpost_fitvals)
    
    def phaseogram(self, weights=None, bins=100, rotate=0.0, size=5,
        alpha=0.25, plotfile=None):
        """
        Make a nice 2-panel phaseogram for the current model
        """
        mjds = self.toas.table['tdbld'].quantity
        phss = self.get_event_phases()
        plot_utils.phaseogram(mjds, phss, weights=self.weights, bins=bins,
            rotate=rotate, size=size, alpha=alpha, plotfile=plotfile)

    def prof_vs_weights(self, nbins=50, use_weights=False):
        """
        Show binned profiles (and H-test values) as a function
        of the minimum weight used. nbins is only for the plots.
        """
        f, ax = plt.subplots(3, 3, sharex=True)
        phss = self.get_event_phases()
        htests = []
        weights = np.linspace(0.0, 0.95, 20)
        for ii, minwgt in enumerate(weights):
            good = self.weights > minwgt
            nphotons = np.sum(good)
            wgts = self.weights[good] if use_weights else None
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
            plt.savefig(self.model.PSR.value+"_profs_v_wgtcut.png")
        else:
            plt.savefig(self.model.PSR.value+"_profs_v_wgtcut_unweighted.png")
        plt.close()
        plt.plot(weights, htests, 'k')
        plt.xlabel("Min Weight")
        plt.ylabel("H-test")
        plt.title(self.model.PSR.value)
        if use_weights:
            plt.savefig(self.model.PSR.value+"_htest_v_wgtcut.png")
        else:
            plt.savefig(self.model.PSR.value+"_htest_v_wgtcut_unweighted.png")
        plt.close()
