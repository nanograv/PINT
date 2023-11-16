"""Markov Chain Monte Carlo fitting."""

import copy

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import vstack
from scipy.stats import norm, uniform
from loguru import logger as log

import pint.plot_utils as plot_utils
from pint.eventstats import hm, hmw
from pint.fitter import Fitter
from pint.models.priors import Prior
from pint.residuals import Residuals
from pint.templates.lctemplate import LCTemplate


__all__ = [
    "MCMCFitter",
    "MCMCFitterAnalyticTemplate",
    "MCMCFitterBinnedTemplate",
    "CompositeMCMCFitter",
]


def concat_toas(toas):
    """Concatenate a list of TOAs objects into a single TOAs object"""
    if len(toas) == 1:
        return toas[0]
    ts = copy.deepcopy(toas[0])
    for t in toas[1:]:
        print(len(ts.table), len(t.table))
        ts.table = vstack([ts.table, t.table])
        print("\t%d" % len(ts.table))
    ts.table = ts.table.group_by("obs")
    return ts


def lnprior_basic(ftr, theta):
    """Basic implementation of log prior.

    This will work for both analytic and
    binned templates, including when the template parameters are part of the
    search space.
    """
    theta_model = ftr.get_model_parameters(theta)
    theta_templ = ftr.get_template_parameters(theta)
    lnsum = 0.0
    for val, key in zip(theta_model, ftr.fitkeys):
        lnsum += getattr(ftr.model, key).prior_pdf(val, logpdf=True)
    # Loop over template parameters here: hard coded uniform for now
    if theta_templ is not None:
        for val, bounds in zip(theta_templ, ftr.tbounds):
            if (val < bounds[0]) or (val > bounds[1]):
                return -np.inf
    return lnsum


def lnlikelihood_basic(ftr, theta):
    """The log of the likelihood function, basic implementation.

    Assumes that the phase is the last parmameter in the parameter list
    """
    ftr.set_parameters(theta)
    phases = ftr.get_event_phases()
    phss = phases.astype(np.float64)

    phss[phss < 0] += 1.0
    phss[phss >= 1] -= 1.0

    probs = ftr.get_template_vals(phss)
    if ftr.weights is None:
        return np.log(probs).sum()
    else:
        return np.log(ftr.weights * probs + 1.0 - ftr.weights).sum()


def lnlikelihood_chi2(ftr, theta):
    ftr.set_parameters(theta)
    return -Residuals(toas=ftr.toas, model=ftr.model).calc_chi2()


def set_priors_basic(ftr, priorerrfact=10.0):
    """Basic method to set priors on parameters in the model.

    This adds a gaussian prior on each parameter with width equal to
    the par file uncertainty * priorerrfact and then puts in some special cases
    """
    fkeys, fvals, ferrs = ftr.fitkeys, ftr.fitvals, ftr.fiterrs
    for key, val, err in zip(fkeys, fvals, ferrs):
        if key in ["SINI", "E", "ECC"]:
            getattr(ftr.model, key).prior = Prior(uniform(0.0, 1.0))
        elif key == "PX":
            getattr(ftr.model, key).prior = Prior(uniform(0.0, 10.0))
        elif key.startswith("GLPH"):
            getattr(ftr.model, key).prior = Prior(uniform(-0.4, 1.0))
        else:
            if err == 0 and not getattr(ftr.model, key).frozen:
                ftr.priors_set = False
                raise ValueError(
                    f"Parameter {key} does not have uncertainty in par file"
                )
            getattr(ftr.model, key).prior = Prior(
                norm(loc=float(val), scale=float(err * priorerrfact))
            )
    ftr.priors_set = True


class MCMCFitter(Fitter):
    """A class for Markov-Chain Monte Carlo optimization style-fitting

    This fitting is similar to that implemented in event_optimize.py

    Parameters
    ----------
    toas : pint.toas.TOAs
    model
    sampler : pint.sampler.MCMCSampler
    template : object or None
        A template profile, for example, of a gaussian pulse.
        If template is none, then all template methods will do nothing,
        or raise an error, or return None. If a template is set, it is
        assumed that a subclass is being used.
    lnprior : callable
        The log prior function - defaults to lnprior above
    lnlike : callable
        The log likelihood function - defaults to lnlikelihood above
    setpriors : callable
        The function for setting the priors on model parameters
    weights : optional
        Weights for likelihood calculations
    minMJD : optional
        Minimium MJD in dataset (used sometimes for get_initial_pos)
    maxMJD : optional
        Maximum MJD in dataset (used sometimes for get_initial_pos)

    """

    def __init__(self, toas, model, sampler, **kwargs):
        super().__init__(toas, model, track_mode=kwargs.get("track_mode", None))
        self.toas = toas
        self.model_init = model
        self.use_resids = kwargs.get("resids", True)
        if self.use_resids:
            self.resids_init = Residuals(toas=toas, model=model)
            self.reset_model()
        else:
            self.model = model

        self.method = "MCMC"
        self.sampler = sampler

        self.lnprior = kwargs.get("lnprior", lnprior_basic)
        self.lnlikelihood = kwargs.get("lnlike", lnlikelihood_basic)
        self.set_priors = kwargs.get("setpriors", set_priors_basic)

        # Default values for these arguments were taken from event_optimize.py
        self.weights = kwargs.get("weights", None)
        self.minMJD = kwargs.get("minMJD", 40000)
        self.maxMJD = kwargs.get("maxMJD", 60000)

        self.fitkeys, self.fitvals, self.fiterrs = self.generate_fit_keyvals()
        self.n_fit_params = len(self.fitvals)

        template = kwargs.get("template", None)
        if template is not None:
            self.set_template(template)
        else:
            self.template = None

        log.info("Fit Keys:\t%s" % (self.fitkeys))
        log.info("Fit Vals:\t%s" % (self.fitvals))
        self.numcalls = 0
        self.maxpost = -np.inf
        self.maxpost_fitvals = self.fitvals
        self.priors_set = False

    def set_template(self, template):
        """
        Sets template and template metadata. Implementation depends on whether
        template is analytic or binned.
        """
        raise NotImplementedError

    def get_template_vals(self, phases):
        """
        Use the template (if it exists) to get probabilities for given phases
        """
        if self.template is None:
            return None
        raise NotImplementedError

    def clip_template_params(self, pos):
        """
        If template parameters are changeable, ensure that they are within bounds
        Any passing the template bounds will be clipped to the edges.
        If template is not being fit to, then this does nothing
        """
        if self.template is None:
            return pos
        raise NotImplementedError

    def get_model_parameters(self, theta):
        """
        Split the parameters related to the model
        """
        if self.template is None:
            return theta
        raise NotImplementedError

    def get_template_parameters(self, theta):
        """
        Split the parameters related to the template
        """
        if self.template is None:
            return None
        raise NotImplementedError

    def get_parameters(self):
        """
        Get all parameters for this fitter
        """
        if self.template is None:
            return self.fitvals
        raise NotImplementedError

    def get_parameter_names(self):
        """
        Get parameter names for this fitter
        """
        if self.template is None:
            return self.fitkeys
        raise NotImplementedError

    def set_parameters(self, theta):
        """
        Set timing and template parameters as necessary
        """
        if self.template is None:
            self.set_params(dict(zip(self.fitkeys, theta)))
        else:
            raise NotImplementedError

    def get_errors(self):
        """
        Get errors associated with all fit parameters
        """
        if self.template is None:
            return self.fiterrs
        raise NotImplementedError

    def get_fit_keyvals(self):
        """
        Basic getter, useful in event_optimize script
        """
        return self.fitkeys, self.fitvals, self.fiterrs

    def generate_fit_keyvals(self):
        """Read the model to determine fitted keys and their values and errors
        from the par file
        """
        fitkeys = [p for p in self.model.params if not getattr(self.model, p).frozen]
        fitvals = []
        fiterrs = []
        for p in fitkeys:
            fitvals.append(getattr(self.model, p).value)
            fiterrs.append(getattr(self.model, p).uncertainty_value)
        return fitkeys, np.asarray(fitvals), np.asarray(fiterrs)

    def get_weights(self):
        return self.weights

    def get_event_phases(self):
        """
        Return pulse phases based on the current model
        """
        phases = self.model.phase(self.toas).frac
        # ensure all positive
        return np.where(phases < 0.0, phases + 1.0, phases)

    def lnposterior(self, theta):
        """
        The log posterior (priors * likelihood)
        """
        self.numcalls += 1
        # Evaluate prior first. Don't compute posterior if prior is not finite
        lnprior = self.lnprior(self, theta)
        if not np.isfinite(lnprior):
            return -np.inf

        lnlikelihood = self.lnlikelihood(self, theta)
        lnpost = lnprior + lnlikelihood
        if lnpost > self.maxpost:
            log.info("New max: %f\tCall %d" % (lnpost, self.numcalls))
            for name, val in zip(self.fitkeys, theta):
                log.info("\t%8s: %25.15g" % (name, val))
            self.maxpost = lnpost
            self.maxpost_fitvals = theta
        return lnpost

    def minimize_func(self, theta):
        """Override superclass minimize_func to make compatible with scipy.optimize"""
        # Scale params based on errors
        ntheta = (self.get_model_parameters(theta) * self.fiterrs) + self.fitvals
        self.set_params(dict(zip(self.fitkeys, ntheta)))
        if not np.isfinite(self.lnprior(self, ntheta)):
            return np.inf
        lnlikelihood = self.lnlikelihood(self, theta)

        return -lnlikelihood

    def fit_toas(self, maxiter=100, pos=None, errfact=0.1, priorerrfact=10.0):
        """Fitting function - calls sampler.run_mcmc to converge using MCMC approach

        Parameters
        ----------
        maxiter : int
            The number of iterations to run_mcmc for
        pos
            The intiial position of the sampler. Default behavior calls
                sampler.get_initial_pos()
        errfact : float, optional
            Multiplicative factor for errors in get_intial_pos
        priorerrfact : float, optional
            Error factor in setting prior widths

        """
        # Set model priors if it hasn't been done yet
        if not self.priors_set:
            self.set_priors(self, priorerrfact)
        # Set initial positions for walkers if they haven't been specified
        if pos is None:
            pos = self.sampler.get_initial_pos(
                self.fitkeys,
                self.fitvals,
                self.fiterrs,
                errfact,
                minMJD=self.minMJD,
                maxMJD=self.maxMJD,
            )

        # If template exists, make sure that template params are within tbound
        pos = self.clip_template_params(pos)

        # Initialize sampler
        self.sampler.initialize_sampler(self.lnposterior, self.n_fit_params)

        # Run sampler for some number of iterations
        self.sampler.run_mcmc(pos, maxiter)

        # Process results and get chi2 for new parameters
        self.set_params(dict(zip(self.fitkeys, self.maxpost_fitvals)))
        if self.use_resids:
            self.resids.update()
        return self.lnposterior(self.maxpost_fitvals)

    def phaseogram(
        self, weights=None, bins=100, rotate=0.0, size=5, alpha=0.25, plotfile=None
    ):
        """Make a nice 2-panel phaseogram for the current model"""
        mjds = self.toas.table["tdbld"].data
        phss = self.get_event_phases()
        plot_utils.phaseogram(
            mjds,
            phss,
            weights=self.get_weights(),
            bins=bins,
            rotate=rotate,
            size=size,
            alpha=alpha,
            plotfile=plotfile,
        )

    def prof_vs_weights(self, nbins=50, use_weights=False):
        """
        Show binned profiles (and H-test values) as a function
        of the minimum weight used. nbins is only for the plots.
        """
        f, ax = plt.subplots(3, 3, sharex=True)
        phss = self.get_event_phases()
        htests = []
        weights = np.linspace(0.0, 0.95, 20)
        swgts = self.get_weights()
        for ii, minwgt in enumerate(weights):
            good = swgts > minwgt
            nphotons = np.sum(good)
            wgts = swgts[good] if use_weights else None
            if nphotons <= 0:
                hval = 0
            else:
                hval = hmw(phss[good], weights=wgts) if use_weights else hm(phss[good])
            htests.append(hval)
            if ii > 0 and ii % 2 == 0 and ii < 20:
                r, c = ((ii - 2) / 2) / 3, ((ii - 2) / 2) % 3
                ax[r][c].hist(
                    phss[good],
                    nbins,
                    range=[0, 1],
                    weights=wgts,
                    color="k",
                    histtype="step",
                )
                ax[r][c].set_title(
                    "%.1f / %.1f / %.0f" % (minwgt, hval, nphotons), fontsize=11
                )
                if c == 0:
                    ax[r][c].set_ylabel("Htest")
                if r == 2:
                    ax[r][c].set_xlabel("Phase")
                f.suptitle(
                    f"{self.model.PSR.value}:  Minwgt / H-test / Approx # events",
                    fontweight="bold",
                )
        if use_weights:
            plt.savefig(f"{self.model.PSR.value}_profs_v_wgtcut.png")
        else:
            plt.savefig(f"{self.model.PSR.value}_profs_v_wgtcut_unweighted.png")
        plt.close()
        plt.plot(weights, htests, "k")
        plt.xlabel("Min Weight")
        plt.ylabel("H-test")
        plt.title(self.model.PSR.value)
        if use_weights:
            plt.savefig(f"{self.model.PSR.value}_htest_v_wgtcut.png")
        else:
            plt.savefig(f"{self.model.PSR.value}_htest_v_wgtcut_unweighted.png")
        plt.close()

    def plot_priors(self, chains, burnin, bins=100, scale=False):
        plot_utils.plot_priors(
            self.model,
            chains,
            self.maxpost_fitvals,
            self.fitvals,
            burnin=burnin,
            bins=bins,
            scale=scale,
        )


class MCMCFitterBinnedTemplate(MCMCFitter):
    """A subclass of MCMCFitter, designed to use a binned template with
    interpolation instead of an analytic function
    """

    def __init__(self, toas, model, sampler, **kwargs):
        super().__init__(toas, model, sampler, **kwargs)

    def set_template(self, template):
        """
        Set template and template metadata. For binned template, we want to
        set the x values for interpolation purposes here.
        """
        self.template = template
        self.ltemp = len(template)
        self.xtemp = np.arange(self.ltemp) * 1.0 / self.ltemp

    def get_template_vals(self, phases):
        if self.template is None:
            raise ValueError("Template has not been initialized in MCMCFitter")
        return np.interp(phases, self.xtemp, self.template, right=self.template[0])

    def clip_template_params(self, pos):
        return pos

    def get_model_parameters(self, theta):
        return theta

    def get_template_parameters(self, theta):
        return None

    def get_parameters(self):
        return self.fitvals

    def get_parameter_names(self):
        return self.fitkeys

    def set_parameters(self, theta):
        self.set_params(dict(zip(self.fitkeys, theta)))

    def get_errors(self):
        return self.fiterrs


class MCMCFitterAnalyticTemplate(MCMCFitter):
    """A subclass of MCMCFitter, designed to use an analytic template rather
    than a binned one that uses interpolation.
    """

    def __init__(self, toas, model, sampler, template, **kwargs):
        super().__init__(toas, model, sampler, template=template, **kwargs)

    def set_template(self, template):
        """
        Metadata for analytic template
        """
        self.template = template
        self.tfitkeys = template.get_parameter_names()
        self.tfitvals = template.get_parameters()
        self.tfiterrs = template.get_errors()
        self.tbounds = template.get_bounds()
        self.n_fit_params = len(self.fitvals) + len(self.tfitvals)

    def get_template_vals(self, phases):
        return self.template(phases, use_cache=True)

    def clip_template_params(self, pos):
        nfitkeys = len(self.fitkeys)
        ret = pos
        for i in range(nfitkeys, self.n_fit_params):
            lo, hi = self.tbounds[i - nfitkeys]
            ret = np.clip(ret[:, i], lo, hi)
        return ret

    def get_parameter_names(self):
        return self.fitkeys + self.template.get_parameter_names()

    def get_model_parameters(self, theta):
        return theta[: len(self.fitkeys)]

    def get_template_parameters(self, theta):
        return theta[len(self.fitkeys) :]

    def get_parameters(self):
        return np.append(self.fitvals, self.tfitvals)

    def set_parameters(self, theta):
        self.set_params(dict(zip(self.fitkeys, self.get_model_parameters(theta))))
        self.template.set_parameters(self.get_template_parameters(theta))

    def get_errors(self):
        return np.append(self.fiterrs, self.tfiterrs)


class CompositeMCMCFitter(MCMCFitter):
    """A subclass of MCMCFitter, designed to work on composite datasets

    Requires a list of TOAs objects formed from different datafiles
    to make up the toas table, as well as a list of log-likelihood methods

    Here, the toas argument to the constructor is a list of TOAs objects,
    while the toas parameter for this class is a concatenated TOAs object
    containing all TOA information from all datasets

    The goal is to fit all of the data sets to a single model, so only one
    model is required in the construction of this object. In addition, only
    one sampler is required.

    Parameters
    ----------
    weights
        an array of weight lists for weighting individual TOAs
    set_weights
        an array of weights for each individual data set in
        toas_list. The basic lnlikelihood function will be given by
        lnlike = sum(setweight(i) * lnlike(toas_list(i)))
        Defaults to an array of 1s
    lnlikes
        a list of lnlikelihood functions to be used on each entry
        in toas_list. This is a required argument
    templates
         a list of templates for fitting to each individual dataset.
        Defaults to None for everything
        TODO: Add support for fitting templates here

    """

    def __init__(self, toas, model, sampler, lnlikes, **kwargs):
        self.toas_list = toas
        self.toas = concat_toas(toas)
        self.model = model
        self.method = "MCMC"
        self.sampler = sampler

        self.lnprior = kwargs.get("lnprior", lnprior_basic)
        self.lnlikelihoods = lnlikes
        self.set_priors = kwargs.get("setpriors", set_priors_basic)

        self.weights = kwargs.get("weights", [None] * len(self.toas_list))
        self.set_weights = kwargs.get("set_weights", [1.0] * len(self.toas_list))
        self.templates = kwargs.get("templates", [None] * len(self.toas_list))
        self.xtemps = [None] * len(self.toas_list)
        phs = kwargs.get("phs", 0.0)
        phserr = kwargs.get("phserr", 0.03)

        self.minMJD = kwargs.get("minMJD", 0)
        self.maxMJD = kwargs.get("maxMJD", 100000)

        self.fitkeys, self.fitvals, self.fiterrs = self.generate_fit_keyvals(
            phs, phserr
        )
        self.n_fit_params = len(self.fitvals)

        self.numcalls = 0
        self.maxpost = -np.inf
        self.maxpost_fitvals = self.fitvals
        self.priors_set = False
        self.use_resids = False

    def clip_template_params(self, pos):
        return pos

    def get_model_parameters(self, theta):
        return theta

    def get_template_parameters(self, theta):
        return None

    def get_parameters(self):
        return self.fitvals

    def get_parameter_names(self):
        return self.fitkeys

    def set_parameters(self, theta):
        self.set_params(dict(zip(self.fitkeys, theta)))

    def get_errors(self):
        return self.fiterrs

    def get_event_phases(self, index=None):
        """Get phases for the TOAs object specified by index in toas_list.

        If index is None, then it will return phases for all TOAs
        """
        if index is None:
            phases = self.model.phase(self.toas)[1]
            print("Showing all %d phases" % len(phases))
        else:
            phases = self.model.phase(self.toas_list[index])[1]
        return np.where(phases < 0.0, phases + 1.0, phases)

    def get_template_vals(self, phases, index):
        if self.templates[index] is None:
            raise ValueError(
                "Template for index %d has not been initialized in CompositeMCMCFitter"
                % index
            )
        if isinstance(self.templates[index], LCTemplate):
            return self.templates[index](phases, use_cache=True)
        if self.xtemps[index] is None:
            ltemp = len(self.templates[index])
            self.xtemps[index] = np.arange(ltemp) * 1.0 / ltemp
        return np.interp(
            phases,
            self.xtemps[index],
            self.templates[index],
            right=self.templates[index][0],
        )

    def get_weights(self, index=None):
        if index is not None:
            return self.weights[index]
        wgts = np.zeros(len(self.toas.table))
        curr = 0
        for i in range(len(self.toas_list)):
            ts = self.toas_list[i]
            nxt = curr + len(ts.table)
            print(curr, nxt, len(ts.table))
            wgts[curr:nxt] = (
                1.0 * self.set_weights[i]
                if self.weights[i] is None
                else self.weights[i] * self.set_weights[i]
            )
            curr = nxt
        return wgts

    def lnlikelihood(self, fitter, theta):
        """Sum over the log-likelihood functions for each dataset, multiply by weights in the sum.

        Note
        ----
        Requires a fitter passed because that is how this function is called by
        lnposterior in the super class.
        """
        self.set_parameters(theta)
        lnsum = 0.0
        curr = 0
        for i in range(len(self.lnlikelihoods)):
            lnsum += self.lnlikelihoods[i](self, theta, i) * self.set_weights[i]
        return lnsum
