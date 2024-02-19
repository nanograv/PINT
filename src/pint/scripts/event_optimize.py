#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
import argparse
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from astropy.coordinates import SkyCoord
from scipy.stats import norm, uniform
import pint.logging
from loguru import logger as log

pint.logging.setup(level=pint.logging.script_level)

import pint.fermi_toas as fermi
import pint.models
import pint.plot_utils as plot_utils
import pint.toa as toa
from pint.eventstats import hm, hmw
from pint.fitter import Fitter
from pint.models.priors import Prior
from pint.observatory.satellite_obs import get_satellite_observatory


__all__ = ["read_gaussfitfile", "marginalize_over_phase", "main"]
# log.setLevel('DEBUG')
# np.seterr(all='raise')

# initialization values
# Should probably figure a way to make these not global variables
maxpost = -9e99
numcalls = 0


class custom_timing(
    pint.models.spindown.Spindown, pint.models.astrometry.AstrometryEcliptic
):
    def __init__(self, parfile):
        super().__init__()
        self.read_parfile(parfile)


def read_gaussfitfile(gaussfitfile, proflen):
    """Read a Gaussian-fit file as created by the output of pygaussfit.py.

    Parameters
    ----------
    gaussfitfile : str
        Name of the input file.
    proflen : int
        The number of bins to include in the resulting template.

    Returns
    -------
    np.array
        A template of length ``proflen``.
    """
    phass = []
    ampls = []
    fwhms = []
    for line in open(gaussfitfile):
        if line.lstrip().startswith("phas"):
            phass.append(float(line.split()[2]))
        if line.lstrip().startswith("ampl"):
            ampls.append(float(line.split()[2]))
        if line.lstrip().startswith("fwhm"):
            fwhms.append(float(line.split()[2]))
    if not (len(phass) == len(ampls) == len(fwhms)):
        log.warning(
            f"Number of phases, amplitudes, and FWHMs are not the same in '{gaussfitfile}'!"
        )
        return 0.0
    phass = np.asarray(phass)
    ampls = np.asarray(ampls)
    fwhms = np.asarray(fwhms)
    # Now sort them all according to decreasing amplitude
    new_order = np.argsort(ampls)
    new_order = new_order[::-1]
    ampls = np.take(ampls, new_order)
    phass = np.take(phass, new_order)
    fwhms = np.take(fwhms, new_order)
    # Now put the biggest gaussian at phase = 0.0
    phass = phass - phass[0]
    phass %= 1
    template = np.zeros(proflen, dtype="d")
    for ii in range(len(ampls)):
        template += ampls[ii] * gaussian_profile(proflen, phass[ii], fwhms[ii])
    return template


def gaussian_profile(N, phase, fwhm):
    """Return a gaussian pulse profile with 'N' bins and an integrated 'flux' of 1 unit.

    Parameters
    ----------
    N : int
        the number of points in the profile
    phase : float
        the pulse phase (0-1)
    fwhm : float
        the gaussian pulses full width at half-max


    Note
    ----
    The FWHM of a gaussian is approx 2.35482 sigma.
    """
    sigma = fwhm / 2.35482
    mean = phase % 1.0
    phsval = np.arange(N, dtype="d") / float(N)
    if mean < 0.5:
        phsval = np.where(phsval > (mean + 0.5), phsval - 1.0, phsval)
    else:
        phsval = np.where(phsval < (mean - 0.5), phsval + 1.0, phsval)
    try:
        zs = (phsval - mean) / sigma
        okzinds = np.compress(np.fabs(zs) < 20.0, np.arange(N))
        okzs = np.take(zs, okzinds)
        retval = np.zeros(N, "d")
        np.put(
            retval, okzinds, np.exp(-0.5 * (okzs) ** 2.0) / (sigma * np.sqrt(2 * np.pi))
        )
        return retval
    except OverflowError:
        log.warning("Problem in gaussian prof:  mean = %f  sigma = %f" % (mean, sigma))
        return np.zeros(N, "d")


def measure_phase(profile, template, rotate_prof=True):
    """
    measure_phase(profile, template):
        Call FFTFIT on the profile and template to determine the
            following parameters: shift,eshift,snr,esnr,b,errb,ngood
            (returned as a tuple).  These are defined as in Taylor's
            talk at the Royal Society.
    """
    import fftfit

    c, amp, pha = fftfit.cprof(template)
    pha1 = pha[0]
    if rotate_prof:
        pha = np.fmod(pha - np.arange(1, len(pha) + 1) * pha1, 2.0 * np.pi)
    shift, eshift, snr, esnr, b, errb, ngood = fftfit.fftfit(profile, amp, pha)
    return shift, eshift, snr, esnr, b, errb, ngood


def profile_likelihood(phs, *otherargs):
    """
    A single likelihood calc for matching phases to a template.
    Likelihood is calculated as per eqn 2 in Pletsch & Clark 2015.
    """
    xvals, phases, template, weights = otherargs
    phss = phases.astype(np.float64) + phs
    phss %= 1
    probs = np.interp(phss, xvals, template, right=template[0])
    if weights is None:
        return np.log(probs).sum()
    else:
        return np.log(weights * probs + 1.0 - weights).sum()


def neg_prof_like(phs, *otherargs):
    return -profile_likelihood(phs, *otherargs)


def marginalize_over_phase(
    phases,
    template,
    weights=None,
    resolution=1.0 / 1024,
    minimize=True,
    fftfit=False,
    showplot=False,
    lophs=0.0,
    hiphs=1.0,
):
    """Find the best fit pulse profile

    a pulse profile comprised of combined photon phases.  A maximum
    likelood technique is used.  The shift and the max log likehood
    are returned.  You probably want to use "minimize" rathre than
    "fftfit" unless you are only sampling very close to your known min.
    """
    ltemp = len(template)
    xtemp = np.arange(ltemp) * 1.0 / ltemp
    if minimize:
        phs, like = marginalize_over_phase(
            phases,
            template,
            weights,
            resolution=1.0 / 64,
            minimize=False,
            showplot=showplot,
        )
        phs = 1.0 - phs / ltemp
        hwidth = 0.03
        lophs, hiphs = phs - hwidth, phs + hwidth
        result = op.minimize(
            neg_prof_like,
            [phs],
            args=(xtemp, phases, template, weights),
            bounds=[[lophs, hiphs]],
        )
        return ltemp - result["x"] * ltemp, -result["fun"]
    if fftfit:
        deltabin = 3
        h, x = np.histogram(
            phases.astype(np.float64), ltemp, range=[0.0, 1.0], weights=weights
        )
        s, es, snr, esnr, b, errb, ngood = measure_phase(h, template, rotate_prof=False)
        # s is in bins based on the template size
        lophs = (ltemp - s - deltabin) / float(ltemp)  # bins below
        if lophs < 0.0:
            lophs += 1.0
        hiphs = lophs + 2.0 * deltabin / float(ltemp)  # bins above
    dphss = np.arange(lophs, hiphs, resolution)
    trials = phases.astype(np.float64) + dphss[:, np.newaxis]
    # ensure that all the phases are within 0-1
    trials[trials > 1.0] -= 1.0
    probs = np.interp(trials, xtemp, template, right=template[0])
    if weights is None:
        lnlikes = (np.log(probs)).sum(axis=1)
    else:
        lnlikes = (np.log(weights * probs + 1.0 - weights)).sum(axis=1)
    if showplot:
        plt.plot(dphss, lnlikes)
        plt.xlabel("Pulse Phase")
        plt.ylabel("Log likelihood")
        plt.show()
    return ltemp - dphss[lnlikes.argmax()] * ltemp, lnlikes.max()


def get_fit_keyvals(model, phs=0.0, phserr=0.1):
    """Read the model to determine fitted keys and their values and errors from the par file"""
    fitkeys = [p for p in model.params if not getattr(model, p).frozen]
    fitvals = []
    fiterrs = []
    for p in fitkeys:
        fitvals.append(getattr(model, p).value)
        fiterrs.append(getattr(model, p).uncertainty_value)
    # The last entry in each of the fit lists is our absolute PHASE term
    # Hopefully this will become a full PINT model param soon.
    fitkeys.append("PHASE")
    fitvals.append(phs)
    fiterrs.append(phserr)
    return fitkeys, np.asarray(fitvals), np.asarray(fiterrs)


def run_sampler_autocorr(sampler, pos, nsteps, burnin, csteps=100, crit1=10):
    """Runs the sampler and checks for chain convergence. Return the converged sampler and the mean autocorrelation time per 100 steps
    Parameters
    ----------
    Sampler
        The Emcee Ensemble Sampler
    pos
        The Initial positions of the walkers
    nsteps : int
        The number of integration steps
    csteps : int
        The interval at which the autocorrelation time is computed.
    crit1 : int
        The ratio of chain length to autocorrelation time to satisfy convergence
    Returns
    -------
    The sampler and the mean autocorrelation times
    Note
    ----
    The function checks for convergence of the chains every specified number of steps.
    The criteria to check for convergence is:
        1. the chain has to be longer than the specified ratio times the estimated autocorrelation time
        2. the change in the estimated autocorrelation time is less than 1%
    """
    autocorr = []
    old_tau = np.inf
    converged1 = False
    converged2 = False
    for sample in sampler.sample(pos, iterations=nsteps, progress=True):
        if not converged1:
            # Checks if the iteration is past the burnin and checks for convergence at 10% tau change
            if sampler.iteration >= burnin and sampler.iteration % csteps == 0:
                tau = sampler.get_autocorr_time(tol=0, quiet=True)
                if np.any(np.isnan(tau)):
                    continue
                else:
                    x = np.mean(tau)
                    autocorr.append(x)
                    converged1 = np.all(tau * crit1 < sampler.iteration)
                    converged1 &= np.all(np.abs(old_tau - tau) / tau < 0.1)
                    # log.info("The mean estimated integrated autocorrelation step is: " + str(x))
                    old_tau = tau
                    if converged1:
                        log.info(
                            "10 % convergence reached with a mean estimated integrated step: "
                            + str(x)
                        )
                    else:
                        continue
            else:
                continue
        else:
            if not converged2:
                # Checks for convergence at every 25 steps instead of 100 and tau change is 1%
                if sampler.iteration % int(csteps / 4) == 0:
                    tau = sampler.get_autocorr_time(tol=0, quiet=True)
                    if np.any(np.isnan(tau)):
                        continue
                    else:
                        x = np.mean(tau)
                        autocorr.append(x)
                        converged2 = np.all(tau * crit1 < sampler.iteration)
                        converged2 &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                        # log.info("The mean estimated integrated autocorrelation step is: " + str(x))
                        old_tau = tau
                        converge_step = sampler.iteration
                else:
                    continue
            if converged2 and (sampler.iteration - burnin) >= 1000:
                log.info(f"Convergence reached at {converge_step}")
                break
            else:
                continue
    return autocorr


class emcee_fitter(Fitter):
    def __init__(
        self, toas=None, model=None, template=None, weights=None, phs=0.5, phserr=0.03
    ):
        # super().__init__(model=model, toas=toas)
        self.toas = toas
        self.model = model
        self.template = template
        if template is not None:
            self.ltemp = len(template)
            self.xtemp = np.arange(self.ltemp) * 1.0 / self.ltemp
        self.weights = weights
        self.fitkeys, self.fitvals, self.fiterrs = get_fit_keyvals(
            self.model, phs, phserr
        )
        self.n_fit_params = len(self.fitvals)

    def get_event_phases(self):
        """
        Return pulse phases based on the current model
        """
        phss = self.model.phase(self.toas).frac
        return phss.value % 1

    def lnprior(self, theta):
        """
        The log prior evaulated at the parameter values specified
        """
        lnsum = 0.0
        for val, key in zip(theta[:-1], self.fitkeys[:-1]):
            lnsum += getattr(self.model, key).prior_pdf(val, logpdf=True)
        # Add the phase term
        return -np.inf if theta[-1] > 1.0 or theta[-1] < 0.0 else lnsum

    def lnposterior(self, theta):
        """
        The log posterior (priors * likelihood)
        """
        global maxpost, numcalls, ftr
        self.set_params(dict(zip(self.fitkeys[:-1], theta[:-1])))

        numcalls += 1
        if numcalls % (nwalkers * nsteps / 100) == 0:
            log.info("~%d%% complete" % (numcalls / (nwalkers * nsteps / 100)))

        # Evaluate the prior FIRST, then don't even both computing
        # the posterior if the prior is not finite
        lnprior = self.lnprior(theta)
        if not np.isfinite(lnprior):
            return -np.inf, -np.inf, -np.inf

        # Call PINT to compute the phases
        phases = self.get_event_phases()
        lnlikelihood = profile_likelihood(
            theta[-1], self.xtemp, phases, self.template, self.weights
        )
        lnpost = lnprior + lnlikelihood
        if lnpost > maxpost:
            log.info("New max: %f" % lnpost)
            for name, val in zip(ftr.fitkeys, theta):
                log.info("  %8s: %25.15g" % (name, val))
            maxpost = lnpost
            self.maxpost_fitvals = theta
        return lnpost, lnprior, lnlikelihood

    def minimize_func(self, theta):
        """
        Returns -log(likelihood) so that we can use scipy.optimize.minimize
        """
        # first scale the params based on the errors
        ntheta = (theta[:-1] * self.fiterrs[:-1]) + self.fitvals[:-1]
        self.set_params(dict(zip(self.fitkeys[:-1], ntheta)))
        if not np.isfinite(self.lnprior(ntheta)):
            return np.inf
        phases = self.get_event_phases()
        lnlikelihood = profile_likelihood(
            theta[-1], self.xtemp, phases, self.template, self.weights
        )
        return -lnlikelihood

    def phaseogram(
        self, weights=None, bins=100, rotate=0.0, size=5, alpha=0.25, plotfile=None
    ):
        """
        Make a nice 2-panel phaseogram for the current model
        """
        mjds = self.toas.table["tdbld"].data
        phss = self.get_event_phases()
        plot_utils.phaseogram(
            mjds,
            phss,
            weights=self.weights,
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
        global ftr
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
                hval = hmw(phss[good], weights=wgts) if use_weights else hm(phss[good])
            htests.append(hval)
            if ii > 0 and ii % 2 == 0 and ii < 20:
                r, c = ((ii - 2) // 2) // 3, ((ii - 2) // 2) % 3
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
            plt.savefig(f"{ftr.model.PSR.value}_profs_v_wgtcut.png")
        else:
            plt.savefig(f"{ftr.model.PSR.value}_profs_v_wgtcut_unweighted.png")
        plt.close()
        plt.plot(weights, htests, "k")
        plt.xlabel("Min Weight")
        plt.ylabel("H-test")
        plt.title(self.model.PSR.value)
        if use_weights:
            plt.savefig(f"{ftr.model.PSR.value}_htest_v_wgtcut.png")
        else:
            plt.savefig(f"{ftr.model.PSR.value}_htest_v_wgtcut_unweighted.png")
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PINT tool for MCMC optimization of timing models using event data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("eventfile", help="event file to use")
    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("gaussianfile", help="gaussian file that defines template")
    parser.add_argument("--ft2", help="Path to FT2 file.", default=None)
    parser.add_argument(
        "--weightcol",
        help="name of weight column (or 'CALC' to have them computed)",
        default=None,
    )
    parser.add_argument(
        "--nwalkers", help="Number of MCMC walkers", type=int, default=200
    )
    parser.add_argument(
        "--burnin",
        help="Number of MCMC steps for burn in",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--nsteps",
        help="Number of MCMC steps to compute",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--minMJD", help="Earliest MJD to use", type=float, default=54680.0
    )
    parser.add_argument(
        "--maxMJD", help="Latest MJD to use", type=float, default=57250.0
    )
    parser.add_argument(
        "--phs", help="Starting phase offset [0-1] (def is to measure)", type=float
    )
    parser.add_argument(
        "--phserr", help="Error on starting phase", type=float, default=0.03
    )
    parser.add_argument(
        "--minWeight",
        help="Minimum weight to include",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--wgtexp",
        help="Raise computed weights to this power (or 0.0 to disable any rescaling of weights)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--testWeights",
        help="Make plots to evalute weight cuts?",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--doOpt",
        help="Run initial scipy opt before MCMC?",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--initerrfact",
        help="Multiply par file errors by this factor when initializing walker starting values",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--priorerrfact",
        help="Multiple par file errors by this factor when setting gaussian prior widths",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--usepickle",
        help="Read events from pickle file, if available?",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=pint.logging.levels,
        default=pint.logging.script_level,
        help="Logging level",
        dest="loglevel",
    )
    parser.add_argument(
        "-v", "--verbosity", default=0, action="count", help="Increase output verbosity"
    )
    parser.add_argument(
        "-q", "--quiet", default=0, action="count", help="Decrease output verbosity"
    )
    parser.add_argument(
        "--multicore",
        default=False,
        action="store_true",
        help="Run event optimize on multiple cores",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=8,
        help="The number of cores for parallel processing",
    )
    parser.add_argument(
        "--backend",
        help="Save chains to a h5 file",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        help="File path to save all output files to",
    )
    parser.add_argument(
        "--basename",
        type=str,
        help="Base name for all output files",
    )
    parser.add_argument(
        "--clobber",
        help="Overwrite previous output files",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--no-autocorr",
        help="Turn the autocorrelation check function off",
        default=False,
        action="store_true",
        dest="noautocorr",
    )

    args = parser.parse_args(argv)
    pint.logging.setup(
        level=pint.logging.get_level(args.loglevel, args.verbosity, args.quiet)
    )
    global nwalkers, nsteps, ftr

    eventfile = args.eventfile
    parfile = args.parfile
    gaussianfile = args.gaussianfile
    weightcol = args.weightcol

    if args.ft2 is not None:
        # Instantiate Fermi observatory once so it gets added to the observatory registry
        get_satellite_observatory("Fermi", args.ft2)

    nwalkers = args.nwalkers
    burnin = args.burnin
    nsteps = args.nsteps
    if burnin >= nsteps:
        log.error("burnin must be < nsteps")
        sys.exit(1)
    nbins = 256  # For likelihood calculation based on gaussians file
    outprof_nbins = 256  # in the text file, for pygaussfit.py, for instance
    minMJD = args.minMJD
    maxMJD = args.maxMJD  # Usually set by coverage of IERS file

    minWeight = args.minWeight
    do_opt_first = args.doOpt
    wgtexp = args.wgtexp
    ncores = args.ncores

    # Read in initial model
    modelin = pint.models.get_model(parfile)

    # File name setup and clobber file check
    filepath = args.filepath or os.getcwd()
    basename = args.basename or modelin.PSR.value
    filename = os.path.join(filepath, basename)

    check_file = os.path.isfile(
        filename + "_pre.png"
    )  # Checks to see if the first generated phaseogram file exists
    if check_file:
        if args.clobber:
            log.warning("Clobber flag is on: Preexisting files will be overwritten")
        else:
            log.warning(
                "Clobber flag is not on: Preexisting files will not be overwritten. Change the basename or filepath to avoid overwritting previous results"
            )
            raise Exception

    # The custom_timing version below is to manually construct the TimingModel
    # class, which allows it to be pickled. This is needed for parallelizing
    # the emcee call over a number of threads.  So far, it isn't quite working
    # so it is disabled.  The code above constructs the TimingModel class
    # dynamically, as usual.
    # modelin = custom_timing(parfile)

    # Remove the dispersion delay as it is unnecessary
    # modelin.delay_funcs['L1'].remove(modelin.dispersion_delay)
    # Set the target coords for automatic weighting if necessary
    if "ELONG" in modelin.params:
        tc = SkyCoord(
            modelin.ELONG.quantity,
            modelin.ELAT.quantity,
            frame="barycentrictrueecliptic",
        )
    else:
        tc = SkyCoord(modelin.RAJ.quantity, modelin.DECJ.quantity, frame="icrs")

    target = tc if weightcol == "CALC" else None

    # TODO: make this properly handle long double
    ts = None
    if args.usepickle:
        try:
            ts = toa.load_pickle(eventfile)
        except IOError:
            pass
    if ts is None:
        ts = fermi.get_Fermi_TOAs(
            eventfile,
            weightcolumn=weightcol,
            targetcoord=target,
            minweight=minWeight,
            minmjd=minMJD,
            maxmjd=maxMJD,
            ephem="DE421",
            planets=False,
        )
        log.info("There are %d events we will use" % len(ts))
        ts.filename = eventfile
        # FIXME: writes to the TOA directory unconditionally
        try:
            toa.save_pickle(ts)
        except IOError:
            pass

    if weightcol is not None:
        if weightcol == "CALC":
            weights = np.asarray([float(x["weight"]) for x in ts.table["flags"]])
            log.info(
                "Original weights have min / max weights %.3f / %.3f"
                % (weights.min(), weights.max())
            )
            # Rescale the weights, if requested (by having wgtexp != 0.0)
            if wgtexp != 0.0:
                weights **= wgtexp
                wmx, wmn = weights.max(), weights.min()
                # make the highest weight = 1, but keep min weight the same
                weights = wmn + ((weights - wmn) * (1.0 - wmn) / (wmx - wmn))
            for ii, x in enumerate(ts.table["flags"]):
                x["weight"] = str(weights[ii])
        weights = np.asarray([float(x["weight"]) for x in ts.table["flags"]])
        log.info(
            "There are %d events, with min / max weights %.3f / %.3f"
            % (len(weights), weights.min(), weights.max())
        )
    else:
        weights = None
        log.info("There are %d events, no weights are being used." % ts.ntoas)

    # Now load in the gaussian template and normalize it
    gtemplate = read_gaussfitfile(gaussianfile, nbins)
    gtemplate /= gtemplate.mean()

    # Set the priors on the parameters in the model, before
    # instantiating the emcee_fitter
    # Currently, this adds a gaussian prior on each parameter
    # with width equal to the par file uncertainty * priorerrfact,
    # and then puts in some special cases.
    # *** This should be replaced/supplemented with a way to specify
    # more general priors on parameters that need certain bounds
    phs = 0.0 if args.phs is None else args.phs
    fitkeys, fitvals, fiterrs = get_fit_keyvals(modelin, phs=phs, phserr=args.phserr)

    for key, v, e in zip(fitkeys[:-1], fitvals[:-1], fiterrs[:-1]):
        if key == "SINI" or key == "E" or key == "ECC":
            getattr(modelin, key).prior = Prior(uniform(0.0, 1.0))
        elif key == "PX":
            getattr(modelin, key).prior = Prior(uniform(0.0, 10.0))
        elif key.startswith("GLPH"):
            getattr(modelin, key).prior = Prior(uniform(-0.5, 1.0))
        else:
            getattr(modelin, key).prior = Prior(
                norm(loc=float(v), scale=float(e * args.priorerrfact))
            )

    # Now define the requirements for emcee
    ftr = emcee_fitter(ts, modelin, gtemplate, weights, phs, args.phserr)

    # Use this if you want to see the effect of setting minWeight
    if args.testWeights:
        log.info("Checking H-test vs weights")
        ftr.prof_vs_weights(use_weights=True)
        ftr.prof_vs_weights(use_weights=False)
        sys.exit()

    # Now compute the photon phases and see if we see a pulse
    phss = ftr.get_event_phases()
    maxbin, like_start = marginalize_over_phase(
        phss, gtemplate, weights=ftr.weights, minimize=True, showplot=False
    )
    log.info("Starting pulse likelihood: %f" % like_start)
    if args.phs is None:
        fitvals[-1] = 1.0 - maxbin[0] / float(len(gtemplate))
        if fitvals[-1] > 1.0:
            fitvals[-1] -= 1.0
        if fitvals[-1] < 0.0:
            fitvals[-1] += 1.0
        log.info("Starting pulse phase: %f" % fitvals[-1])
    else:
        log.warning(
            "Measured starting pulse phase is %f, but using %f"
            % (1.0 - maxbin / float(len(gtemplate)), args.phs)
        )
        fitvals[-1] = args.phs
    ftr.fitvals[-1] = fitvals[-1]
    ftr.phaseogram(plotfile=filename + "_pre.png")
    plt.close()
    # ftr.phaseogram()

    # Write out the starting pulse profile
    vs, xs = np.histogram(
        ftr.get_event_phases(), outprof_nbins, range=[0, 1], weights=ftr.weights
    )
    f = open(filename + "_prof_pre.txt", "w")
    for x, v in zip(xs, vs):
        f.write("%.5f  %12.5f\n" % (x, v))
    f.close()

    # Try normal optimization first to see how it goes
    if do_opt_first:
        result = op.minimize(ftr.minimize_func, np.zeros_like(ftr.fitvals))
        newfitvals = np.asarray(result["x"]) * ftr.fiterrs + ftr.fitvals
        like_optmin = -result["fun"]
        log.info("Optimization likelihood: %f" % like_optmin)
        ftr.set_params(dict(zip(ftr.fitkeys, newfitvals)))
        ftr.phaseogram()
    else:
        like_optmin = -np.inf

    # Set up the initial conditions for the emcee walkers.  Use the
    # scipy.optimize newfitvals instead if they are better
    ndim = ftr.n_fit_params
    if like_start > like_optmin:
        # Keep the starting deviations small...
        pos = [
            ftr.fitvals + ftr.fiterrs * args.initerrfact * np.random.randn(ndim)
            for ii in range(nwalkers)
        ]
        # Set starting params
        for param in ["GLPH_1", "GLEP_1", "SINI", "M2", "E", "ECC", "PX", "A1"]:
            if param in ftr.fitkeys:
                idx = ftr.fitkeys.index(param)
                if param == "GLPH_1":
                    svals = np.random.uniform(-0.5, 0.5, nwalkers)
                elif param == "GLEP_1":
                    svals = np.random.uniform(minMJD + 100, maxMJD - 100, nwalkers)
                    # svals = 55422.0 + np.random.randn(nwalkers)
                elif param == "SINI":
                    svals = np.random.uniform(0.0, 1.0, nwalkers)
                elif param == "M2":
                    svals = np.random.uniform(0.1, 0.6, nwalkers)
                elif param in ["E", "ECC", "PX", "A1"]:
                    # Ensure all positive
                    svals = np.fabs(
                        ftr.fitvals[idx] + ftr.fiterrs[idx] * np.random.randn(nwalkers)
                    )
                    if param in ["E", "ECC"]:
                        svals[svals > 1.0] = 1.0 - (svals[svals > 1.0] - 1.0)
                for ii in range(nwalkers):
                    pos[ii][idx] = svals[ii]
    else:
        pos = [
            newfitvals + ftr.fiterrs * args.initerrfact * np.random.randn(ndim)
            for i in range(nwalkers)
        ]
    # Set the 0th walker to have the initial pre-fit solution
    # This way, one walker should always be in a good position
    pos[0] = ftr.fitvals

    import emcee

    # Setting up a backend to save the chains into an h5 file
    if args.backend:
        try:
            backend = emcee.backends.HDFBackend(filename + "_chains.h5")
            backend.reset(nwalkers, ndim)
        except ImportError:
            log.warning("h5py package not installed. Backend set to None")
            backend = None
    else:
        backend = None

    dtype = [("lnprior", float), ("lnlikelihood", float)]

    # Following are for parallel processing tests...
    if args.multicore:
        try:
            import pathos.multiprocessing as mp

            def unwrapped_lnpost(theta):
                return ftr.lnposterior(theta)

            with mp.ProcessPool(nodes=ncores) as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers,
                    ndim,
                    unwrapped_lnpost,
                    blobs_dtype=dtype,
                    pool=pool,
                    backend=backend,
                )
                if args.noautocorr:
                    sampler.run_mcmc(pos, nsteps, progress=True)
                else:
                    autocorr = run_sampler_autocorr(sampler, pos, nsteps, burnin)
            pool.close()
            pool.join()
        except ImportError:
            log.info("Pathos module not available, using single core")
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, ftr.lnposterior, blobs_dtype=dtype, backend=backend
            )
            if args.noautocorr:
                sampler.run_mcmc(pos, nsteps, progress=True)
            else:
                autocorr = run_sampler_autocorr(sampler, pos, nsteps, burnin)
    else:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, ftr.lnposterior, blobs_dtype=dtype, backend=backend
        )
        if args.noautocorr:
            sampler.run_mcmc(pos, nsteps, progress=True)
        else:
            autocorr = run_sampler_autocorr(sampler, pos, nsteps, burnin)

    def chains_to_dict(names, sampler):
        samples = np.transpose(sampler.get_chain(), (1, 0, 2))
        chains = [samples[:, :, ii].T for ii in range(len(names))]
        return dict(zip(names, chains))

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
            plt.close()
        else:
            plt.show()
            plt.close()

    chains = chains_to_dict(ftr.fitkeys, sampler)
    plot_chains(chains, file=filename + "_chains.png")

    # Make the triangle plot.
    samples = np.transpose(sampler.get_chain(discard=burnin), (1, 0, 2)).reshape(
        (-1, ndim)
    )

    blobs = sampler.get_blobs()
    lnprior_samps = blobs["lnprior"]
    lnlikelihood_samps = blobs["lnlikelihood"]
    lnpost_samps = lnprior_samps + lnlikelihood_samps
    ind = np.unravel_index(
        np.argmax(lnpost_samps[:][burnin:]), lnpost_samps[:][burnin:].shape
    )
    ftr.maxpost_fitvals = [chains[ii][burnin:][ind] for ii in ftr.fitkeys]

    try:
        import corner

        fig = corner.corner(
            samples,
            labels=ftr.fitkeys,
            bins=50,
            truths=ftr.maxpost_fitvals,
            plot_contours=True,
        )
        fig.savefig(filename + "_triangle.png")
        plt.close()
    except ImportError:
        pass

    # Plot the scaled prior probability alongside the initial gaussian probability distribution and the histogrammed samples
    ftr.plot_priors(chains, burnin, scale=True)
    plt.savefig(filename + "_priors.png")
    plt.close()

    # Make a phaseogram with the 50th percentile values
    # ftr.set_params(dict(zip(ftr.fitkeys, np.percentile(samples, 50, axis=0))))
    # Make a phaseogram with the best MCMC result
    ftr.set_params(dict(zip(ftr.fitkeys[:-1], ftr.maxpost_fitvals[:-1])))
    ftr.phaseogram(plotfile=filename + "_post.png")
    plt.close()

    # Write out the output pulse profile
    vs, xs = np.histogram(
        ftr.get_event_phases(), outprof_nbins, range=[0, 1], weights=ftr.weights
    )
    f = open(filename + "_prof_post.txt", "w")
    for x, v in zip(xs, vs):
        f.write("%.5f  %12.5f\n" % (x, v))
    f.close()

    # Write out the par file for the best MCMC parameter est
    centered_samples = [
        samples[:, i] - ftr.maxpost_fitvals[i] for i in range(len(ftr.fitkeys))
    ]
    errors = [
        np.percentile(np.abs(centered_samples[i]), 68) for i in range(len(ftr.fitkeys))
    ]
    ftr.set_param_uncertainties(dict(zip(ftr.fitkeys[:-1], errors[:-1])))

    f = open(filename + "_post.par", "w")
    f.write(ftr.model.as_parfile())
    f.close()

    # Print the best MCMC values and ranges
    ranges = map(
        lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
        zip(*np.percentile(samples, [16, 50, 84], axis=0)),
    )
    log.info("Post-MCMC values (50th percentile +/- (16th/84th percentile):")
    for name, vals in zip(ftr.fitkeys, ranges):
        log.info("%8s:" % name + "%25.15g (+ %12.5g  / - %12.5g)" % vals)

    # Put the same stuff in a file
    f = open(filename + "_results.txt", "w")

    f.write("Post-MCMC values (50th percentile +/- (16th/84th percentile):\n")
    for name, vals in zip(ftr.fitkeys, ranges):
        f.write("%8s:" % name + " %25.15g (+ %12.5g  / - %12.5g)\n" % vals)

    f.write("\nMaximum likelihood par file:\n")
    f.write(ftr.model.as_parfile())
    f.close()

    import pickle

    pickle.dump(samples, open(filename + "_samples.pickle", "wb"))
