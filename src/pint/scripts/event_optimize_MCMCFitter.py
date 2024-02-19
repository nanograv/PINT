#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from astropy.coordinates import SkyCoord
import pint.logging
from loguru import logger as log

log.remove()
log.add(
    sys.stderr,
    level="WARNING",
    colorize=True,
    format=pint.logging.format,
    filter=pint.logging.LogFilter(),
)
import pint.fermi_toas as fermi
import pint.models
import pint.toa as toa
from pint.mcmc_fitter import MCMCFitterBinnedTemplate
from pint.observatory.satellite_obs import get_satellite_observatory
from pint.sampler import EmceeSampler
from pint.scripts.event_optimize import marginalize_over_phase, read_gaussfitfile


__all__ = ["main"]
# np.seterr(all='raise')

# initialization values
# Should probably figure a way to make these not global variables
maxpost = -9e99
numcalls = 0


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
        help="Number of MCMC steps for burn in ",
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
        choices=("TRACE", "DEBUG", "INFO", "WARNING", "ERROR"),
        default=pint.logging.script_level,
        help="Logging level",
        dest="loglevel",
    )
    global nwalkers, nsteps, ftr

    args = parser.parse_args(argv)
    log.remove()
    log.add(
        sys.stderr,
        level=args.loglevel,
        colorize=True,
        format=pint.logging.format,
        filter=pint.logging.LogFilter(),
    )
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

    # Read in initial model
    modelin = pint.models.get_model(parfile)

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
    if not args.usepickle or (
        not (
            os.path.isfile(eventfile + ".pickle")
            or os.path.isfile(eventfile + ".pickle.gz")
        )
    ):
        # Read event file and return list of TOA objects
        tl = fermi.load_Fermi_TOAs(
            eventfile, weightcolumn=weightcol, targetcoord=target, minweight=minWeight
        )
        # Limit the TOAs to ones in selected MJD range and above minWeight
        tl = [
            tl[ii]
            for ii in range(len(tl))
            if (
                tl[ii].mjd.value > minMJD
                and tl[ii].mjd.value < maxMJD
                and (weightcol is None or tl[ii].flags["weight"] > minWeight)
            )
        ]
        log.info("There are %d events we will use" % len(tl))
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
        if weightcol == "CALC":
            weights = np.asarray([x["weight"] for x in ts.table["flags"]])
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
                x["weight"] = weights[ii]
        weights = np.asarray([x["weight"] for x in ts.table["flags"]])
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

    sampler = EmceeSampler(nwalkers)
    ftr = MCMCFitterBinnedTemplate(
        ts,
        modelin,
        sampler,
        template=gtemplate,
        weights=weights,
        phs=phs,
        phserr=args.phserr,
        minMJD=minMJD,
        maxMJD=maxMJD,
    )

    fitkeys, fitvals, fiterrs = ftr.get_fit_keyvals()

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
        fitvals[-1] = (1.0 - maxbin[0] / float(len(gtemplate))) % 1
        log.info("Starting pulse phase: %f" % fitvals[-1])
    else:
        log.info(
            "Measured starting pulse phase is %f, but using %f"
            % (1.0 - maxbin / float(len(gtemplate)), args.phs)
        )
        fitvals[-1] = args.phs
    ftr.fitvals[-1] = fitvals[-1]
    ftr.phaseogram(plotfile=ftr.model.PSR.value + "_pre.png")
    plt.close()

    # Write out the starting pulse profile
    vs, xs = np.histogram(
        ftr.get_event_phases(), outprof_nbins, range=[0, 1], weights=ftr.weights
    )
    f = open(ftr.model.PSR.value + "_prof_pre.txt", "w")
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
        pos = None
    else:
        pos = [
            newfitvals + ftr.fiterrs * args.initerrfact * np.random.randn(ndim)
            for i in range(nwalkers)
        ]
        pos[0] = ftr.fitvals

    ftr.fit_toas(
        maxiter=nsteps,
        pos=pos,
        priorerrfact=args.priorerrfact,
        errfact=args.initerrfact,
    )

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

    chains = sampler.chains_to_dict(ftr.fitkeys)
    plot_chains(chains, file=ftr.model.PSR.value + "_chains.png")

    # Make the triangle plot.
    # samples = sampler.sampler.chain[:, burnin:, :].reshape((-1, ftr.n_fit_params))
    samples = np.transpose(
        sampler.sampler.get_chain(discard=burnin), (1, 0, 2)
    ).reshape((-1, ftr.n_fit_params))
    try:
        import corner

        fig = corner.corner(
            samples,
            labels=ftr.fitkeys,
            bins=50,
            truths=ftr.maxpost_fitvals,
            plot_contours=True,
        )
        fig.savefig(ftr.model.PSR.value + "_triangle.png")
        plt.close()
    except ImportError:
        pass

    # Plot the scaled prior probability alongside the initial gaussian probability distribution and the histogrammed samples
    ftr.plot_priors(chains, burnin, scale=True)
    plt.savefig(ftr.model.PSR.value + "_priors.png")
    plt.close()

    # Make a phaseogram with the 50th percentile values
    # ftr.set_params(dict(zip(ftr.fitkeys, np.percentile(samples, 50, axis=0))))
    # Make a phaseogram with the best MCMC result
    ftr.set_params(dict(zip(ftr.fitkeys[:-1], ftr.maxpost_fitvals[:-1])))
    ftr.phaseogram(plotfile=ftr.model.PSR.value + "_post.png")
    plt.close()

    # Write out the output pulse profile
    vs, xs = np.histogram(
        ftr.get_event_phases(), outprof_nbins, range=[0, 1], weights=ftr.weights
    )
    f = open(ftr.model.PSR.value + "_prof_post.txt", "w")
    for x, v in zip(xs, vs):
        f.write("%.5f  %12.5f\n" % (x, v))
    f.close()

    # Write out the par file for the best MCMC parameter est
    f = open(ftr.model.PSR.value + "_post.par", "w")
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
    f = open(ftr.model.PSR.value + "_results.txt", "w")

    f.write("Post-MCMC values (50th percentile +/- (16th/84th percentile):\n")
    for name, vals in zip(ftr.fitkeys, ranges):
        f.write("%8s:" % name + " %25.15g (+ %12.5g  / - %12.5g)\n" % vals)

    f.write("\nMaximum likelihood par file:\n")
    f.write(ftr.model.as_parfile())
    f.close()

    import pickle

    pickle.dump(samples, open(ftr.model.PSR.value + "_samples.pickle", "wb"))
