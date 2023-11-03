#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
import argparse
import contextlib
import sys
import pickle

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
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
from pint.templates import lctemplate, lcfitters
from pint.residuals import Residuals
from pint.mcmc_fitter import CompositeMCMCFitter
from pint.observatory.satellite_obs import get_satellite_observatory
from pint.sampler import EmceeSampler
from pint.scripts.event_optimize import read_gaussfitfile


__all__ = ["main"]
# np.seterr(all='raise')

# initialization values
# Should probably figure a way to make these not global variables
maxpost = -9e99
numcalls = 0


def get_toas(evtfile, flags, tcoords=None, minweight=0, minMJD=0, maxMJD=100000):
    if evtfile[:-3] == "tim":
        usepickle = flags["usepickle"] if "usepickle" in flags else False
        ts = toa.get_TOAs(evtfile, usepickle=usepickle)
        # Prune out of range MJDs
        mask = np.logical_or(
            ts.get_mjds() < minMJD * u.day, ts.get_mjds() > maxMJD * u.day
        )
        ts.table.remove_rows(mask)
        ts.table = ts.table.group_by("obs")
    else:
        if "usepickle" in flags and flags["usepickle"]:
            with contextlib.suppress(Exception):
                picklefile = toa._check_pickle(evtfile) or evtfile
                return toa.TOAs(picklefile)
        weightcol = flags["weightcol"] if "weightcol" in flags else None
        target = tcoords if weightcol == "CALC" else None
        tl = fermi.load_Fermi_TOAs(
            evtfile, weightcolumn=weightcol, targetcoord=target, minweight=minweight
        )
        tl = filter(lambda t: (t.mjd.value > minMJD) and (t.mjd.value < maxMJD), tl)
        ts = toa.TOAs(toalist=tl)
        ts.filename = evtfile
        ts.compute_TDBs()
        ts.compute_posvels(ephem="DE421", planets=False)
        ts.pickle()
    log.info("There are %d events we will use" % len(ts.table))
    return ts


def load_eventfiles(infile, tcoords=None, minweight=0, minMJD=0, maxMJD=100000):
    """Load events from multiple sources:

    The format of each line of infile is:

        <eventfile> <log_likelihood function> <template> [flags]

    Allowed flags are:
        setweights
            A multiplicative weight to apply to the probability function for this
            eventfile
        usepickle
            Load from a pickle file
        weightcol
            The weight column in the fits file

    """
    lines = open(infile, "r").read().split("\n")
    eventinfo = {
        "toas": [],
        "lnlikes": [],
        "templates": [],
        "weightcol": [],
        "setweights": [],
    }
    for line in lines:
        log.info(f"{line}")
        if len(line) == 0:
            continue
        try:
            words = line.split()

            flags = {}
            if len(words) > 3:
                kvs = words[3:]
                for i in range(0, len(flags), 2):
                    k, v = kvs[i].lstrip("-"), kvs[i + 1]
                    flags[k] = v
            ts = get_toas(
                words[0],
                flags,
                tcoords=tcoords,
                minweight=minweight,
                minMJD=minMJD,
                maxMJD=maxMJD,
            )
            eventinfo["toas"].append(ts)
            log.info("%s has %d events" % (words[0], len(ts.table)))
            eventinfo["lnlikes"].append(words[1])
            eventinfo["templates"].append(words[2])
            if "setweights" in flags:
                eventinfo["setweights"].append(float(flags["setweights"]))
            else:
                eventinfo["setweights"].append(1.0)
            if "weightcol" in flags:
                eventinfo["weightcol"].append(flags["weightcol"])
            else:
                eventinfo["weightcol"].append(None)
        except Exception as e:
            log.error(f"{str(e)}")
            log.error(f"Could not load {line}")

    return eventinfo


def lnlikelihood_prob(ftr, theta, index):
    phases = ftr.get_event_phases(index)
    phss = (phases.astype(np.float64) + theta[-1]) % 1

    probs = ftr.get_template_vals(phss, index)
    if ftr.weights[index] is None:
        return np.log(probs).sum()
    else:
        return np.log(ftr.weights[index] * probs + 1.0 - ftr.weights[index]).sum()


def lnlikelihood_resid(ftr, theta, index):
    return -Residuals(toas=ftr.toas_list[index], model=ftr.model).chi2.value


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="PINT tool for MCMC optimization of timing models using event data from multiple sources.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("eventfiles", help="Specify a file listing all event files")
    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("--ft2", help="Path to FT2 file.", default=None)
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
        "--samples",
        help="Pickle file containing samples from a previous run",
        default=None,
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
    parfile = args.parfile

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
    wgtexp = args.wgtexp

    # Read in initial model
    modelin = pint.models.get_model(parfile)

    # Set the target coords for automatic weighting if necessary
    if "ELONG" in modelin.params:
        tc = SkyCoord(
            modelin.ELONG.quantity,
            modelin.ELAT.quantity,
            frame="barycentrictrueecliptic",
        )
    else:
        tc = SkyCoord(modelin.RAJ.quantity, modelin.DECJ.quantity, frame="icrs")

    eventinfo = load_eventfiles(
        args.eventfiles, tcoords=tc, minweight=minWeight, minMJD=minMJD, maxMJD=maxMJD
    )

    nsets = len(eventinfo["toas"])
    log.info(
        "Total number of events:\t%d"
        % np.array([len(t.table) for t in eventinfo["toas"]]).sum()
    )
    log.info("Total number of datasets:\t%d" % nsets)

    funcs = {"prob": lnlikelihood_prob, "resid": lnlikelihood_resid}
    lnlike_funcs = [None] * nsets
    wlist = [None] * nsets
    gtemplates = [None] * nsets

    # Loop over all TOA sets
    for i in range(nsets):
        # Determine lnlikelihood function for this set
        try:
            lnlike_funcs[i] = funcs[eventinfo["lnlikes"][i]]
        except:
            raise ValueError(f'{eventinfo["lnlikes"][i]} is not a recognized function')

        # Load in weights
        ts = eventinfo["toas"][i]
        if eventinfo["weightcol"][i] is not None:
            if eventinfo["weightcol"][i] == "CALC":
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
        wlist[i] = weights

        # Load in templates
        tname = eventinfo["templates"][i]
        if tname == "none":
            continue
        if tname[-6:] == "pickle" or tname == "analytic":
            # Analytic template
            try:
                gtemplate = pickle.load(file(tname))
            except Exception:
                phases = (modelin.phase(ts)[1].value).astype(np.float64) % 1
                gtemplate = lctemplate.get_gauss2()
                lcf = lcfitters.LCFitter(gtemplate, phases, weights=wlist[i])
                lcf.fit(unbinned=False)
                pickle.dump(
                    gtemplate,
                    file("%s_template%d.pickle" % (jname, i), "wb"),
                    protocol=2,
                )
            phases = (modelin.phase(ts)[1].value).astype(np.float64) % 1
            lcf = lcfitters.LCFitter(
                gtemplate, phases, weights=wlist[i], binned_bins=200
            )
            lcf.fit_position(unbinned=False)
            lcf.fit(overall_position_first=True, estimate_errors=False, unbinned=False)
            for prim in lcf.template:
                prim.free[:] = False
            lcf.template.norms.free[:] = False
        else:
            # Binned template
            gtemplate = read_gaussfitfile(tname, nbins)
            gtemplate /= gtemplate.mean()

        gtemplates[i] = gtemplate

    # Set the priors on the parameters in the model, before
    # instantiating the emcee_fitter
    # Currently, this adds a gaussian prior on each parameter
    # with width equal to the par file uncertainty * priorerrfact,
    # and then puts in some special cases.
    # *** This should be replaced/supplemented with a way to specify
    # more general priors on parameters that need certain bounds
    phs = 0.0 if args.phs is None else args.phs

    sampler = EmceeSampler(nwalkers)
    ftr = CompositeMCMCFitter(
        eventinfo["toas"],
        modelin,
        sampler,
        lnlike_funcs,
        templates=gtemplates,
        weights=wlist,
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

    ftr.phaseogram(plotfile=f"{ftr.model.PSR.value}_pre.png")
    like_start = ftr.lnlikelihood(ftr, ftr.get_parameters())
    log.info("Starting Pulse Likelihood:\t%f" % like_start)

    # Set up the initial conditions for the emcee walkers
    ndim = ftr.n_fit_params
    if args.samples is None:
        pos = None
    else:
        chains = pickle.load(file(args.samples))
        chains = np.reshape(chains, [nwalkers, -1, ndim])
        pos = chains[:, -1, :]

    ftr.fit_toas(
        nsteps, pos=pos, priorerrfact=args.priorerrfact, errfact=args.initerrfact
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
    plot_chains(chains, file=f"{ftr.model.PSR.value}_chains.png")

    # Make the triangle plot.
    # samples = sampler.sampler.chain[:, burnin:, :].reshape((-1, ftr.n_fit_params))
    samples = np.transpose(
        sampler.sampler.get_chain(discard=burnin), (1, 0, 2)
    ).reshape((-1, ftr.n_fit_params))
    with contextlib.suppress(ImportError):
        import corner

        fig = corner.corner(
            samples,
            labels=ftr.fitkeys,
            bins=50,
            truths=ftr.maxpost_fitvals,
            plot_contours=True,
        )
        fig.savefig(f"{ftr.model.PSR.value}_triangle.png")
        plt.close()
    # Make a phaseogram with the 50th percentile values
    # ftr.set_params(dict(zip(ftr.fitkeys, np.percentile(samples, 50, axis=0))))
    # Make a phaseogram with the best MCMC result
    ftr.set_parameters(ftr.maxpost_fitvals)
    ftr.phaseogram(plotfile=f"{ftr.model.PSR.value}_post.png")
    plt.close()

    with open(f"{ftr.model.PSR.value}_post.par", "w") as f:
        f.write(ftr.model.as_parfile())
    # Print the best MCMC values and ranges
    ranges = map(
        lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
        zip(*np.percentile(samples, [16, 50, 84], axis=0)),
    )
    log.info("Post-MCMC values (50th percentile +/- (16th/84th percentile):")
    for name, vals in zip(ftr.fitkeys, ranges):
        log.info("%8s:" % name + "%25.15g (+ %12.5g  / - %12.5g)" % vals)

    with open(f"{ftr.model.PSR.value}_results.txt", "w") as f:
        f.write("Post-MCMC values (50th percentile +/- (16th/84th percentile):\n")
        for name, vals in zip(ftr.fitkeys, ranges):
            f.write("%8s:" % name + " %25.15g (+ %12.5g  / - %12.5g)\n" % vals)

        f.write("\nMaximum likelihood par file:\n")
        f.write(ftr.model.as_parfile())
    with open(f"{ftr.model.PSR.value}_samples.pickle", "wb") as smppkl:
        pickle.dump(samples, smppkl)
