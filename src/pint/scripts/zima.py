#!/usr/bin/env python -W ignore::FutureWarning -W ignore::UserWarning -W ignore::DeprecationWarning
"""PINT-based tool for making simulated TOAs

"""
from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np
from astropy import log
from astropy.time import TimeDelta

import pint.fitter
import pint.models
import pint.toa as toa
from pint.observatory import get_observatory

__all__ = ["main"]

# log.setLevel("INFO")


def get_freq_array(base_freq_values, ntoas):
    """Right now it is a very simple frequency array simulation.
       It just simulates an alternating frequency arrays
    """
    freq = np.zeros(ntoas)
    num_freqs = len(base_freq_values)
    for ii, fv in enumerate(base_freq_values):
        freq[ii::num_freqs] = fv
    return freq


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="PINT tool for simulating TOAs")
    parser.add_argument("parfile", help="par file to read model from")
    parser.add_argument("timfile", help="Output TOA file name")
    parser.add_argument(
        "--inputtim",
        help="Input tim file for fake TOA sampling",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--startMJD",
        help="MJD of first fake TOA (default=56000.0)",
        type=float,
        default=56000.0,
    )
    parser.add_argument(
        "--ntoa", help="Number of fake TOAs to generate", type=int, default=100
    )
    parser.add_argument(
        "--duration", help="Span of TOAs to generate (days)", type=float, default=400.0
    )
    parser.add_argument("--obs", help="Observatory code (default: GBT)", default="GBT")
    parser.add_argument(
        "--freq",
        help="Frequency for TOAs (MHz) (default: 1400)",
        nargs="+",
        type=float,
        default=1400.0,
    )
    parser.add_argument(
        "--error",
        help="Random error to apply to each TOA (us, default=1.0)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--fuzzdays",
        help="Standard deviation of 'fuzz' distribution (jd) (default: 0.0)",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--plot", help="Plot residuals", action="store_true", default=False
    )
    parser.add_argument("--ephem", help="Ephemeris to use", default="DE421")
    parser.add_argument(
        "--planets",
        help="Use planetary Shapiro delay",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--format", help="The format of out put .tim file.", default="TEMPO2"
    )
    args = parser.parse_args(argv)

    log.info("Reading model from {0}".format(args.parfile))
    m = pint.models.get_model(args.parfile)

    out_format = args.format
    error = args.error * u.microsecond

    if args.inputtim is None:
        log.info("Generating uniformly spaced TOAs")
        duration = args.duration * u.day
        # start = Time(args.startMJD,scale='utc',format='pulsar_mjd',precision=9)
        start = np.longdouble(args.startMJD) * u.day
        freq = np.atleast_1d(args.freq) * u.MHz
        site = get_observatory(args.obs)
        scale = site.timescale

        times = np.linspace(0, duration.to(u.day).value, args.ntoa) * u.day + start

        # 'Fuzz' out times
        if args.fuzzdays > 0.0:
            fuzz = np.random.normal(scale=args.fuzzdays, size=len(times)) * u.day
            times += fuzz

        # Add mulitple frequency
        freq_array = get_freq_array(freq, len(times))
        tl = [
            toa.TOA(t.value, error=error, obs=args.obs, freq=f, scale=scale)
            for t, f in zip(times, freq_array)
        ]
        ts = toa.TOAs(toalist=tl)
    else:
        log.info("Reading initial TOAs from {0}".format(args.inputtim))
        ts = toa.TOAs(toafile=args.inputtim)
        ts.table["error"][:] = error

    # WARNING! I'm not sure how clock corrections should be handled here!
    # Do we apply them, or not?
    if not any(["clkcorr" in f for f in ts.table["flags"]]):
        log.info("Applying clock corrections.")
        ts.apply_clock_corrections()
    if "tdb" not in ts.table.colnames:
        log.info("Getting IERS params and computing TDBs.")
        ts.compute_TDBs(ephem=args.ephem)
    if "ssb_obs_pos" not in ts.table.colnames:
        log.info("Computing observatory positions and velocities.")
        ts.compute_posvels(args.ephem, args.planets)

    log.info("Creating TOAs")
    F_local = m.d_phase_d_toa(ts)
    rs = m.phase(ts).frac / F_local

    # Adjust the TOA times to put them where their residuals will be 0.0
    ts.adjust_TOAs(TimeDelta(-1.0 * rs))
    rspost = m.phase(ts).frac / F_local

    log.info("Second iteration")
    # Do a second iteration
    ts.adjust_TOAs(TimeDelta(-1.0 * rspost))

    err = np.random.randn(len(ts.table)) * error
    # Add the actual error fuzzing
    ts.adjust_TOAs(TimeDelta(err))

    # Write TOAs to a file
    ts.write_TOA_file(args.timfile, name="fake", format=out_format)

    if args.plot:
        # This should be a very boring plot with all residuals flat at 0.0!
        import matplotlib.pyplot as plt
        from astropy.visualization import quantity_support

        quantity_support()

        rspost2 = m.phase(ts).frac / F_local
        plt.errorbar(
            ts.get_mjds(), rspost2.to(u.us), yerr=ts.get_errors().to(u.us), fmt="."
        )
        newts = pint.toa.get_TOAs(args.timfile, ephem=args.ephem, planets=args.planets)
        rsnew = m.phase(newts).frac / F_local
        plt.errorbar(
            newts.get_mjds(), rsnew.to(u.us), yerr=newts.get_errors().to(u.us), fmt="."
        )
        # plt.plot(ts.get_mjds(),rspost.to(u.us),'x')
        plt.xlabel("MJD")
        plt.ylabel("Residual (us)")
        plt.grid(True)
        plt.show()
