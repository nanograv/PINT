#!/usr/bin/env python
import sys

import astropy.io.fits as pyfits
import numpy as np

import pint.logging
from loguru import logger as log

pint.logging.setup(level=pint.logging.script_level)

import pint.models
import pint.residuals
import pint.toa as toa
from pint.event_toas import get_event_TOAs
from pint.eventstats import h2sig, hm
from pint.fits_utils import read_fits_event_mjds
from pint.observatory.satellite_obs import get_satellite_observatory
from pint.plot_utils import phaseogram_binned
import pint.polycos as polycos

__all__ = ["main"]


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Use PINT to compute event phases and make plots of photon event files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "eventfile",
        help="Photon event FITS file name (e.g. from NICER, RXTE, XMM, Chandra).",
    )
    parser.add_argument("parfile", help="par file to construct model from")
    parser.add_argument("--orbfile", help="Name of orbit file", default=None)
    parser.add_argument(
        "--maxMJD", help="Maximum MJD to include in analysis", default=None
    )
    parser.add_argument(
        "--minMJD", help="Minimum MJD to include in analysis", default=None
    )
    parser.add_argument("--plotfile", help="Output figure file name", default=None)
    parser.add_argument(
        "--addphase",
        help="Write FITS file with added phase column",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--addorbphase",
        help="Write FITS file with added orbital phase column",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--absphase",
        help="Write FITS file with integral portion of pulse phase (ABS_PHASE)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--barytime",
        help="Write FITS file with a column containing the barycentric time as double precision MJD.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--outfile",
        help="Output FITS file name (default=same as eventfile)",
        default=None,
    )
    parser.add_argument("--ephem", help="Planetary ephemeris to use", default="DE421")
    parser.add_argument(
        "--tdbmethod",
        help="Method for computing TT to TDB (default=astropy)",
        default="default",
    )
    parser.add_argument(
        "--plot", help="Show phaseogram plot.", action="store_true", default=False
    )
    parser.add_argument(
        "--use_gps",
        default=False,
        action="store_true",
        help="Apply GPS to UTC clock corrections",
    )
    parser.add_argument(
        "--use_bipm",
        default=False,
        action="store_true",
        help="Use TT(BIPM) instead of TT(TAI)",
    )
    parser.add_argument(
        "--polycos",
        default=False,
        action="store_true",
        help="Use polycos to calculate phases; use when working with very large event files",
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

    args = parser.parse_args(argv)
    pint.logging.setup(
        level=pint.logging.get_level(args.loglevel, args.verbosity, args.quiet)
    )
    #    parser.add_argument("--fix",help="Apply 1.0 second offset for NICER", action='store_true', default=False)

    # If outfile is specified, that implies addphase
    if args.outfile is not None:
        args.addphase = True

    # If plotfile is specified, that implies plot
    if args.plotfile is not None:
        args.plot = True

    # set MJD ranges
    maxmjd = np.inf if (args.maxMJD is None) else float(args.maxMJD)
    minmjd = 0.0 if (args.minMJD is None) else float(args.minMJD)

    # Read event file header to figure out what instrument is is from
    hdr = pyfits.getheader(args.eventfile, ext=1)

    log.info(
        "Event file TELESCOPE = {0}, INSTRUMENT = {1}".format(
            hdr["TELESCOP"], hdr["INSTRUME"]
        )
    )

    telescope = hdr["TELESCOP"].lower()

    # Instantiate observatory once so it gets added to the observatory registry
    if args.orbfile is not None:
        log.info(f"Setting up {telescope.upper()} observatory")
        try:
            get_satellite_observatory(telescope, args.orbfile)
        except Exception:
            log.error(
                "The orbit file is not recognized. It is likely that this mission is not supported. "
                "Please barycenter the event file using the official mission tools before processing with PINT"
            )
    # Read in model
    modelin = pint.models.get_model(args.parfile)
    use_planets = False
    if "PLANET_SHAPIRO" in modelin.params:
        if modelin.PLANET_SHAPIRO.value:
            use_planets = True
    if "AbsPhase" not in modelin.components:
        log.error(
            "TimingModel does not include AbsPhase component, which is required "
            "for computing phases. Make sure you have TZR* parameters in your par file!"
        )
        raise ValueError("Model missing AbsPhase component.")

    # Read event file and return list of TOA objects, if not using polycos
    if args.polycos == False:
        try:
            # tl = load_event_TOAs(
            #     args.eventfile, telescope, minmjd=minmjd, maxmjd=maxmjd
            # )
            ts = get_event_TOAs(
                args.eventfile,
                telescope,
                minmjd=minmjd,
                maxmjd=maxmjd,
                ephem=args.ephem,
                include_bipm=args.use_bipm,
                include_gps=args.use_gps,
                planets=use_planets,
            )
        except KeyError:
            log.error(
                "Observatory not recognized. This probably means you need to provide an orbit file or barycenter the event file."
            )
            sys.exit(1)

        # Now convert to TOAs object and compute TDBs and posvels
        if len(ts) == 0:
            log.error("No TOAs, exiting!")
            sys.exit(0)

    if args.addorbphase and (not hasattr(modelin, "binary_model_name")):
        log.error(
            "TimingModel does not include a binary model, which is required for "
            "computing orbital phases. Make sure you have BINARY and associated "
            "model parameters in your par file!"
        )
        raise ValueError("Model missing BINARY component.")

    # Use polycos to calculate pulse phases
    if args.polycos:
        log.info("Using polycos to get pulse phases.")

        if args.addorbphase:
            raise ValueError("Cannot use orbphase with polycos.")

        # Polycos parameters
        segLength = 120  # in minutes
        ncoeff = 10
        obsfreq = 0

        # Open event file and get start and end mjds
        hdulist = pyfits.open(args.eventfile)
        data = hdulist[1].data
        mjds = read_fits_event_mjds(hdulist[1])
        minmjd = min(mjds)
        maxmjd = max(mjds)

        # Check if event file is barycentered
        if hdulist[1].header["TIMESYS"] != "TDB":
            raise ValueError(
                "The event file has not been barycentered. Polycos can only be used with barycentered data."
            )

        # Create polycos table
        log.debug("Generating polycos")
        telescope_n = "@"
        p = polycos.Polycos()
        ptable = p.generate_polycos(
            modelin, minmjd, maxmjd, telescope_n, segLength, ncoeff, obsfreq
        )

        # Calculate phases
        log.debug("Evaluating polycos")
        phases = ptable.eval_phase(mjds)
        phases[phases < 0] += 1.0
        h = float(hm(phases))
        print("Htest : {0:.2f} ({1:.2f} sigma)".format(h, h2sig(h)))
    else:  # Normal mode, not polycos
        ts.filename = args.eventfile
        #    if args.fix:
        #        ts.adjust_TOAs(TimeDelta(np.ones(len(ts.table))*-1.0*u.s,scale='tt'))

        print(ts.get_summary())
        mjds = ts.get_mjds()
        print(mjds.min(), mjds.max())

        # Compute model phase for each TOA
        iphss, phss = modelin.phase(ts, abs_phase=True)
        phases = phss.value % 1
        h = float(hm(phases))
        print("Htest : {0:.2f} ({1:.2f} sigma)".format(h, h2sig(h)))

    if args.plot:
        phaseogram_binned(mjds, phases, bins=100, plotfile=args.plotfile)

    # Compute orbital phases for each photon TOA
    if args.addorbphase:
        delay = modelin.delay(ts)
        orbits = modelin.binary_instance.orbits()
        # These lines are already in orbits.orbit_phase() in binary_orbits.py.
        # What is the correct syntax is to call this function here?
        norbits = np.array(np.floor(orbits), dtype=int)
        orbphases = orbits - norbits  # fractional phase

    if args.addphase or args.addorbphase:
        # Read input FITS file (again).
        # If overwriting, open in 'update' mode
        if args.outfile is None:
            hdulist = pyfits.open(args.eventfile, mode="update")
        else:
            hdulist = pyfits.open(args.eventfile)

        datacol = []
        data_to_add = {}

        # Handle case where minMJD/maxMJD do not exceed length of events
        mjds_float = read_fits_event_mjds(hdulist[1])
        time_mask = np.logical_and((mjds_float > minmjd), (mjds_float < maxmjd))
        if args.polycos:
            time_mask = np.logical_and((mjds_float >= minmjd), (mjds_float <= maxmjd))

        if args.addphase:
            if time_mask.sum() != len(phases):
                raise RuntimeError(
                    "Mismatch between data selection {0} and length of phase array ({1})!".format(
                        time_mask.sum(), len(phases)
                    )
                )
            data_to_add["PULSE_PHASE"] = [phases, "D"]

        if args.absphase:
            data_to_add["ABS_PHASE"] = [iphss, "K"]

        if args.barytime:
            bats = modelin.get_barycentric_toas(ts)
            data_to_add["BARY_TIME"] = [bats, "D"]

        if args.addorbphase:
            if time_mask.sum() != len(orbphases):
                raise RuntimeError(
                    "Mismatch between data selection ({0}) and length of orbital phase array ({1})!".format(
                        time_mask.sum(), len(orbphases)
                    )
                )
            data_to_add["ORBIT_PHASE"] = [orbphases, "D"]
        # End if args.addorbphase

        for key in data_to_add.keys():
            if key in hdulist[1].columns.names:
                log.info("Found existing %s column, overwriting..." % key)
                # Overwrite values in existing Column
                hdulist[1].data[key][time_mask] = data_to_add[key][0]
            else:
                # Construct and append new column, preserving HDU header and name
                log.info("Adding new %s column." % key)
                new_dat = np.full(time_mask.shape, -1, dtype=data_to_add[key][0].dtype)
                new_dat[time_mask] = data_to_add[key][0]
                datacol.append(
                    pyfits.ColDefs(
                        [
                            pyfits.Column(
                                name=key, format=data_to_add[key][1], array=new_dat
                            )
                        ]
                    )
                )

        if len(datacol) > 0:
            cols = hdulist[1].columns
            for c in datacol:
                cols = cols + c
            bt = pyfits.BinTableHDU.from_columns(
                cols, header=hdulist[1].header, name=hdulist[1].name
            )
            hdulist[1] = bt

        if args.outfile is None:
            # Overwrite the existing file
            log.info("Overwriting existing FITS file " + args.eventfile)
            hdulist.flush(verbose=True, output_verify="warn")
        else:
            # Write to new output file
            log.info("Writing output FITS file " + args.outfile)
            hdulist.writeto(
                args.outfile, overwrite=True, checksum=True, output_verify="warn"
            )
