#!/usr/bin/env python
import argparse

import astropy.io.fits as pyfits
import numpy as np
from astropy.coordinates import SkyCoord

import pint.logging
from loguru import logger as log

pint.logging.setup(level=pint.logging.script_level)

import pint.models
import pint.residuals
import pint.toa as toa
from pint.eventstats import h2sig, hmw
from pint.fermi_toas import get_Fermi_TOAs
from pint.fits_utils import read_fits_event_mjds_tuples
from pint.observatory.satellite_obs import get_satellite_observatory
from pint.plot_utils import phaseogram

__all__ = ["main"]

# log.setLevel('DEBUG')


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Use PINT to compute H-test and plot Phaseogram from a Fermi FT1 event file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("eventfile", help="Fermi event FITS file name.")
    parser.add_argument("parfile", help="par file to construct model from")
    parser.add_argument(
        "weightcol", help="Column name for event weights (or 'CALC' to compute them)"
    )
    parser.add_argument("--ft2", help="Path to FT2 file.", default=None)
    parser.add_argument(
        "--addphase",
        help="Write FT1 file with added phase column",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--plot", help="Show phaseogram plot.", action="store_true", default=False
    )
    parser.add_argument("--plotfile", help="Output figure file name", default=None)
    parser.add_argument(
        "--maxMJD", help="Maximum MJD to include in analysis", default=None
    )
    parser.add_argument(
        "--minMJD", help="Minimum MJD to include in analysis", default=None
    )
    parser.add_argument(
        "--outfile",
        help="Output figure file name (default is to overwrite input file)",
        default=None,
    )
    parser.add_argument(
        "--planets",
        help="Use planetary Shapiro delay in calculations",
        default=False,
        action="store_true",
    )
    parser.add_argument("--ephem", help="Planetary ephemeris to use", default="DE421")
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

    # If outfile is specified, that implies addphase
    if args.outfile is not None:
        args.addphase = True

    # Read in model
    modelin = pint.models.get_model(args.parfile)
    if "ELONG" in modelin.params:
        tc = SkyCoord(
            modelin.ELONG.quantity,
            modelin.ELAT.quantity,
            frame="barycentrictrueecliptic",
        )
    else:
        tc = SkyCoord(modelin.RAJ.quantity, modelin.DECJ.quantity, frame="icrs")

    if args.ft2 is not None:
        # Instantiate Fermi observatory once so it gets added to the observatory registry
        get_satellite_observatory("Fermi", args.ft2)

    # Read event file and return list of TOA objects
    maxmjd = np.inf if (args.maxMJD is None) else float(args.maxMJD)
    minmjd = 0.0 if (args.minMJD is None) else float(args.minMJD)
    # Now convert to TOAs object and compute TDBs and posvels
    # For Fermi, we are not including GPS or TT(BIPM) corrections
    ts = get_Fermi_TOAs(
        args.eventfile,
        maxmjd=maxmjd,
        minmjd=minmjd,
        weightcolumn=args.weightcol,
        targetcoord=tc,
        planets=args.planets,
        ephem=args.ephem,
    )
    ts.filename = args.eventfile

    print(ts.get_summary())
    mjds = ts.get_mjds()
    print(mjds.min(), mjds.max())

    # Compute model phase for each TOA
    iphss, phss = modelin.phase(ts, abs_phase=True)
    phss %= 1
    phases = phss.value
    mjds = ts.get_mjds()
    weights, _ = ts.get_flag_value("weight", as_type=float)
    weights = np.array(weights)
    h = float(hmw(phases, weights))
    print("Htest : {0:.2f} ({1:.2f} sigma)".format(h, h2sig(h)))
    if args.plot:
        log.info("Making phaseogram plot with {0} photons".format(len(mjds)))
        phaseogram(mjds, phases, weights, bins=100, plotfile=args.plotfile)

    if args.addphase:
        # Read input FITS file (again).
        # If overwriting, open in 'update' mode
        if args.outfile is None:
            hdulist = pyfits.open(args.eventfile, mode="update")
        else:
            hdulist = pyfits.open(args.eventfile)
        event_hdu = hdulist[1]
        event_hdr = event_hdu.header
        event_dat = event_hdu.data
        event_mjds = read_fits_event_mjds_tuples(event_hdu)
        mjds_float = np.asarray([r[0] + r[1] for r in event_mjds])
        time_mask = np.logical_and((mjds_float > minmjd), (mjds_float < maxmjd))
        new_phases = np.full(len(event_dat), -1, dtype=float)
        new_phases[time_mask] = phases

        if "PULSE_PHASE" in event_hdu.columns.names:
            log.info("Found existing PULSE_PHASE column, overwriting...")
            # Overwrite values in existing Column
            event_dat["PULSE_PHASE"] = new_phases
        else:
            # Construct and append new column, preserving HDU header and name
            log.info("Adding new PULSE_PHASE column.")
            phasecol = pyfits.ColDefs(
                [pyfits.Column(name="PULSE_PHASE", format="D", array=new_phases)]
            )
            bt = pyfits.BinTableHDU.from_columns(
                event_hdu.columns + phasecol, header=event_hdr, name=event_hdu.name
            )
            hdulist[1] = bt
        if args.outfile is None:
            # Overwrite the existing file
            log.info(f"Overwriting existing FITS file {args.eventfile}")
            hdulist.flush(verbose=True, output_verify="warn")
        else:
            # Write to new output file
            log.info(f"Writing output FITS file {args.outfile}")
            hdulist.writeto(
                args.outfile, overwrite=True, checksum=True, output_verify="warn"
            )

    return 0
