#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import sys

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy import log

import pint.models
import pint.residuals
import pint.toa as toa
from pint.event_toas import (
    load_NICER_TOAs,
    load_NuSTAR_TOAs,
    load_RXTE_TOAs,
    load_XMM_TOAs,
)
from pint.eventstats import h2sig, hm
from pint.observatory.nicer_obs import NICERObs
from pint.observatory.nustar_obs import NuSTARObs
from pint.observatory.rxte_obs import RXTEObs
from pint.plot_utils import phaseogram_binned
from pint.pulsar_mjd import Time

__all__ = ["main"]


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Use PINT to compute event phases and make plots of photon event files."
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
        "--plotfile", help="Output figure file name (default=None)", default=None
    )
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
    parser.add_argument(
        "--ephem", help="Planetary ephemeris to use (default=DE421)", default="DE421"
    )
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
    #    parser.add_argument("--fix",help="Apply 1.0 second offset for NICER", action='store_true', default=False)
    args = parser.parse_args(argv)

    # If outfile is specified, that implies addphase
    if args.outfile is not None:
        args.addphase = True

    # If plotfile is specified, that implies plot
    if args.plotfile is not None:
        args.plot = True

    # Read event file header to figure out what instrument is is from
    hdr = pyfits.getheader(args.eventfile, ext=1)

    log.info(
        "Event file TELESCOPE = {0}, INSTRUMENT = {1}".format(
            hdr["TELESCOP"], hdr["INSTRUME"]
        )
    )

    if hdr["TELESCOP"] == "NICER":

        # Instantiate NICERObs once so it gets added to the observatory registry
        if args.orbfile is not None:
            log.info("Setting up NICER observatory")
            NICERObs(name="NICER", FPorbname=args.orbfile, tt2tdb_mode="pint")
        # Read event file and return list of TOA objects
        try:
            tl = load_NICER_TOAs(args.eventfile)
        except KeyError:
            log.error(
                "Observatory not recognized.  This probably means you need to provide an orbit file or barycenter the event file."
            )
            sys.exit(1)
    elif hdr["TELESCOP"] == "XTE":

        # Instantiate RXTEObs once so it gets added to the observatory registry
        if args.orbfile is not None:
            # Determine what observatory type is.
            log.info("Setting up RXTE observatory")
            RXTEObs(name="RXTE", FPorbname=args.orbfile, tt2tdb_mode="pint")
        # Read event file and return list of TOA objects
        tl = load_RXTE_TOAs(args.eventfile)
    elif hdr["TELESCOP"].startswith("XMM"):
        # Not loading orbit file here, since that is not yet supported.
        tl = load_XMM_TOAs(args.eventfile)
    elif hdr["TELESCOP"].lower().startswith("nustar"):
        if args.orbfile is not None:
            log.info("Setting up NuSTAR observatory")
            NuSTARObs(name="NuSTAR", FPorbname=args.orbfile, tt2tdb_mode="pint")
        tl = load_NuSTAR_TOAs(args.eventfile)
    else:
        log.error(
            "FITS file not recognized, TELESCOPE = {0}, INSTRUMENT = {1}".format(
                hdr["TELESCOP"], hdr["INSTRUME"]
            )
        )
        sys.exit(1)

    # Now convert to TOAs object and compute TDBs and posvels
    if len(tl) == 0:
        log.error("No TOAs, exiting!")
        sys.exit(0)

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

    if args.addorbphase and (not hasattr(modelin, "binary_model_name")):
        log.error(
            "TimingModel does not include a binary model, which is required for "
            "computing orbital phases. Make sure you have BINARY and associated "
            "model parameters in your par file!"
        )
        raise ValueError("Model missing BINARY component.")

    # Discard events outside of MJD range
    if args.maxMJD is not None:
        tlnew = []
        print("pre len : ", len(tl))
        maxT = Time(float(args.maxMJD), format="mjd")
        print("maxT : ", maxT)
        for tt in tl:
            if tt.mjd < maxT:
                tlnew.append(tt)
        tl = tlnew
        print("post len : ", len(tlnew))

    ts = toa.get_TOAs_list(
        tl,
        ephem=args.ephem,
        include_bipm=args.use_bipm,
        include_gps=args.use_gps,
        planets=use_planets,
        tdb_method=args.tdbmethod,
    )
    ts.filename = args.eventfile
    #    if args.fix:
    #        ts.adjust_TOAs(TimeDelta(np.ones(len(ts.table))*-1.0*u.s,scale='tt'))

    print(ts.get_summary())
    mjds = ts.get_mjds()
    print(mjds.min(), mjds.max())

    # Compute model phase for each TOA
    iphss, phss = modelin.phase(ts, abs_phase=True)
    # ensure all postive
    negmask = phss < 0.0
    phases = np.where(negmask, phss + 1.0, phss)
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
        norbits = np.array(np.floor(orbits), dtype=np.long)
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

        if args.addphase:
            if len(hdulist[1].data) != len(phases):
                raise RuntimeError(
                    "Mismatch between length of FITS table ({0}) and length of phase array ({1})!".format(
                        len(hdulist[1].data), len(phases)
                    )
                )
            data_to_add["PULSE_PHASE"] = [phases, "D"]

        if args.absphase:
            data_to_add["ABS_PHASE"] = [iphss - negmask, "K"]

        if args.barytime:
            bats = modelin.get_barycentric_toas(ts)
            data_to_add["BARY_TIME"] = [bats, "D"]

        if args.addorbphase:
            if len(hdulist[1].data) != len(orbphases):
                raise RuntimeError(
                    "Mismatch between length of FITS table ({0}) and length of orbital phase array ({1})!".format(
                        len(hdulist[1].data), len(orbphases)
                    )
                )
            data_to_add["ORBIT_PHASE"] = [orbphases, "D"]
        # End if args.addorbphase

        for key in data_to_add.keys():
            if key in hdulist[1].columns.names:
                log.info("Found existing %s column, overwriting..." % key)
                # Overwrite values in existing Column
                hdulist[1].data[key] = data_to_add[key][0]
            else:
                # Construct and append new column, preserving HDU header and name
                log.info("Adding new %s column." % key)
                datacol.append(
                    pyfits.ColDefs(
                        [
                            pyfits.Column(
                                name=key,
                                format=data_to_add[key][1],
                                array=data_to_add[key][0],
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
