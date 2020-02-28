#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import argparse

import astropy.io.fits as pyfits
import astropy.units as u
import numpy as np
from astropy import log
from astropy.coordinates import SkyCoord

import pint.models
import pint.residuals
import pint.toa as toa
from pint.eventstats import h2sig, hmw
from pint.fermi_toas import load_Fermi_TOAs
from pint.observatory.fermi_obs import FermiObs
from pint.plot_utils import phaseogram
from pint.pulsar_mjd import Time

__all__ = ["main"]

# log.setLevel('DEBUG')


def main(argv=None):

    parser = argparse.ArgumentParser(
        description="Use PINT to compute H-test and plot Phaseogram from a Fermi FT1 event file."
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
    parser.add_argument(
        "--plotfile", help="Output figure file name (default=None)", default=None
    )
    parser.add_argument(
        "--maxMJD", help="Maximum MJD to include in analysis", default=None
    )
    parser.add_argument(
        "--outfile",
        help="Output figure file name (default is to overwrite input file)",
        default=None,
    )
    parser.add_argument(
        "--planets",
        help="Use planetary Shapiro delay in calculations (default=False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--ephem", help="Planetary ephemeris to use (default=DE421)", default="DE421"
    )
    args = parser.parse_args(argv)

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
        # Instantiate FermiObs once so it gets added to the observatory registry
        FermiObs(name="Fermi", ft2name=args.ft2)

    # Read event file and return list of TOA objects
    tl = load_Fermi_TOAs(args.eventfile, weightcolumn=args.weightcol, targetcoord=tc)

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

    # Now convert to TOAs object and compute TDBs and posvels
    # For Fermi, we are not including GPS or TT(BIPM) corrections
    ts = toa.get_TOAs_list(
        tl,
        include_gps=False,
        include_bipm=False,
        planets=args.planets,
        ephem=args.ephem,
    )
    ts.filename = args.eventfile

    print(ts.get_summary())
    mjds = ts.get_mjds()
    print(mjds.min(), mjds.max())

    # Compute model phase for each TOA
    iphss, phss = modelin.phase(ts, abs_phase=True)
    # ensure all postive
    phss = phss.to(u.cycle).value
    phases = np.where(phss < 0.0, phss + 1.0, phss)
    mjds = ts.get_mjds()
    weights = np.array([w["weight"] for w in ts.table["flags"]])
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
        if len(event_dat) != len(phases):
            raise RuntimeError(
                "Mismatch between length of FITS table ({0}) and length of phase array ({1})!".format(
                    len(event_dat), len(phases)
                )
            )
        if "PULSE_PHASE" in event_hdu.columns.names:
            log.info("Found existing PULSE_PHASE column, overwriting...")
            # Overwrite values in existing Column
            event_dat["PULSE_PHASE"] = phases
        else:
            # Construct and append new column, preserving HDU header and name
            log.info("Adding new PULSE_PHASE column.")
            phasecol = pyfits.ColDefs(
                [pyfits.Column(name="PULSE_PHASE", format="D", array=phases)]
            )
            bt = pyfits.BinTableHDU.from_columns(
                event_hdu.columns + phasecol, header=event_hdr, name=event_hdu.name
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

    return 0
