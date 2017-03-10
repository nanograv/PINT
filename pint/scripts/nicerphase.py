#!/usr/bin/env python
from __future__ import division, print_function

import os,sys
import numpy as np
import pint.toa as toa
import pint.models
import pint.residuals
import astropy.units as u
from pint.nicer_toas import nicer_phaseogram, load_NICER_TOAs
from pint.observatory.nicer_obs import NICERObs
from astropy.time import Time
from pint.eventstats import hmw, hm, h2sig
from astropy.coordinates import SkyCoord
from astropy import log
import astropy.io.fits as pyfits
import uuid

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Use PINT to compute event phases and make plots of NICER event files.")
    parser.add_argument("eventfile",help="NICER event FITS file name.")
    parser.add_argument("orbfile",help="Name of FPorbit file or 'none'.")
    parser.add_argument("parfile",help="par file to construct model from")
    parser.add_argument("--maxMJD",help="Maximum MJD to include in analysis", default=None)
    parser.add_argument("--plotfile",help="Output figure file name (default=None)", default=None)
    parser.add_argument("--addphase",help="Write FITS file with added phase column",
        default=False,action='store_true')
    parser.add_argument("--outfile",help="Output FITS file name (default=same as eventfile)", default=None)
    parser.add_argument("--planets",help="Use planetary Shapiro delay in calculations (default=False)", default=False, action="store_true")
    parser.add_argument("--ephem",help="Planetary ephemeris to use (default=DE421)", default="DE421")
    parser.add_argument("--plot",help="Show phaseogram plot.", action='store_true', default=False)
    args = parser.parse_args(argv)
    
    # If outfile is specified, that implies addphase
    if args.outfile is not None:
        args.addphase = True

    # Instantiate NICERObs once so it gets added to the observatory registry
    if not args.orbfile.lower() == 'none':
        NICERObs(name='NICER',FPorbname=args.orbfile)

    # Read in model
    modelin = pint.models.get_model(args.parfile)

    # Read event file and return list of TOA objects
    tl  = load_NICER_TOAs(args.eventfile)

    # Discard events outside of MJD range    
    if args.maxMJD is not None:
        tlnew = []
        print("pre len : ",len(tl))
        maxT = Time(float(args.maxMJD),format='mjd')
        print("maxT : ",maxT)
        for tt in tl:
            if tt.mjd < maxT:
                tlnew.append(tt)
        tl=tlnew
        print("post len : ",len(tlnew))

    # Now convert to TOAs object and compute TDBs and posvels
    ts = toa.TOAs(toalist=tl)
    ts.filename = args.eventfile
    ts.compute_TDBs()
    ts.compute_posvels(ephem=args.ephem,planets=args.planets)

    print(ts.get_summary())
    mjds = ts.get_mjds()
    print(mjds.min(),mjds.max())

    # Compute model phase for each TOA
    phss = modelin.phase(ts.table)[1]
    # ensure all postive
    phases = np.where(phss < 0.0, phss + 1.0, phss)
    mjds = ts.get_mjds()
    h = float(hm(phases))
    print("Htest : {0:.2f} ({1:.2f} sigma)".format(h,h2sig(h)))
    if args.plot:
        nicer_phaseogram(mjds,phases,bins=100,plotfile = args.plotfile)
        
    if args.addphase:
        # Read input FITS file (again).
        # If overwriting, open in 'update' mode
        if args.outfile is None:
            hdulist = pyfits.open(args.eventfile,mode='update')
        else:
            hdulist = pyfits.open(args.eventfile)
        event_hdu = hdulist[1]
        event_hdr=event_hdu.header
        event_dat=event_hdu.data
        if len(event_dat) != len(phases):
            raise RuntimeError('Mismatch between length of FITS table ({0}) and length of phase array ({1})!'.format(len(event_dat),len(phases)))
        if 'PULSE_PHASE' in event_hdu.columns.names:
            log.info('Found existing PULSE_PHASE column, overwriting...')
            # Overwrite values in existing Column
            event_dat['PULSE_PHASE'] = phases
        else:
            # Construct and append new column, preserving HDU header and name
            log.info('Adding new PULSE_PHASE column.')
            phasecol = pyfits.ColDefs([pyfits.Column(name='PULSE_PHASE', format='D',
                array=phases)])
            bt = pyfits.BinTableHDU.from_columns( event_hdu.columns + phasecol, 
                header=event_hdr,name=event_hdu.name)
            hdulist[1] = bt
        if args.outfile is None:
            # Overwrite the existing file
            log.info('Overwriting existing FITS file '+args.eventfile)
            hdulist.flush(verbose=True, output_verify='warn')
        else:
            # Write to new output file
            log.info('Writing output FITS file '+args.outfile)
            hdulist.writeto(args.outfile,overwrite=False, checksum=True, output_verify='warn')


