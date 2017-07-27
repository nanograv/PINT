#!/usr/bin/env python
from __future__ import division, print_function

import os,sys
import numpy as np
import pint.toa as toa
import pint.models
import pint.residuals
import astropy.units as u
from pint.event_toas import load_NICER_TOAs
from pint.event_toas import load_RXTE_TOAs
from pint.event_toas import load_XMM_TOAs
from pint.plot_utils import phaseogram_binned
from pint.observatory.nicer_obs import NICERObs
from pint.observatory.rxte_obs import RXTEObs
from astropy.time import Time
from pint.eventstats import hmw, hm, h2sig
from astropy.coordinates import SkyCoord
from astropy import log
import astropy.io.fits as pyfits
import uuid

def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Use PINT to compute event phases and make plots of photon event files.")
    parser.add_argument("eventfile",help="Photon event FITS file name (e.g. from NICER, RXTE, XMM, Chandra).")
    parser.add_argument("parfile",help="par file to construct model from")
    parser.add_argument("--orbfile",help="Name of orbit file", default=None)
    parser.add_argument("--maxMJD",help="Maximum MJD to include in analysis", default=None)
    parser.add_argument("--plotfile",help="Output figure file name (default=None)", default=None)
    parser.add_argument("--addphase",help="Write FITS file with added phase column",default=False,action='store_true')
    parser.add_argument("--absphase",help="Write FITS file with integral portion of pulse phase (ABS_PHASE)",default=False,action='store_true')
    parser.add_argument("--barytime",help="Write FITS file with a column containing the barycentric time as double precision MJD.",default=False,action='store_true')
    parser.add_argument("--outfile",help="Output FITS file name (default=same as eventfile)", default=None)
    parser.add_argument("--planets",help="Use planetary Shapiro delay in calculations (default=False)", default=False, action="store_true")
    parser.add_argument("--ephem",help="Planetary ephemeris to use (default=DE421)", default="DE421")
    parser.add_argument("--plot",help="Show phaseogram plot.", action='store_true', default=False)
    args = parser.parse_args(argv)

    # If outfile is specified, that implies addphase
    if args.outfile is not None:
        args.addphase = True

    # Read event file header to figure out what instrument is is from
    hdr = pyfits.getheader(args.eventfile,ext=1)

    log.info('Event file TELESCOPE = {0}, INSTRUMENT = {1}'.format(hdr['TELESCOP'],
        hdr['INSTRUME']))
    if hdr['TELESCOP'] == 'NICER':
        # Instantiate NICERObs once so it gets added to the observatory registry
        if args.orbfile is not None:
            log.info('Setting up NICER observatory')
            NICERObs(name='NICER',FPorbname=args.orbfile,tt2tdb_mode='none')
        # Read event file and return list of TOA objects
        try:
            tl  = load_NICER_TOAs(args.eventfile)
        except KeyError:
            log.error("Observatory not recognized.  This probably means you need to provide an orbit file or barycenter the event file.")
            sys.exit(1)
    elif hdr['TELESCOP'] == 'XTE':
        # Instantiate RXTEObs once so it gets added to the observatory registry
        if args.orbfile is not None:
            # Determine what observatory type is.
            log.info('Setting up RXTE observatory')
            RXTEObs(name='RXTE',FPorbname=args.orbfile,tt2tdb_mode='none')
        # Read event file and return list of TOA objects
        tl  = load_RXTE_TOAs(args.eventfile)
    elif hdr['TELESCOP'].startswith('XMM'):
        # Not loading orbit file here, since that is not yet supported.
        tl  = load_XMM_TOAs(args.eventfile)
    else:
        log.error("FITS file not recognized, TELESCOPE = {0}, INSTRUMENT = {1}".format(
            hdr['TELESCOP'], hdr['INSTRUME']))
        sys.exit(1)

    # Read in model
    modelin = pint.models.get_model(args.parfile)

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
    phases = np.where(phss < 0.0 * u.cycle, phss + 1.0 * u.cycle, phss)
    h = float(hm(phases))
    print("Htest : {0:.2f} ({1:.2f} sigma)".format(h,h2sig(h)))
    if args.plot:
        phaseogram_binned(mjds,phases,bins=100,plotfile = args.plotfile)

    if args.addphase:
        # Read input FITS file (again).
        # If overwriting, open in 'update' mode
        if args.outfile is None:
            hdulist = pyfits.open(args.eventfile,mode='update')
        else:
            hdulist = pyfits.open(args.eventfile)
        if len(hdulist[1].data) != len(phases):
            raise RuntimeError('Mismatch between length of FITS table ({0}) and length of phase array ({1})!'.format(len(hdulist[1].data),len(phases)))
        data_to_add = {'PULSE_PHASE':[phases,'D']}
        if args.absphase:
            data_to_add['ABS_PHASE'] = [modelin.phase(ts.table)[0],'K']
        if args.barytime:
            tdbs = np.asarray([t.mjd for t in ts.table['tdb']])
            data_to_add['BARY_TIME'] = [tdbs,'D']
        for key in data_to_add.keys():
            if key in hdulist[1].columns.names:
                log.info('Found existing %s column, overwriting...'%key)
                # Overwrite values in existing Column
                hdulist[1].data[key] = data_to_add[key][0]
            else:
                # Construct and append new column, preserving HDU header and name
                log.info('Adding new %s column.'%key)
                datacol = pyfits.ColDefs([pyfits.Column(name=key, 
                    format=data_to_add[key][1], array=data_to_add[key][0])])
                bt = pyfits.BinTableHDU.from_columns(
                    hdulist[1].columns + datacol, header=hdulist[1].header,
                    name=hdulist[1].name)
                hdulist[1] = bt
        if args.outfile is None:
            # Overwrite the existing file
            log.info('Overwriting existing FITS file '+args.eventfile)
            hdulist.flush(verbose=True, output_verify='warn')
        else:
            # Write to new output file
            log.info('Writing output FITS file '+args.outfile)
            hdulist.writeto(args.outfile,overwrite=True, checksum=True, output_verify='warn')
