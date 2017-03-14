"""FITS handling functions"""

import astropy.io.fits as pyfits
import numpy as np
from astropy import log
from astropy.extern import six
try:
    from astropy.erfa import DAYSEC as SECS_PER_DAY
except ImportError:
    from astropy._erfa import DAYSEC as SECS_PER_DAY
from .utils import fortran_float

def read_fits_event_mjds_tuples(event_hdu,timecolumn='TIME'):
    """Read a set of MJDs from a FITS HDU, with proper converstion of times to MJD

    The FITS time format is defined here:
    https://heasarc.gsfc.nasa.gov/docs/journal/timing3.html

    Returns
    -------
    mjds: MJDs returned are tuples of two doubles (jd1, jd2), as use by
        astropy Time() objects.

    """

    event_hdr=event_hdu.header
    event_dat=event_hdu.data

    # Collect TIMEZERO
    # IMPORTANT: TIMEZERO is in SECONDS (not days)!
    try:
        TIMEZERO = np.longdouble(event_hdr['TIMEZERO'])
    except KeyError:
        TIMEZERO = np.longdouble(event_hdr['TIMEZERI']) + np.longdouble(event_hdr['TIMEZERF'])
    log.info("TIMEZERO = {0}".format(TIMEZERO))

    # Collect MJDREF
    try:
        MJDREF = np.longdouble(event_hdr['MJDREF'])
    except KeyError:
        # Here I have to work around an issue where the MJDREFF key is stored
        # as a string in the header and uses the "1.234D-5" syntax for floats, which
        # is not supported by Python
        if isinstance(event_hdr['MJDREFF'],six.string_types):
            MJDREF = np.longdouble(event_hdr['MJDREFI']) + \
            fortran_float(event_hdr['MJDREFF'])
        else:
            MJDREF = np.longdouble(event_hdr['MJDREFI']) + np.longdouble(event_hdr['MJDREFF'])
    log.info("MJDREF = {0}".format(MJDREF))

    # Should check timecolumn units to be sure they are seconds!

    # MJD = (TIMECOLUMN + TIMEZERO)/SECS_PER_DAY + MJDREF
    mjds = np.array([(MJDREF, tt) for tt in (event_dat.field(timecolumn)
        + TIMEZERO)/SECS_PER_DAY])

    return mjds

def read_fits_event_mjds(event_hdu,timecolumn='TIME'):
    """Read a set of MJDs from a FITS HDU, with proper converstion of times to MJD

    The FITS time format is defined here:
    https://heasarc.gsfc.nasa.gov/docs/journal/timing3.html

    MJDs returned are double precision floats
    """

    event_hdr=event_hdu.header
    event_dat=event_hdu.data

    # Collect TIMEZERO
    # IMPORTANT: TIMEZERO is in SECONDS (not days)!
    try:
        TIMEZERO = np.float(event_hdr['TIMEZERO'])
    except KeyError:
        TIMEZERO = np.float(event_hdr['TIMEZERI']) + np.float(event_hdr['TIMEZERF'])
    log.info("TIMEZERO = {0}".format(TIMEZERO))

    # Collect MJDREF
    try:
        MJDREF = np.float(event_hdr['MJDREF'])
    except KeyError:
        # Here I have to work around an issue where the MJDREFF key is stored
        # as a string in the header and uses the "1.234D-5" syntax for floats, which
        # is not supported by Python
        if isinstance(event_hdr['MJDREFF'],six.string_types):
            MJDREF = np.float(event_hdr['MJDREFI']) + \
            fortran_float(event_hdr['MJDREFF'])
        else:
            MJDREF = np.float(event_hdr['MJDREFI']) + np.float(event_hdr['MJDREFF'])
    log.info("MJDREF = {0}".format(MJDREF))

    # Should check timecolumn units to be sure they are seconds!

    # MJD = (TIMECOLUMN + TIMEZERO)/SECS_PER_DAY + MJDREF
    mjds = (np.array(event_dat.field(timecolumn),dtype=np.float)+ TIMEZERO)/SECS_PER_DAY + MJDREF

    return mjds
