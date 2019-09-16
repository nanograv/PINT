from __future__ import absolute_import, print_function, division
from astropy.time.formats import TimeFormat
from astropy.time.utils import day_frac
import astropy._erfa as erfa
import numpy


def safe_kind_conversion(values, dtype):
    try:
        from collections.abc import Iterable
    except ImportError:
        from collections import Iterable
    if isinstance(values, Iterable):
        return numpy.asarray(values, dtype=dtype)
    else:
        return dtype(values)


class TimePulsarMJD(TimeFormat):
    """MJD using tempo/tempo2 convention for time within leap second days.
    This is only relevant if scale='utc', otherwise will act like the
    standard astropy MJD time format."""

    name = 'pulsar_mjd'
    # This can be removed once we only support astropy >=3.1.
    # The str(c) is necessary for python2/numpy -> no unicode literals...
    _new_ihmsfs_dtype = numpy.dtype([(str(c), numpy.intc) for c in 'hmsf'])

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)
        if self._scale == 'utc':
            # To get around leap second issues, first convert to YMD,
            # then back to astropy/ERFA-convention jd1,jd2 using the
            # ERFA dtf2d() routine which handles leap seconds.
            v1, v2 = day_frac(val1, val2)
            (y,mo,d,f) = erfa.jd2cal(erfa.DJM0+v1,v2)
            # Fractional day to HMS.  Uses 86400-second day always.
            # Seems like there should be a ERFA routine for this..
            h = safe_kind_conversion(numpy.floor(f*24.0), dtype=int)
            m = safe_kind_conversion(numpy.floor(numpy.remainder(f*1440.0,60)),
                              dtype=int)
            s = numpy.remainder(f*86400.0,60)
            self.jd1, self.jd2 = erfa.dtf2d('UTC',y,mo,d,h,m,s)
        else:
            self.jd1, self.jd2 = day_frac(val1, val2)
            self.jd1 += erfa.DJM0

    @property
    def value(self):
        if self._scale == 'utc':
            # Do the reverse of the above calculation
            # Note this will return an incorrect value during
            # leap seconds, so raise an exception in that
            # case.
            y, mo, d, hmsf = erfa.d2dtf('UTC',9,self.jd1,self.jd2)
            # For ASTROPY_LT_3_1, convert to the new structured array dtype that
            # is returned by the new erfa gufuncs.
            if not hmsf.dtype.names:
                hmsf = hmsf.view(self._new_ihmsfs_dtype)
            if numpy.any(hmsf['s'] == 60):
                raise ValueError('UTC times during a leap second cannot be represented in pulsar_mjd format')
            j1, j2 = erfa.cal2jd(y,mo,d)
            return (j1 - erfa.DJM0 + j2) + (hmsf['h']/24.0
                    + hmsf['m']/1440.0
                    + hmsf['s']/86400.0
                    + hmsf['f']/86400.0e9)
        else:
            # As in TimeMJD
            return (self.jd1 - erfa.DJM0) + self.jd2
