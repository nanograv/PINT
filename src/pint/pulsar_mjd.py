"""PulsarMJD special time format.

MJDs seem simple but if you want to use them with the UTC time scale you have
to deal with the fact that every so often a UTC day is either 86401 or 86399
seconds long. :class:`astropy.time.Time` has a policy on how to interpret MJDs
in UTC, and in that time scale all times can be expressed in MJD, but the rate
at which MJDs advance is not one day per 86400 SI seconds. (This technique,
when applied to UNIX time, is called a "leap_smear_" and is used by all Google
APIs.) This is not how (some? all?) observatories construct the MJDs they
record; observatories record times by converting UNIX time to MJDs in a way
that ignores leap seconds; this means that there is more than one time that
will produce the same MJD on a leap second day. (Observatories usually use a
maser to keep very accurate time, but this is used only to identify the
beginnings of seconds; NTP is used to determine which second to record.) We
therefore introduce the "pulsar_mjd" time format to capture the way in which
our data actually occurs.

An MJD expressed in the "pulsar_mjd" time scale will never occur during a
leap second. No negative leap seconds have yet been inserted, that is,
all days have been either 86400 or 86401 seconds long. If a negative leap
second does occur, it is not totally clear what will happen if an MJD
is provided that corresponds to a nonexistent time.

This is not a theoretical consideration: at least one pulsar observer
was observing the sky at the moment a leap second was introduced.

.. _leap_smear: https://developers.google.com/time/smear
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from astropy.time.formats import TimeFormat

from pint.utils import (
    jds_to_mjds,
    jds_to_mjds_pulsar,
    mjds_to_jds,
    mjds_to_jds_pulsar,
    mjds_to_str,
    str_to_mjds,
)


class PulsarMJD(TimeFormat):
    """Change handling of days with leap seconds.

    MJD using tempo/tempo2 convention for time within leap second days.
    This is only relevant if scale='utc', otherwise will act like the
    standard astropy MJD time format.
    """

    name = "pulsar_mjd"

    def set_jds(self, val1, val2):
        self._check_scale(self._scale)
        # for times before the first leap second we don't need to do anything
        # it's annoying to handle parts-of-arrays like this though
        if self._scale == "utc":
            self.jd1, self.jd2 = mjds_to_jds_pulsar(val1, val2)
        else:
            self.jd1, self.jd2 = mjds_to_jds(val1, val2)

    @property
    def value(self):
        if self._scale == "utc":
            mjd1, mjd2 = jds_to_mjds_pulsar(self.jd1, self.jd2)
            return mjd1 + np.longdouble(mjd2)
        else:
            mjd1, mjd2 = jds_to_mjds(self.jd1, self.jd2)
            return mjd1 + np.longdouble(mjd2)


class MJDLong(TimeFormat):
    """Support conversion of MJDs to or from long double.

    On machines with 80-bit floating-point (in particular x86 hardware), this
    representation gives a quantization error a bit under a nanosecond.
    """

    name = "mjd_long"


class MJDString(TimeFormat):
    """Support full-accuracy reading and writing of MJDs in string form."""

    name = "mjd_string"

    # FIXME: arrays!
    def set_jds(self, val1, val2):
        # What do I do with val2?
        self._check_scale(self._scale)
        self.jd1, self.jd2 = str_to_mjds(val1)

    @property
    def value(self):
        return mjds_to_str(self.jd1, self.jd2)


class PulsarMJDLong(TimeFormat):
    """Support conversion of pulsar MJDs to or from long double."""


class PulsarMJDString(TimeFormat):
    """Support full-accuracy reading and writing of pulsar MJDs in string form."""

    name = "pulsar_mjd_string"

    # FIXME: arrays!
    def set_jds(self, val1, val2):
        # What do I do with val2?
        self._check_scale(self._scale)
        self.jd1, self.jd2 = str_to_mjds(val1)

    @property
    def value(self):
        return mjds_to_str(self.jd1, self.jd2)
