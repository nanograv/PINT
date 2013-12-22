# utils.py
# Miscellaneous potentially-helpful functions
import string
import mpmath
import astropy.time

def fortran_float(x):
    """
    fortran_float(x)

    returns a copy of the input string with all 'D' or 'd' turned
    into 'e' characters.  Intended for dealing with exponential 
    notation in tempo1-generated parfiles.
    """
    return float(x.translate(string.maketrans('Dd','ee')))

def time_from_mjd_string(s):
    """
    time_from_mjd_string(s)

    Returns an astropy Time object generated from a MJD string input.
    """
    imjd_s,fmjd_s = s.split('.')
    imjd = int(imjd_s)
    fmjd = float("0." + fmjd_s)
    # TODO: what to do about scale?
    return astropy.time.Time(imjd,fmjd,scale='utc',format='mjd',
            precision=9)

def time_to_mjd_string(t):
    """
    time_to_mjd_string(t)

    print an MJD time with lots of digits.  astropy does not
    seem to provide this capability..
    """
    # XXX Assume that that jd1 represents an integer MJD, and jd2 
    # is the frac part.. is this legit??
    imjd = int(t.jd1 - astropy.time.core.MJD_ZERO)
    fmjd = t.jd2
    return str(imjd) + ('%.15f'%fmjd)[1:]

def time_to_mjd_mpf(t):
    """
    time_to_mjd_mpf(t)

    Return an astropy Time value as MJD in mpmath float format.  
    mpmath.mp.dps needs to be set to the desired precision before 
    calling this.
    """
    return mpmath.mpf(t.jd1 - astropy.time.core.MJD_ZERO) \
            + mpmath.mpf(t.jd2)

def timedelta_to_mpf_sec(t):
    """
    timedelta_to_mpf_sec(t):

    Return astropy TimeDelta as mpmath value in seconds.
    """
    return (mpmath.mpf(t.jd1) 
            + mpmath.mpf(t.jd2))*astropy.time.core.SECS_PER_DAY

