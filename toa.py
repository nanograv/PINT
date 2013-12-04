import astropy.constants as c
import astropy.time as t
import numpy as npy

class toa(object):
    """
    toa(MJD, error=0.0, obs='bary', freq=float("inf"), scale='utc',...)
        A time of arrival class.

        MJD will be stored in astropy.time.Time format, and can be
            passed as a double (not recommended), a string, a
            tuple of component parts (day and fraction of day).
        error is the TOA uncertainty in microseconds
        obs is the observatory name as defined in XXX
        freq is the observatory-centric frequency in MHz
        freq
        other keyword/value pairs can be specified as needed

    Example:
        >>> a = toa("54567.876876876876876", 4.5, freq=1400.0,
                    obs="GBT", backend="GUPPI")
        >>> print a
        54567.876876876876876: 4.500 us at 'GBT' at 1400.0000 MHz
        {'backend': 'GUPPI'}

    """
    __std_params = ['mjd', 'freq', 'error', 'obs', 'scale']

    def __init__(self, MJD, # required
                 error=0.0, obs='bary', freq=float("inf"), scale='utc', # with defaults
                 **kwargs):  # keyword args that are completely optional
        if type(MJD) is tuple:
            self.mjd = t.Time(*MJD, scale=scale, format='mjd', precision=9)
        else:
            if type(MJD) is not str:
                sys.stderr.write("Warning:  possibly loss of precision with %s" %
                                 str(type(MJD)))
            self.mjd = t.Time(MJD, scale=scale, format='mjd', precision=9)
        # set the other three required values
        self.obs = obs
        self.freq = freq
        self.error = error
        # set any other optional params
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __str__(self):
        s = str(self.mjd) + ": %.3f us at '%s' at %.4f MHz" % \
            (self.error, self.obs, self.freq)
        other_keys = [k for k in self.__dict__.keys() if k not in self.__std_params]
        if len(other_keys):
            s += "\n" + str(dict([(k,self.__dict__[k]) for k in other_keys]))
        return s

