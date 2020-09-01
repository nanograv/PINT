import numpy as np
from numpy.fft import rfft
from pint.profile.fftfit_aarchiba import wrap

try:
    import presto.fftfit
except ImportError:
    presto = None

class FFTFITResult:
    pass

def fftfit_full(template, profile):
    if len(template) != len(profile):
        raise ValueError("template has length {} but profile has length {}".format(len(template),len(profile)))
    if len(template) > 2**13:
        raise ValueError("template has length {} which is too long".format(len(template)))
    tc = rfft(template)
    shift, eshift, snr, esnr, b, errb, ngood = presto.fftfit.fftfit(profile, np.abs(tc)[1:], -np.angle(tc)[1:]) 
    r = FFTFITResult()
    # Need to add 1 to the shift for some reason
    r.shift = wrap((shift + 1)/len(template))
    r.uncertainty = eshift/len(template)
    return r

def fftfit_basic(template, profile):
    return fftfit_full(template, profile).shift

