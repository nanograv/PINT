import numpy as np
from numpy.fft import rfft

import pint.profile

try:
    import presto.fftfit
except ImportError:
    presto = None


def fftfit_full(template, profile):
    if len(template) != len(profile):
        raise ValueError(
            "template has length {} but profile has length {}".format(
                len(template), len(profile)
            )
        )
    if len(template) > 2 ** 13:
        raise ValueError(
            "template has length {} which is too long".format(len(template))
        )
    _, amp, pha = pint.profile.fftfit_cprof(template)
    shift, eshift, snr, esnr, b, errb, ngood = presto.fftfit.fftfit(
        profile,
    )
    r = pint.profile.FFTFITResult()
    # Need to add 1 to the shift for some reason
    r.shift = pint.profile.wrap((shift + 1) / len(template))
    r.uncertainty = eshift / len(template)
    return r


def fftfit_basic(template, profile):
    return fftfit_full(template, profile).shift
