"""Tools for working with pulse profiles.

The key tool here is `fftfit`, which allows one to find the phase shift that
optimally aligns a template with a profile, but there are also tools here for
doing those shifts and generating useful profiles.
"""
import numpy as np
import scipy.stats
import pint.profile.fftfit_aarchiba
import pint.profile.fftfit_nustar
import pint.profile.fftfit_presto


__all__ = [
    "wrap",
    "vonmises_profile",
    "upsample",
    "shift",
    "fftfit_full",
    "fftfit_basic",
]


def wrap(a):
    """Wrap a floating-point number or array to the range -0.5 to 0.5."""
    return (a + 0.5) % 1 - 0.5


def zap_nyquist(profile):
    if len(profile) % 2:
        return profile
    else:
        c = np.fft.rfft(profile)
        c[-1] = 0
        return np.fft.irfft(c)


def vonmises_profile(kappa, n, phase=0):
    """Generate a profile based on a von Mises distribution.

    The von Mises distribution is a cyclic analogue of a Gaussian distribution. The width is
    specified by the parameter kappa, which for large kappa is approximately 1/(2*pi*sigma**2).
    """
    return np.diff(
        scipy.stats.vonmises(kappa).cdf(
            np.linspace(-2 * np.pi * phase, 2 * np.pi * (1 - phase), n + 1)
        )
    )


def upsample(profile, factor):
    """Produce an up-sampled version of a pulse profile.

    This uses a Fourier algorithm, with zero in the new Fourier coefficients.
    """
    output_len = len(profile) * factor
    if output_len % 2:
        raise ValueError("Cannot cope with odd output profile lengths")
    c = np.fft.rfft(profile)
    output_c = np.zeros(output_len // 2 + 1, dtype=complex)
    output_c[: len(c)] = c * factor
    output = np.fft.irfft(output_c)
    assert len(output) == output_len
    return output


def shift(profile, phase):
    """Shift a profile in phase.

    This is a shift towards later phases - if your profile has a 1 in bin zero
    and apply a phase shift of 1/4, the 1 will now be in bin n/4.  If the
    profile has even length, do not modify the Nyquist component.
    """
    c = np.fft.rfft(profile)
    if len(profile) % 2:
        c *= np.exp(-2.0j * np.pi * phase * np.arange(len(c)))
    else:
        c[:-1] *= np.exp(-2.0j * np.pi * phase * np.arange(len(c) - 1))
    return np.fft.irfft(c, len(profile))


def irfft_value(c, phase, n=None):
    """Evaluate the inverse real FFT at a particular position.

    If the phase is one of the usual grid points the result will agree with
    the results of `np.fft.irfft` there.

    No promises if n is small enough to imply truncation.
    """
    natural_n = (len(c) - 1) * 2
    if n is None:
        n = natural_n
    phase = np.asarray(phase)
    s = phase.shape
    phase = np.atleast_1d(phase)
    c = np.array(c)
    c[0] /= 2
    if n == natural_n:
        c[-1] /= 2
    return (
        (
            c[:, None]
            * np.exp(2.0j * np.pi * phase[None, :] * np.arange(len(c))[:, None])
        )
        .sum(axis=0)
        .real
        * 2
        / n
    ).reshape(s)


def fftfit_full(template, profile, code="aarchiba"):
    """Match template to profile and return match properties.

    The returned object has a `.shift` attribute indicating the optimal shift,
    a `.uncertainty` attribute containting an estimate of the uncertainty, and
    possibly certain other attributes depending on which version of the code is
    run.
    """
    if code == "aarchiba":
        return pint.profile.fftfit_aarchiba.fftfit_full(template, profile)
    elif code == "nustar":
        return pint.profile.fftfit_nustar.fftfit_full(template, profile)
    elif code == "presto":
        if pint.profile.fftfit_presto.presto is None:
            raise ValueError("The PRESTO compiled code is not available")
        return pint.profile.fftfit_presto.fftfit_full(template, profile)
    else:
        raise ValueError("Unrecognized FFTFIT implementation {}".format(code))


def fftfit_basic(template, profile, code="aarchiba"):
    """Return the optimal phase shift to match template to profile."""
    if code == "aarchiba":
        return pint.profile.fftfit_aarchiba.fftfit_basic(template, profile)
    else:
        return fftfit_full(template, profile, code=code).shift
