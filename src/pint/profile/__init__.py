"""Tools for working with pulse profiles.

The key tool here is FFTFIT (:func:`pint.profile.fftfit_full`), which allows
one to find the phase shift that optimally aligns a template with a profile,
but there are also tools here for doing those shifts and generating useful
profiles.
"""

import numpy as np
import scipy.stats
from numpy.fft import rfft, irfft

import pint.profile.fftfit_aarchiba
import pint.profile.fftfit_nustar
import pint.profile.fftfit_presto

__all__ = [
    "fftfit_full",
    "fftfit_basic",
    "FFTFITResult",
    "fftfit_cprof",
    "fftfit_classic",
    "wrap",
    "vonmises_profile",
    "upsample",
    "shift",
]


class FFTFITResult:
    """Summary of the results of an FFTFit operation.

    Not all of these attributes may be present in every object; which are
    returned depends on the algorithm used and the options it is passed.

    If these quantities are available, then
    r.scale*shift(template, r.shift) + r.offset
    should be as close as possible to the profile used in the fitting.

    Attributes
    ----------
    shift : float
        The shift required to make the template match the profile. Between 0 and 1.
    scale : float
        The amount the template must be scaled by to match the profile.
    offset : float
        The amount to add to the scaled template to match the profile.
    uncertainty : float
        The estimated one-sigma uncertainty in the shift attribute.
    """

    pass


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

    The returned object, a :class:`pint.profile.FFTFITResult`, has a
    `.shift` attribute indicating the optimal shift,
    a `.uncertainty` attribute containting an estimate of the uncertainty, and
    possibly certain other attributes depending on which version of the code is
    run.

    The ``.shift`` attribute is computed so that ``shift(template, r.shift)`` is
    as closely aligned with ``profile`` as possible.

    Parameters
    ----------
    template : array
        The template representing the ideal pulse profile.
    profile : array
        The observed profile the template should be aligned with.
    code : "aarchiba", "nustar", "presto"
        Which underlying algorithm and code should be used to carry out the
        operation. Generally the "aarchiba" code base is the best tested
        under idealized circumstances, and "presto" is the compiled FORTRAN
        code FFTFIT that has been in use for a very long time (but is not
        available unless PINT has access to the compiled code base of PRESTO).
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
    """Return the optimal phase shift to match template to profile.

    This calls :func:`pint.profile.fftfit_full` and extracts the ``.shift`` attribute.

    Parameters
    ----------
    template : array
        The template representing the ideal pulse profile.
    profile : array
        The observed profile the template should be aligned with.
    code : "aarchiba", "nustar", "presto"
        Which code to use. See :func:`pint.profile.fftfit_full` for details.
    """
    if code == "aarchiba":
        return pint.profile.fftfit_aarchiba.fftfit_basic(template, profile)
    else:
        return fftfit_full(template, profile, code=code).shift


def fftfit_cprof(template):
    """Transform a template for use with fftfit_classic.

    Emulate the version of fftfit.cprof in PRESTO.
    Returns results suitable for :func:`pint.profile.fftfit_classic`.

    Parameter
    ---------
    template : array

    Returns
    -------
    c : float
        The constant term? This may not agree with PRESTO.
    amp : array
        Real values representing the Fourier amplitudes of the template, not
        including the constant term.
    pha : array
        Real values indicating the angles of the Fourier coefficients of the
        template, not including the constant term; the sign is the negative
        of that returned by ``np.fft.rfft``.
    """
    tc = rfft(template)
    tc *= np.exp(-2.j*np.pi*np.arange(len(tc))/len(template))
    return 2*tc, 2*np.abs(tc)[1:], -np.angle(tc)[1:]


def fftfit_classic(profile, template_amplitudes, template_angles, code="aarchiba"):
    """Emulate the version of fftfit in PRESTO.

    This has a different calling and return convention.
    The template can be transformed appropriately with
    :func:`pint.profile.fftfit_cprof`.

    Parameters
    ----------
    profile : array
        The observed profile.
    template_amplitudes : array
        Real values representing the Fourier amplitudes of the template, not
        including the constant term.
    template_angles : array
        Real values indicating the angles of the Fourier coefficients of the
        template, not including the constant term; the sign is the negative
        of that returned by ``np.fft.rfft``.

    Returns
    -------
    shift : float
        The shift, in bins, plus one.
    eshift : float
        The uncertainty in the shift, in bins.
    snr : float
        Some kind of signal-to-noise ratio; not implemented.
    esnr : float
        Uncertainty in the above; not implemented.
    b : float
        Unknown; not implemented.
    errb : float
        Uncertainty in the above; not implemented.
    ngood
        Unknown, maybe the number of harmonics used? Not implemented.
    """
    if code == "presto":
        import presto.fftfit

        return presto.fftfit.fftfit(profile, template_amplitudes, template_angles)
    if len(profile) % 2:
        raise ValueError("fftfit_classic only works on even-length profiles")
    template_f = np.zeros(len(template_amplitudes) + 1, dtype=complex)
    template_f[1:] = template_amplitudes * np.exp(-1.0j * template_angles)
    template = irfft(template_f)
    r = fftfit_full(template, profile, code=code)

    shift = (r.shift % 1) * len(profile) 
    eshift = r.uncertainty * len(profile)
    snr = np.nan
    esnr = np.nan
    b = np.nan
    errb = np.nan
    ngood = np.nan

    return shift, eshift, snr, esnr, b, errb, ngood
