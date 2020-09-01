# by mateobachetti
# from https://github.com/NuSTAR/nustar-clock-utils/blob/master/nuclockutils/diagnostics/fftfit.py
from collections import namedtuple
import numpy as np
from scipy.optimize import minimize, brentq


def find_delay_with_ccf(amp, pha):
    nh = 32
    nprof = nh * 2
    CCF = np.zeros(64, dtype=np.complex)
    CCF[:nh] = amp[:nh] * np.cos(pha[:nh]) + 1.0j * amp[:nh] * np.sin(pha[:nh])
    CCF[nprof : nprof - nh : -1] = np.conj(CCF[nprof : nprof - nh : -1])
    CCF[nh // 2 : nh] = 0
    CCF[nprof - nh // 2 : nprof - nh : -1] = 0
    ccf = np.fft.ifft(CCF)

    imax = np.argmax(ccf.real)
    cmax = ccf[imax]
    shift = normalize_phase_0d5(imax / nprof)

    # plt.figure()
    # plt.plot(ccf.real)
    # plt.show()
    # fb=np.real(cmax)
    # ia=imax-1
    # if(ia == -1): ia=nprof-1
    # fa=np.real(ccf[ia])
    # ic=imax+1
    # if(ic == nprof): ic=0
    # fc=np.real(ccf[ic])
    # if ((2*fb-fc-fa) != 0):
    #     shift=imax+0.5*(fa-fc)/(2*fb-fc-fa)
    #     shift = normalize_phase_0d5(shift / nprof)
    return shift


def best_phase_func(tau, amp, pha, ngood=20):
    # tau = params['tau']
    # good = slice(1, idx.size // 2 + 1)
    good = slice(1, ngood + 1)
    idx = np.arange(1, ngood + 1, dtype=int)
    res = np.sum(idx * amp[good] * np.sin(-pha[good] + TWOPI * idx * tau))
    # print(tau, res)
    return res


TWOPI = 2 * np.pi


def chi_sq(b, tau, P, S, theta, phi, ngood=20):
    # tau = params['tau']
    # good = slice(1, idx.size // 2 + 1)
    good = slice(1, ngood + 1)
    idx = np.arange(1, ngood + 1, dtype=int)
    angle_diff = phi[good] - theta[good] + TWOPI * idx * tau
    exp_term = np.exp(1.0j * angle_diff)

    to_square = P[good] - b * S[good] * exp_term
    res = np.sum((to_square * to_square.conj()))

    return res.real


def chi_sq_alt(b, tau, P, S, theta, phi, ngood=20):
    # tau = params['tau']
    # good = slice(1, idx.size // 2 + 1)
    good = slice(1, ngood + 1)
    idx = np.arange(1, ngood + 1, dtype=int)
    angle_diff = phi[good] - theta[good] + TWOPI * idx * tau
    chisq_1 = P[good] ** 2 + b ** 2 * S[good] ** 2
    chisq_2 = -2 * b * P[good] * S[good] * np.cos(angle_diff)
    res = np.sum(chisq_1 + chisq_2)

    return res


FFTFITResult = namedtuple(
    "FFTFITResult", ["mean_amp", "std_amp", "mean_phase", "std_phase"]
)


def fftfit(prof, template):
    """Align a template to a pulse profile.
    Parameters
    ----------
    prof : array
        The pulse profile
    template : array, default None
        The template of the pulse used to perform the TOA calculation. If None,
        a simple sinusoid is used
    Returns
    -------
    mean_amp, std_amp : floats
        Mean and standard deviation of the amplitude
    mean_phase, std_phase : floats
        Mean and standard deviation of the phase
    """
    prof = prof - np.mean(prof)

    nbin = len(prof)

    template = template - np.mean(template)

    temp_ft = np.fft.fft(template)
    prof_ft = np.fft.fft(prof)
    freq = np.fft.fftfreq(prof.size)
    good = freq == freq

    P = np.abs(prof_ft[good])
    theta = np.angle(prof_ft[good])
    S = np.abs(temp_ft[good])
    phi = np.angle(temp_ft[good])

    assert np.allclose(temp_ft[good], S * np.exp(1.0j * phi))
    assert np.allclose(prof_ft[good], P * np.exp(1.0j * theta))

    amp = P * S
    pha = theta - phi

    mean = np.mean(amp)
    ngood = np.count_nonzero(amp >= mean)

    dph_ccf = find_delay_with_ccf(amp, pha)

    idx = np.arange(0, len(P), dtype=int)
    sigma = np.std(prof_ft[good])

    def func_to_minimize(tau):
        return best_phase_func(-tau, amp, pha, ngood=ngood)

    start_val = dph_ccf
    start_sign = np.sign(func_to_minimize(start_val))

    count_down = 0
    count_up = 0
    trial_val_up = start_val
    trial_val_down = start_val
    while True:
        if np.sign(func_to_minimize(trial_val_up)) != start_sign:
            best_dph = trial_val_up
            break
        if np.sign(func_to_minimize(trial_val_down)) != start_sign:
            best_dph = trial_val_down
            break
        trial_val_down -= 1 / nbin
        count_down += 1
        trial_val_up += 1 / nbin
        count_up += 1

    a, b = best_dph - 2 / nbin, best_dph + 2 / nbin

    shift, res = brentq(func_to_minimize, a, b, full_output=True)

    nmax = ngood
    good = slice(1, nmax)

    big_sum = np.sum(
        idx[good] ** 2 * amp[good] * np.cos(-pha[good] + 2 * np.pi * idx[good] * -shift)
    )

    b = np.sum(
        amp[good] * np.cos(-pha[good] + 2 * np.pi * idx[good] * -shift)
    ) / np.sum(S[good] ** 2)

    eshift = sigma ** 2 / (2 * b * big_sum)

    eb = sigma ** 2 / (2 * np.sum(S[good] ** 2))

    return FFTFITResult(b, np.sqrt(eb), normalize_phase_0d5(shift), np.sqrt(eshift))


def normalize_phase_0d5(phase):
    """Normalize phase between -0.5 and 0.5
    Examples
    --------
    >>> normalize_phase_0d5(0.5)
    0.5
    >>> normalize_phase_0d5(-0.5)
    0.5
    >>> normalize_phase_0d5(4.25)
    0.25
    >>> normalize_phase_0d5(-3.25)
    -0.25
    """
    while phase > 0.5:
        phase -= 1
    while phase <= -0.5:
        phase += 1
    return phase


def fftfit_basic(template, profile):
    n, seb, shift, eshift = fftfit(profile, template)
    return shift


class FullResult:
    pass


def fftfit_full(template, profile):
    r = fftfit(profile, template)
    ro = FullResult()
    ro.shift = r.mean_phase
    ro.uncertainty = r.std_phase
    return ro
