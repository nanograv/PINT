---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pint.profile
```

```python
template = np.zeros(256)
template[:16] = 1
plt.plot(np.linspace(0, 1, len(template), endpoint=False), template)
up_template = pint.profile.upsample(template, 16)
plt.plot(np.linspace(0, 1, len(up_template), endpoint=False), up_template)
plt.xlim(0, 1)
```

```python
template = np.diff(scipy.stats.vonmises(100).cdf(np.linspace(0, 2 * np.pi, 1024 + 1)))
plt.plot(np.linspace(0, 1, len(template), endpoint=False), template)
up_template = pint.profile.upsample(template, 16)
plt.plot(np.linspace(0, 1, len(up_template), endpoint=False), up_template)
plt.plot(
    np.linspace(0, 1, len(template), endpoint=False), pint.profile.shift(template, 0.25),
)
plt.xlim(0, 1)
```

```python
if False:
    template = np.diff(scipy.stats.vonmises(10).cdf(np.linspace(0, 2 * np.pi, 64 + 1)))
    profile = pint.profile.shift(template, 0.25)
else:
    template = np.random.randn(64)
    profile = np.random.randn(len(template))

upsample = 8
if len(template) != len(profile):
    raise ValueError(
        "Template is length %d but profile is length %d" % (len(template), len(profile))
    )
t_c = np.fft.rfft(template)
p_c = np.fft.rfft(profile)
ccf_c = np.zeros((len(template) * upsample) // 2 + 1, dtype=complex)
ccf_c[: len(t_c)] = t_c
ccf_c[: len(p_c)] *= np.conj(p_c)
ccf = np.fft.irfft(ccf_c)
x = np.argmax(ccf) / len(ccf)
l, r = x - 1 / len(ccf), x + 1 / len(ccf)

plt.figure()
xs = np.linspace(0, 1, len(ccf), endpoint=False)
plt.plot(xs, ccf)
plt.axvspan(r, l, alpha=0.2)
plt.axvline(x)


def gof(x):
    return -(ccf_c * np.exp(2.0j * np.pi * np.arange(len(ccf_c)) * x)).sum().real


plt.plot(xs, [-2 * gof(x) / len(xs) for x in xs])

plt.figure()
xs = np.linspace(x - 4 / len(ccf), x + 4 / len(ccf), 100)
plt.plot(xs, [gof(x) for x in xs])
plt.axvspan(l, r, alpha=0.2)
plt.axvline(x)
```

```python
template = pint.profile.upsample(
    np.diff(scipy.stats.vonmises(10).cdf(np.linspace(0, 2 * np.pi, 16 + 1))), 2,
)
for s in np.linspace(0, 1 / len(template), 33):
    profile = pint.profile.shift(template, s)
    print((s - pint.profile.fftfit_basic(template, profile)) * len(template))
```

```python
a = np.random.randn(256)
a_c = np.fft.rfft(a)
a_c[-1] = 0
a_c[0] = 0
xs = np.linspace(0, 1, len(a), endpoint=False)
a_m = (
    (a_c[:, None] * np.exp(2.0j * np.pi * xs[None, :] * np.arange(len(a_c))[:, None]))
    .sum(axis=0)
    .real
    * 2
    / len(a)
)
a_i = np.fft.irfft(a_c)
plt.plot(xs, a_m)
plt.plot(xs, a_i)
np.sqrt(np.mean((a_m - a_i) ** 2))
```

```python
c = np.zeros(6, dtype=complex)
c[-1] = 1
np.fft.irfft(c)
```

```python
r = np.random.randn(256)
r_c = np.fft.rfft(r)
r_1 = np.fft.irfft(np.conj(r_c))
plt.plot(r)
plt.plot(r_1[1::-1])
```

```python
n = 16
c = np.zeros(5, dtype=complex)
c[0] = 1
print(pint.profile.irfft_value(c, 0, n))
np.fft.irfft(c, n)
```

```python
n = 8
c = np.zeros(5, dtype=complex)
c[-1] = 1
print(pint.profile.irfft_value(c, 0, n))
np.fft.irfft(c, n)
```

```python
n = 16
c = np.zeros(5, dtype=complex)
c[-1] = 1
print(pint.profile.irfft_value(c, 0, n))
np.fft.irfft(c, n)
```

```python
a = np.ones(8)
a[::2] *= -1
pint.profile.shift(pint.profile.shift(a, 1 / 16), -1 / 16)
```

```python
s = 1 / 3
t = pint.profile.vonmises_profile(10, 16)
t_c = np.fft.rfft(t)
t_s_c = np.fft.rfft(pint.profile.shift(t, s))
ccf_c = np.conj(t_c) * t_s_c
ccf_c[-1] = 0
plt.plot(np.fft.irfft(ccf_c, 256))
```

```python
s = 1 / 8
kappa = 1.0
n = 4096
template = pint.profile.vonmises_profile(kappa, n)
profile = pint.profile.shift(template, s / n)
rs = pint.profile.fftfit_basic(template, profile)
print(s, rs * n)
upsample = 8

n_long = len(template) * upsample
t_c = np.fft.rfft(template)
p_c = np.fft.rfft(profile)
ccf_c = t_c.copy()
ccf_c *= np.conj(p_c)
ccf_c[0] = 0
ccf_c[-1] = 0
ccf = np.fft.irfft(ccf_c, n_long)
i = np.argmax(ccf)
assert ccf[i] >= ccf[(i - 1) % len(ccf)]
assert ccf[i] >= ccf[(i + 1) % len(ccf)]
x = i / len(ccf)
l, r = x - 1 / len(ccf), x + 1 / len(ccf)


def gof(x):
    return -pint.profile.irfft_value(ccf_c, x, n_long)


print(l, gof(l))
print(x, gof(x))
print(r, gof(r))
print(-s / n, gof(-s / n))

res = scipy.optimize.minimize_scalar(gof, bracket=(l, x, r), method="brent", tol=1e-10)
res
```

```python
t = pint.profile.vonmises_profile(10, 1024, 1 / 3)
plt.plot(np.linspace(0, 1, len(t), endpoint=False), t)
plt.xlim(0, 1)
```

```python
profile1 = pint.profile.vonmises_profile(1, 512, phase=0.3)
profile2 = pint.profile.vonmises_profile(10, 1024, phase=0.7)
s = pint.profile.fftfit_basic(profile1, profile2)
pint.profile.fftfit_basic(pint.profile.shift(profile1, s), profile2)
```

Okay, so let's try to work out the uncertainties on the outputs.

Let's view the problem as this: we have a set of Fourier coefficients $t_j$ for the template and a set of Fourier coefficients $p_j$ for the profile. We are looking for $a$ and $\phi$ that minimize

$$ \chi^2 =  \sum_{j=1}^m \left|ae^{2\pi i j \phi} t_j - p_j\right|^2. $$

Put another way we have a vector-valued function $F(a,\phi)$ and we are trying to match the observed profile vector. We can estimate the uncertainties using the Jacobian of $F$.

$$\frac{\partial F}{\partial a}_j = e^{2\pi i j \phi} t_j, $$

and

$$\frac{\partial F}{\partial \phi}_j = a 2\pi i j e^{2\pi i j \phi} t_j. $$

If this forms a matrix $J$, and the uncertainties on the input data are of size $\sigma$, then the covariance matrix for the fit parameters will be $\sigma^2(J^TJ)^{-1}$.


```python
n = 8

r = []
for i in range(10000):
    t = np.random.randn(n)
    t_c = np.fft.rfft(t)

    r.append(np.mean(np.abs(t_c[1:-1]) ** 2) / (n * np.mean(np.abs(t) ** 2)))
np.mean(r)
```

```python
template = pint.profile.vonmises_profile(1, 256)
plt.plot(template)
plt.xlim(0, len(template))
std = 1
shift = 0
scale = 1
r = pint.profile.fftfit_aarchiba.fftfit_full(template, scale * pint.profile.shift(template, shift), std=std)
r.shift, r.scale, r.offset, r.uncertainty, r.cov
```

```python
fftfit.fftfit_full?
```

```python
def gen_shift():
    return pint.profile.wrap(
        pint.profile.fftfit_basic(
            template, scale * template + std * np.random.randn(len(template))
        )
    )


shifts = []
```

```python
for i in range(1000):
    shifts.append(gen_shift())
np.std(shifts)
```

```python
r.uncertainty / np.std(shifts)
```

```python
snrs = {}

scale = 1e-3
template = pint.profile.vonmises_profile(100, 1024, 1 / 3) + 0.5*pint.profile.vonmises_profile(50, 1024, 1 / 2)
plt.plot(np.linspace(0, 1, len(template), endpoint=False), scale*template)

def gen_prof(std):
    shift = np.random.uniform(0, 1)
    shift_template = pint.profile.shift(template, shift)
    return scale*shift_template + scale*std * np.random.standard_normal(len(template)) / np.sqrt(
        len(template)
    )


plt.plot(np.linspace(0, 1, len(template), endpoint=False), gen_prof(0.01))
plt.xlim(0, 1)


def gen_shift(std):
    shift = np.random.uniform(0, 1)
    shift_template = pint.profile.shift(template, shift)
    profile = scale*shift_template + scale*std * np.random.standard_normal(len(template)) / np.sqrt(
        len(template)
    )
    return pint.profile.wrap(pint.profile.fftfit_basic(template, profile) - shift)


gen_shift(0.01)
```

```python
def gen_uncert(std):
    return pint.profile.fftfit_aarchiba.fftfit_full(template, gen_prof(std), std=scale*std/np.sqrt(len(template))).uncertainty

def gen_uncert_estimate(std):
    return pint.profile.fftfit_full(template, gen_prof(std)).uncertainty
```

```python
for s in np.geomspace(1, 1e-4, 9):
    if s not in snrs:
        snrs[s] = []
    for i in range(1000):
        snrs[s].append(gen_shift(s))
```

```python
snr_list = sorted(snrs.keys())
plt.loglog(snr_list, [np.std(snrs[s]) for s in snr_list], "o", label="measured std.")
plt.loglog(snr_list, [gen_uncert(s) for s in snr_list], ".", label="computed uncert.")
plt.loglog(snr_list, [gen_uncert_estimate(s) for s in snr_list], "+", label="computed uncert. w/estimate")
plt.legend()
plt.xlabel("SNR (some strange units)")
plt.ylabel("Uncertainty or standard deviation (phase)")

```

```python
p = 1-2*scipy.stats.norm.sf(1)
p
```

```python
scipy.stats.binom.isf(0.01, 100, p), scipy.stats.binom.isf(0.99, 100, p)
```

```python
scipy.stats.binom(16, p).ppf(0.99)
```

```python

```
