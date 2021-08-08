# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload

# %autoreload 2

# %%
import functools

import astropy.constants as c
import astropy.units as u
import numpy as np
import pint
import sympy

from numdifftools import Jacobian


# %% [markdown]
# ## A better parametrization for the Shapiro delay
#
# Clean millisecond pulsars with small eccentricities have approximately-sinusoidal timing residuals. These deviate from a sinusoid due to the eccentricity of the pulsar (often small), but depending on the companion's mass and the orbit's orientation (and on whether general relativity is correct) they can also deviate from a sinusoid as a result of the Shapiro delay - pulses travel from the pulsar to us along a curved path, bent by the companion's gravity, and as a result arrive later than would be expected for straight-line propagation. This post-Newtonian effect can break the degeneracies in the Roemer delay, so measuring the Shapiro delay is valuable. It is conventionally parametrized with "range" ($r$) and "shape" ($s$) parameters, or with the companion mass ($m_2$) and the sine of the inclination angle ($\sin i$). Unfortunately these parametrizations are not very directly related to measurable quantities - pulse arrival times - and as a result fitting for them can be cumbersome.
#
# Freire and Wex 2010 (https://ui.adsabs.harvard.edu/abs/2010MNRAS.409..199F/abstract) introduced a new parametrization (several in fact) that connect better to observatble quantities and that are more reliable fitting parameters. These are $h_3$ and $h_4$ or $\varsigma$ (`VARSIGMA` or `STIGMA` in timing programs; the LaTeX code is `\varsigma`).
#
# Although Freire and Wex introduce these specifically for small-eccentricity orbits, and although they demonstrate their effectiveness only for small-eccentricity orbits, PINT implements these for all binary models, whether the eccentricity is large or small.

# %% [markdown]
# ## Working through the math
#
# Without approximation, if we define $\phi$ to be the true anomaly and $\omega$ to be the longitude of periastron relative to the ascending node, the Shapiro delay is given by
# $$
# \Delta_S = 2r\ln\left(\frac{1+e\cos\phi}{1-s\sin(\omega+\phi)}\right).
# $$
# For most theories of gravity, $s=\sin i$; and if GR is correct, $r = T_\odot m_c$ where $T_\odot$ is the mass of the Sun in time units and $m_c$ is the companion mass in solar masses. Most variant theories of gravity enter into this simply by changing the degree to which light is bent by a given amount of mass, so they only rescale $r$.
#
# For low-eccentricity orbits, the approximations simplify the situation. Start by defining $\Phi$ to be the orbital phase, in this approximation just $(T-T_{asc})/P_b$. Then ignoring terms of order eccentricity ($e$) squared, we can write the Shapiro delay as
# $$ \Delta_S = -2r\ln(1-s\sin\Phi). $$
#
# Expanding the logarithm part of this as a Fourier series in $\Phi$, we see that the first coefficients can be expressed fairly simply in terms of
# $$\varsigma = \frac{s}{1+\sqrt{1-s^2}}.$$
#
# While the first two harmonics in the Fourier series are indistinguishable from eccentricity (if no others can be detected), for small eccentricity the higher harmonics come only from Shapiro delay. Thus parametrizing these describes the measurable part of the Shapiro delay more directly.
#
# The parameter $h_3$ measures the overall amplitude of this measurable part, and is
# $$
# h_3 = r\varsigma^3.
# $$
# Freire and Wex point out that $h_3$ is very well-behaved, observationally, since if higher harmonics cannot be detected it is proportional to the third harmonic - if it cannot be detected then it is simply consistent with zero.
#
# The parameter $h_4$ measures the fourth harmonic, and is defined by
# $$
# h_4 = h_3\varsigma.
# $$
# For moderate inclinations, if not much information about harmonics five and higher is available, then this is uncorrelated with $h_3$ and those make a good parametrization of the Shapiro delay.
#
# If the inclination is high - the system is nearly edge-on - information about the higher harmonics will allow the ratio $h_3/h_4$ to be measured more accurately than either $h_3$ or $h_4$ and they will become correlated; there Freire and Wex recommend using $h_3$ and $\varsigma$.
#
# A few convenient expressions for converting the other way:
# $$
# s = \frac{2h_3h_4}{h_3^2+h_4^2}
# $$
# and
# $$
# r = \frac{h_3^4}{h_4^3}.
# $$
#

# %%
def h3h4_from_rs(r, s):
    c = np.sqrt(1 - s ** 2)
    varsigma = s / (1 + c)
    h3 = r * varsigma ** 3
    h4 = h3 * c
    return h3, h4


def rs_from_h3h4(h3, h4):
    s = 2 * h3 * h4 / (h3 ** 2 + h4 ** 2)
    r = h3 ** 4 / h4 ** 3
    return r, s


def h3varsigma_from_rs(r, s):
    c = np.sqrt(1 - s ** 2)
    varsigma = s / (1 + c)
    h3 = r * varsigma ** 3
    return h3, varsigma


def rs_from_h3varsigma(h3, varsigma):
    s = 2 * varsigma / (1 + varsigma ** 2)
    r = h3 / r ** 3
    return r, s


def h3h4_from_h3varsigma(h3, varsigma):
    h4 = h3 * varsigma
    return h3, h4


def h3varsigma_from_h3h4(h3, h4):
    varsigma = h3 / h4
    return h3, varsigma


# %%

# %%
def numpy_1dify(f):
    @functools.wraps
    def wrapper(*args, **kwargs):
        if not args:
            raise ValueError(
                "Must have at least one positional argument to make into an array"
            )
        bargs = np.broadcast_arrays(*args)
        if not bargs[0].shape():
            bargs = [np.array([b]) for b in bargs]


# %%
a = np.broadcast_arrays(0, 1, 2)
a

# %%
a[0].shape

# %%

# %%
pint.Tsun.to(u.us)

# %%
s = 0.0001
c = np.sqrt(1 - s ** 2)
varsigma = s / (1 + c)
2 * varsigma / (1 + varsigma ** 2)

# %%
varsigma = sympy.var("varsigma")
s = sympy.var("s")

# %%
d = sympy.diff(2 * varsigma / (1 + varsigma ** 2), varsigma)
d

# %%
(d - 2 * (1 - varsigma ** 2) / (1 + varsigma ** 2) ** 2).simplify()

# %%
sympy.diff(s / (1 + (1 - s ** 2) ** (1 / 2)), s).simplify()


# %%
def f(xy):
    x, y = xy
    return np.array([x + y ** 2, y])


Jacobian(f)(np.array([1, 2]))

# %% [markdown]
# If we have $a,b = f(x,y)$, then `numdifftools.Jacobian` produces
# $$
# \begin{bmatrix}
# \frac{\partial a}{\partial x} & \frac{\partial a}{\partial y} \\
# \frac{\partial b}{\partial x} & \frac{\partial b}{\partial y}
# \end{bmatrix}
# $$
# so let's use that orientation for our Jacobians.

# %%
sympy.diff(varsigma ** (-3), varsigma)

# %%
m = np.array([[1, 2], [3, 4]])
x = np.array([5, 6])
m @ x

# %%
Jacobian(lambda x: m @ x)(np.zeros(2))

# %%
(np.finfo(np.longdouble).eps * 50000 * u.day).to(u.ns)

# %%
