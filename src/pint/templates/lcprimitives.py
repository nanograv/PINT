"""Components of a pulsar light curve.

LCPrimitive and its subclasses implement
components of a pulsar light curve.  Includes primitives (Gaussian,
Lorentzian), etc.  as well as more sophisticated holistic templates that
provide single-parameter (location) representations of the light curve.

author: M. Kerr <matthew.kerr@gmail.com>

"""

# NB -- big TODO -- I don't think wrapped primitives quite correctly return
# Monte Carlo variables because they don't account for the uniform approx.
# perhaps this isn't a big deal

from math import atan, cos, cosh, sin, sinh, tan

import numpy as np
from scipy.integrate import quad, simps
from scipy.interpolate import interp1d
from scipy.special import erf, i0, i1
from scipy.stats import cauchy, norm

ROOT2PI = (2 * np.pi) ** 0.5
R2DI = (2 / np.pi) ** 0.5
ROOT2 = 2 ** 0.5
TWOPI = 2 * np.pi
PI = np.pi * 1
MAXWRAPS = 15
MINWRAPS = 3
WRAPEPS = 1e-8


def two_comp_mc(n, w1, w2, loc, func):
    """Generate MC photons from a two-sided distribution.

    Helper function.  This should work as is if w1,w2,loc are vectors.

    Parameters
    ----------
    n : int
        total number of photons
    w1 : float or array-like
        scale parameter for func, lefthand peak
    w2 : float or array-like
        scale parameter for func, righthand peak
    loc : float or array-like
        position of peak
    func : callable
        an 'rvs' function from scipy
    """
    frac1 = w1 / (w1 + w2)
    # number of photons required from left side
    n1 = (np.random.rand(n) < frac1).sum()
    r1 = func(loc=0, scale=w1, size=n1)
    # reflect and relocate photons to right or lef side
    r1 = loc + np.where(r1 <= 0, r1, -r1)
    r2 = func(loc=0, scale=w2, size=n - n1)
    r2 = loc + np.where(r2 > 0, r2, -r2)
    return np.mod(np.append(r1, r2), 1)


def approx_gradient(func, phases, log10_ens=None, eps=1e-6):
    """ Return a numerical gradient.  This works for both LCPrimitive and
        LCTemplate objects.  HOW AWESOME!
    """
    orig_p = func.get_parameters(free=False).copy()
    g = np.zeros([len(orig_p), len(phases)])
    weights = np.asarray([-1, 8, -8, 1]) / (12 * eps)

    def do_step(which, eps):
        p0 = orig_p.copy()
        p0[which] += eps
        func.set_parameters(p0, free=False)
        return func(phases, log10_ens)

    for i in range(len(orig_p)):
        # use a 4th-order central difference scheme
        for step, w in zip([2, 1, -1, -2], weights):
            g[i, :] += w * do_step(i, step * eps)

    func.set_parameters(orig_p, free=False)
    return g


def approx_hessian(func, phases, log10_ens=None, eps=1e-4):
    """ Return a numerical hessian.  This works for both LCPrimitive and
        LCTemplate objects.  HOW AWESOME!
    """
    orig_p = func.get_parameters(free=False).copy()
    g = np.zeros([len(orig_p), len(orig_p), len(phases)])
    weights = np.asarray([1, -1, -1, 1]) / (4 * eps ** 2)

    def do_step(which, eps):
        which1, which2 = which
        eps1, eps2 = eps
        p0 = orig_p.copy()
        p0[which1] += eps1
        p0[which2] += eps2
        func.set_parameters(p0, free=False)
        return func(phases)

    steps = np.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1]]) * eps
    for i in range(len(orig_p)):
        for j in range(len(orig_p)):
            # use a 2th-order central difference scheme
            for weight, step in zip(weights, steps):
                g[i, j, :] += weight * do_step((i, j), step)

    func.set_parameters(orig_p, free=False)
    return g


def check_gradient(func, atol=1e-8, rtol=1e-5, quiet=False):
    """ Test gradient function with a set of MC photons.
        This works with either LCPrimitive or LCTemplate objects.

        TODO -- there is trouble with the numerical gradient when
        a for the location-related parameters when the finite step
        causes the peak to shift from one side of an evaluation phase
        to the other."""
    en = np.random.rand(1000) * 2 + 1  # 100 MeV to 10 GeV
    ph = func.random(en)
    if hasattr(func, "closest_to_peak"):
        eps = min(1e-6, 0.2 * func.closest_to_peak(ph))
    else:
        eps = 1e-6
    g1 = func.gradient(ph, en, free=False)
    g2 = func.approx_gradient(ph, en, eps=eps)
    anyfail = False
    for i in range(g1.shape[0]):
        d1 = np.abs(g1[i] - g2[i])
        a = np.argmax(d1)
        fail = np.any(d1 > (atol + rtol * np.abs(g2)))
        if not quiet:
            pass_string = "FAILED" if fail else "passed"
            print("%02d (%s) %.3g (abs)" % (i, pass_string, d1.max()))
        anyfail = anyfail or fail
    return not anyfail


class LCPrimitive:
    """ Base class for various components of a light curve.  All "analytic"
        light curve models must inherit and must implement the three
        'virtual' functions below."""

    def is_energy_dependent(self):
        return False

    def is_two_sided(self):
        """ True if primitive is asymmetric.  Default is False, two-sided
            child classes should override."""
        return False

    def copy(self):
        from copy import deepcopy

        return deepcopy(self)

    def __call__(self, phases):
        raise NotImplementedError(
            "Virtual function must be implemented by child class."
        )

    def integrate(self, x1=0, x2=1, log10_ens=3):
        """ Base implemention with scipy quad."""
        f = lambda ph: self(ph, log10_ens)
        return quad(f, x1, x2)[0]

    def cdf(self, x, log10_ens=3):
        return self.integrate(x1=0, x2=x, log10_ens=3)

    def fwhm(self):
        """Return the full-width at half-maximum of the light curve model."""
        return self.hwhm(0) + self.hwhm(1)

    def hwhm(self, right=False):
        """Return the half-width at half-maximum of the light curve model."""
        raise NotImplementedError(
            "Virtual function must be implemented by child class."
        )

    def init(self):
        self.p = np.asarray([1])
        self.pnames = []
        self.name = "Default"
        self.shortname = "None"

    def _asarrays(self):
        for key in ["p", "free", "bounds", "errors", "slope", "slope_free"]:
            if hasattr(self, key):
                v = self.__dict__[key]
                if v is not None:
                    self.__dict__[key] = np.asarray(
                        v, dtype=bool if "free" in key else float
                    )

    def _default_bounds(self):
        bounds = [[]] * len(self.p)
        # this order works for LCHarmonic, too
        bounds[0] = [0.005, 0.5]  # width
        bounds[-1] = [-1, 1]  # position
        if len(bounds) > 2:
            bounds[1] = [0.005, 0.5]  # width
        return bounds

    def _default_priors(self):
        loc = self.p.copy()
        width = np.asarray([0.1] * len(self.p))
        enable = np.asarray([False] * len(self.p))
        return loc, width, enable

    def __init__(self, **kwargs):
        """ Generally, class-specific setup work is performed in init.
            Here, init is called and certain guaranteed default members
            are established."""
        self.init()
        if not hasattr(self, "bounds"):
            self.bounds = self._default_bounds()  # default
        self.errors = np.zeros_like(self.p)
        self.free = np.asarray([True] * len(self.p))
        self.__dict__.update(kwargs)
        self._asarrays()
        (
            self.gauss_prior_loc,
            self.gauss_prior_width,
            self.gauss_prior_enable,
        ) = self._default_priors()
        self.shift_mode = False

    def _make_p(self, log10_ens=3):
        """ Internal method to return parameters appropriate for use
            in functional form."""
        return [None] + list(self.p)

    def set_parameters(self, p, free=True):
        if free:
            self.p[self.free] = p
        else:
            self.p[:] = p
        # adjust position to be between 0 and 1
        self.p[-1] = self.p[-1] % 1
        return np.all(self.p >= 0)

    def get_parameters(self, free=True):
        if free:
            return self.p[self.free]
        return self.p

    def get_parameter_names(self, free=True):
        return [p for (p, b) in zip(self.pnames, self.free) if b]

    def set_errors(self, errs):
        n = self.free.sum()
        self.errors[:] = 0.0
        self.errors[self.free] = errs[:n]
        return n

    def get_errors(self, free=True):
        return self.errors[self.free]

    def get_bounds(self):
        return self.bounds[self.free]

    def get_gauss_prior_parameters(self):
        mod_array = [False] * (len(self.p) - 1) + [True]
        return (
            self.gauss_prior_loc[self.free],
            self.gauss_prior_width[self.free],
            np.asarray(mod_array)[self.free],
            self.gauss_prior_enable[self.free],
        )

    def enable_gauss_prior(self, enable=True):
        """ [Convenience] Turn on gaussian prior."""
        self.gauss_prior_enable[:] = enable

    def center_gauss_prior(self, enable=False):
        """ [Convenience] Set gauss mode to current params."""
        self.gauss_prior_loc[:] = self.p[:]
        if enable:
            self.enable_gauss_prior()

    def get_location(self, error=False):
        if error:
            return np.asarray([self.p[-1], self.errors[-1]])
        return self.p[-1]

    def set_location(self, loc):
        self.p[-1] = loc

    def get_norm(self, error=False):
        # if error: return np.asarray([self.p[0],self.errors[0]])
        # return self.p[0]
        return 1

    def get_width(self, error=False, hwhm=False, right=False):
        """Return the width of the distribution.

        Parameters
        ----------
        error : bool
            if True, return tuple with value and error
        hwhm : bool
            if True, scale width to be HWHM
        right : bool
            if True, return "right" component, else "left".
            There is no distinction for symmetric dists.
        """
        scale = self.hwhm(right=right) / self.p[int(right)] if hwhm else 1
        if error:
            return np.asarray([self.p[int(right)], self.errors[int(right)]]) * scale
        return self.p[int(right)] * scale

    def get_gradient(self, phases, log10_ens=3):
        raise DeprecationWarning()
        return self.gradient(phases, log10_ens, free=True)

    def gradient(self, phases, log10_ens=3, free=False):
        """ Return the gradient of the primitives wrt the parameters.
        """
        raise NotImplementedError("No gradient function found for this object.")

    def gradient_derivative(self, phases, log10_ens=3, free=False):
        """ Return d/dphi(gradient).  This is needed for computing the
            hessian of the profile for parameters that affect the timing
            model and hence pulse phase.
        """
        raise NotImplementedError(
            "No gradient_derivative function found for this object."
        )

    def derivative(self, phases, log10_ens=3, order=1):
        """ Return d^np(phi)/dphi^n, with n=order."""
        raise NotImplementedError("No derivative function found for this object.")

    def random(self, n):
        """ Default is accept/reject."""
        if n < 1:
            return 0
        M = self(np.asarray([self.p[-1]]))  # peak amplitude
        rvals = np.empty(n)
        position = 0
        rfunc = np.random.rand
        while True:
            cand_phases = rfunc(n)
            cand_phases = cand_phases[rfunc(n) < self(cand_phases) / M]
            ncands = len(cand_phases)
            if ncands == 0:
                continue
            rvals[position : position + ncands] = cand_phases[: n - position]
            position += ncands
            if position >= n:
                break
        return rvals

    def __str__(self):
        m = max([len(n) for n in self.pnames])
        l = []
        errors = self.errors if hasattr(self, "errors") else [0] * len(self.pnames)
        for i in range(len(self.pnames)):
            fstring = "" if self.free[i] else " [FIXED]"
            n = self.pnames[i][:m]
            t_n = n + (m - len(n)) * " "
            l += [t_n + ": %.4f +\\- %.4f%s" % (self.p[i], errors[i], fstring)]
        l = [self.name + "\n------------------"] + l
        return "\n".join(l)

    def approx_gradient(self, phases, log10_ens=3, eps=1e-5):
        return approx_gradient(self, phases, log10_ens, eps=eps)

    def check_gradient(self, atol=1e-8, rtol=1e-5, quiet=False):
        return check_gradient(self, atol=atol, rtol=rtol, quiet=quiet)

    def sanity_checks(self, eps=1e-6):
        """ A few checks on normalization, integration, etc. """
        errfac = 1
        # Normalization test
        y, ye = quad(self, 0, 1)
        # t1 = abs(self.p[0]-y)<(ye*errfac)
        t1 = abs(1 - y) < (ye * errfac)
        # integrate method test
        # t2 = abs(self.p[0]-self.integrate(0,1))<eps
        t2 = abs(1 - self.integrate(0, 1)) < eps
        # FWHM test
        t3 = (self(self.p[-1]) * 0.5 - self(self.p[-1] - self.fwhm() / 2)) < eps
        # gradient test
        try:
            t4 = self.check_gradient(quiet=True)
        except:
            t4 = False
        # boundary conditions
        t5 = abs(self(0) - self(1 - eps)) < eps
        if not t1:
            print("Failed Normalization test")
        if not t2:
            print("Failed integrate method test")
        if not t3:
            print("Failed FWHM test")
        if not t4:
            print("Failed gradient test")
        if not t5:
            print("Did not pass boundary conditions")
        return np.all([t1, t2, t3, t4, t5])

    def eval_string(self):
        """ Return a string that can be evaled to instantiate a nearly-
            identical object."""
        return "%s(p=%s,free=%s,slope=%s,slope_free=%s)" % (
            self.__class__.__name__,
            str(list(self.p)),
            str(list(self.free)),
            str(list(self.slope)) if hasattr(self, "slope") else None,
            str(list(self.slope_free)) if hasattr(self, "slope_free") else None,
        )

    def dict_string(self):
        """ Return a string to express the object as a dictionary that can
            be easily instantiated using its keys."""

        def pretty_list(l, places=5):
            fmt = "%." + "%d" % places + "f"
            s = ", ".join([fmt % x for x in l])
            return "[" + s + "]"

        t = [
            "name = %s" % self.__class__.__name__,
            "p = %s" % (pretty_list(self.p)),
            "free = %s" % (str(list(self.free))),
            "slope = %s"
            % (pretty_list(self.slope) if hasattr(self, "slope") else None),
            "slope_free = %s"
            % (str(list(self.slope_free)) if hasattr(self, "slope_free") else None),
        ]
        # return 'dict(\n'+'\n    '.join(t)+'\n
        return t

    def closest_to_peak(self, phases):
        """ Return the minimum distance between a member of the array of
            phases and the position of the mode of the primitive."""
        return np.abs(phases - self.get_location()).min()

    def get_fixed_energy_version(self, log10_en=3):
        return self


class LCWrappedFunction(LCPrimitive):
    """ Super-class for profiles derived from wrapped functions.

        While some distributions (e.g. the wrapped normal) converge
        quickly, others (e.g. the wrapped Lorentzian) converge very slowly
        and must be truncated before machine precision is reached.

        In order to preserve normalization, the pdf is slightly adjusted:
        f(phi) = sum_(i,-N,N,g(phi+i)) + (1 - int(phi,-N,N,g(phi)) ).

        This introduces an additional parameteric dependence which must
        be accounted for by computation of the gradient.
    """

    def _norm(self, nwraps, log10_ens=3):
        """ Compute the truncated portion of the template."""
        # return self.p[0]-self.base_int(-nwraps,nwraps+1)
        return 1 - self.base_int(-nwraps, nwraps + 1, log10_ens)

    def _grad_norm(self, nwraps, log10_ens=3):
        """ Compute the gradient terms due to truncated portion.  That is,
            since we add on a uniform component beyond nwraps, the
            amplitude of this component depends on the CDF and hence on
            the parameters.

            Default implementation is to ignore these terms, applicable
            for rapidly-converging distributions (e.g. wrapped normal with
            small width parameter).  On the other hand, it is not
            negligible for long-tailed distributions, e.g. Lorentzians."""
        return None

    def _grad_deriv_norm(self, nwraps, log10_ens=3):
        return None

    def _grad_hess(self, nwraps, log10_ens=3):
        """ Compute the hessian terms due to truncated portion.
            See _grad_norm.
        """
        return None

    def __call__(self, phases, log10_ens=3):
        """ Return wrapped template + DC component corresponding to truncation."""
        results = self.base_func(phases, log10_ens)
        for i in range(1, MAXWRAPS + 1):
            t = self.base_func(phases, log10_ens, index=i)
            t += self.base_func(phases, log10_ens, index=-i)
            results += t
            if (i >= MINWRAPS) and (np.all(t < WRAPEPS)):
                break
        return results + self._norm(i, log10_ens)

    def gradient(self, phases, log10_ens=3, free=False):
        """ Return the gradient evaluated at a vector of phases.

            output : a num_parameter x len(phases) ndarray,
                     the num_parameter-dim gradient at each phase
        """
        results = self.base_grad(phases, log10_ens)
        for i in range(1, MAXWRAPS + 1):
            t = self.base_grad(phases, log10_ens, index=i)
            t += self.base_grad(phases, log10_ens, index=-i)
            results += t
            if (i >= MINWRAPS) and (np.all(t < WRAPEPS)):
                break
        gn = self._grad_norm(i, log10_ens)
        if gn is not None:
            for i in range(len(gn)):
                results[i, :] += gn[i]
        if free:
            return results[self.free]
        return results

    def gradient_derivative(self, phases, log10_ens=3, free=False):
        """ Return the gradient evaluated at a vector of phases.

            output : a num_parameter x len(phases) ndarray,
                     the num_parameter-dim gradient at each phase
        """
        results = self.base_grad_deriv(phases, log10_ens)
        for i in range(1, MAXWRAPS + 1):
            t = self.base_grad_deriv(phases, log10_ens, index=i)
            t += self.base_grad_deriv(phases, log10_ens, index=-i)
            results += t
            if (i >= MINWRAPS) and (np.all(t < WRAPEPS)):
                break
        gn = self._grad_deriv_norm(i, log10_ens)
        if gn is not None:
            for i in range(len(gn)):
                results[i, :] += gn[i]
        if free:
            return results[self.free]
        return results

    def hessian(self, phases, log10_ens=3, free=False):
        """ Return the hessian evaluated at a vector of phases.

            NB that this is restricted to the sub-space for this primitive.

            output : a num_parameter x num_parameter x len(phases) ndarray,
                     the num_parameter-dim^2 hessian at each phase
        """
        results = self.base_hess(phases, log10_ens)
        for i in range(1, MAXWRAPS + 1):
            t = self.base_hess(phases, log10_ens, index=i)
            t += self.base_hess(phases, log10_ens, index=-i)
            results += t
            if (i >= MINWRAPS) and (np.all(t < WRAPEPS)):
                break
        gh = self._grad_hess(i, log10_ens)
        if gh is not None:
            raise NotImplementedError
            # for i in range(len(gn)):
            # results[i,:] += gn[i]
        if free:
            return results[self.free, self.free]
        return results

    def derivative(self, phases, log10_ens=3, order=1):
        """ Return the phase gradient (dprim/dphi) at a vector of phases.

            order: order of derivative (1=1st derivative, etc.)

            output : a len(phases) ndarray, dprim/dphi

            NB this will generally be opposite in sign to the gradient of
            the location parameter.
        """
        results = self.base_derivative(phases, log10_ens, order=order)
        for i in range(1, MAXWRAPS + 1):
            t = self.base_derivative(phases, log10_ens, index=i, order=order)
            t += self.base_derivative(phases, log10_ens, index=-i, order=order)
            results += t
            if (i >= MINWRAPS) and (np.all(t < WRAPEPS)):
                break
        return results

    def integrate(self, x1, x2, log10_ens=3):
        # if(x1==0) and (x2==0): return 1.
        # NB -- this method is probably overkill now.
        results = self.base_int(x1, x2, log10_ens, index=0)
        for i in range(1, MAXWRAPS + 1):
            t = self.base_int(x1, x2, log10_ens, index=i)
            t += self.base_int(x1, x2, log10_ens, index=-i)
            results += t
            if np.all(t < WRAPEPS):
                break
        return results + (x2 - x1) * self._norm(i, log10_ens)

    def base_func(self, phases, log10_ens=3, index=0):
        raise NotImplementedError("No base_func function found for this object.")

    def base_grad(self, phases, log10_ens=3, index=0):
        raise NotImplementedError("No base_grad function found for this object.")

    def base_grad_deriv(self, phases, log10_ens=3, index=0):
        raise NotImplementedError("No base_grad_deriv function found for this object.")

    def base_hess(self, phases, log10_ens=3, index=0):
        raise NotImplementedError("No base_hess function found for this object.")

    def base_derivative(self, phases, log10_ens=3, index=0, order=1):
        raise NotImplementedError("No base_derivative function found for this object.")

    def base_int(self, phases, log10_ens=3, index=0):
        raise NotImplementedError("No base_int function found for this object.")


class LCGaussian(LCWrappedFunction):
    """ Represent a (wrapped) Gaussian peak.

        Parameters
        Width     the standard deviation parameter of the norm dist.
        Location  the mode of the Gaussian distribution
    """

    def init(self):
        self.p = np.asarray([0.03, 0.5])
        self.pnames = ["Width", "Location"]
        self.name = "Gaussian"
        self.shortname = "G"

    def hwhm(self, right=False):
        return self.p[0] * (2 * np.log(2)) ** 0.5

    def base_func(self, phases, log10_ens=3, index=0):
        e, width, x0 = self._make_p(log10_ens)
        z = (phases + index - x0) / width
        return (1.0 / (width * ROOT2PI)) * np.exp(-0.5 * z ** 2)

    def base_grad(self, phases, log10_ens=3, index=0):
        e, width, x0 = self._make_p(log10_ens)
        z = (phases + index - x0) / width
        f = (1.0 / (width * ROOT2PI)) * np.exp(-0.5 * z ** 2)
        return np.asarray([f / width * (z ** 2 - 1.0), f / width * z])

    def base_grad_deriv(self, phases, log10_ens=3, index=0):
        e, width, x0 = self._make_p(log10_ens)
        z = (phases + index - x0) / width
        f = (1.0 / (width * ROOT2PI)) * np.exp(-0.5 * z ** 2)
        q = f / width ** 2
        z2 = z ** 2
        return np.asarray([q * z * (3 - z2), q * (1 - z2)])

    def base_hess(self, phases, log10_ens=3, index=0):
        e, width, x0 = self._make_p(log10_ens)
        z = (phases + index - x0) / width
        f = (1.0 / (width * ROOT2PI)) * np.exp(-0.5 * z ** 2)
        q = f / width ** 2
        z2 = z ** 2
        rvals = np.empty((2, 2, len(z)))
        rvals[0, 0] = q * (z2 ** 2 - 5 * z2 + 2)
        rvals[0, 1] = q * (z2 - 3) * z
        rvals[1, 1] = q * (z2 - 1)
        rvals[1, 0] = rvals[0, 1]
        return rvals

    def base_derivative(self, phases, log10_ens=3, index=0, order=1):
        e, width, x0 = self._make_p(log10_ens)
        z = (phases + index - x0) / width
        f = (1.0 / (width * ROOT2PI)) * np.exp(-0.5 * z ** 2)
        if order == 1:
            return f / (-width) * z
        elif order == 2:
            return f / width ** 2 * (z ** 2 - 1)
        else:
            raise NotImplementedError

    def base_int(self, x1, x2, log10_ens=3, index=0):
        e, width, x0 = self._make_p(log10_ens)
        z1 = (x1 + index - x0) / width
        z2 = (x2 + index - x0) / width
        return 0.5 * (erf(z2 / ROOT2) - erf(z1 / ROOT2))

    def random(self, n):
        if hasattr(n, "__len__"):
            n = len(n)
        return np.mod(norm.rvs(loc=self.p[-1], scale=self.p[0], size=n), 1)


class LCGaussian2(LCWrappedFunction):
    """ Represent a (wrapped) two-sided Gaussian peak.

        Parameters
        Width1    the standard deviation parameter of the norm dist.
        Width2    the standard deviation parameter of the norm dist.
        Location  the mode of the distribution
    """

    def init(self):
        self.p = np.asarray([0.03, 0.03, 0.5])
        self.pnames = ["Width1", "Width2", "Location"]
        self.name = "Gaussian2"
        self.shortname = "G2"

    def is_two_sided(self):
        return True

    def hwhm(self, right=False):
        return (self.p[int(right)]) * (2 * np.log(2)) ** 0.5

    def base_func(self, phases, log10_ens=3, index=0):
        e, width1, width2, x0 = self._make_p(log10_ens)
        z = phases + (index - x0)
        z *= np.where(z <= 0, 1.0 / width1, 1.0 / width2)
        return (R2DI / (width1 + width2)) * np.exp(-0.5 * z ** 2)

    def base_grad(self, phases, log10_ens=3, index=0):
        e, width1, width2, x0 = self._make_p(log10_ens)
        z = phases + (index - x0)
        m = z <= 0
        w = np.where(m, width1, width2)
        z /= w
        f = (R2DI / (width1 + width2)) * np.exp(-0.5 * z ** 2)
        k = 1.0 / (width1 + width2)
        z2w = z ** 2 / w
        t = f * (z2w - k)
        g1 = f * (z2w * (m) - k)
        g2 = f * (z2w * (~m) - k)
        g3 = f * z / w
        return np.asarray([g1, g2, g3])

    def base_int(self, x1, x2, log10_ens=3, index=0):
        e, width1, width2, x0 = self._make_p(log10_ens)
        if index == 0 and (x1 < x0) and (x2 > x0):
            z1 = (x1 + index - x0) / width1
            z2 = (x2 + index - x0) / width2
            k1 = 2 * width1 / (width1 + width2)
            k2 = 2 * width2 / (width1 + width2)
            return 0.5 * (k2 * erf(z2 / ROOT2) - k1 * erf(z1 / ROOT2))
        w = width1 if ((x1 + index) < x0) else width2
        z1 = (x1 + index - x0) / w
        z2 = (x2 + index - x0) / w
        k = 2 * w / (width1 + width2)
        return 0.5 * k * (erf(z2 / ROOT2) - erf(z1 / ROOT2))

    def random(self, n):
        """ Use multinomial technique to return random photons from
            both components."""
        if hasattr(n, "__len__"):
            n = len(n)
        return two_comp_mc(n, self.p[0], self.p[1], self.p[-1], norm.rvs)


class LCLorentzian(LCPrimitive):
    """Represent a (wrapped) Lorentzian peak.

    Parameters
    ----------
    Width
        the width paramater of the wrapped Cauchy distribution, namely HWHM*2PI
        for narrow distributions
    Location
        the center of the peak in phase
    """

    def init(self):
        self.p = np.asarray([0.1, 0.5])
        self.pnames = ["Width", "Location"]
        self.name = "Lorentzian"
        self.shortname = "L"

    def hwhm(self, right=False):
        # NB -- bounds on p[1] set such that this is well-defined
        return np.arccos(2 - cosh(self.p[0])) / TWOPI

    def __call__(self, phases, log10_ens=3):
        e, gamma, loc = self._make_p(log10_ens)
        z = TWOPI * (phases - loc)
        # NB -- numpy call not as efficient as math.sinh etc.
        # but this allows easy inheritance for the energy-dependence
        return np.sinh(gamma) / (np.cosh(gamma) - np.cos(z))

    def gradient(self, phases, log10_ens=3, free=False):
        e, gamma, loc = self._make_p(log10_ens)
        z = TWOPI * (phases - loc)
        s1 = np.sinh(gamma)
        c1 = np.cosh(gamma)
        c = np.cos(z)
        s = np.sin(z)
        f = s1 / (c1 - c)
        f2 = f ** 2
        g1 = f * (c1 / s1) - f2
        g2 = f2 * (TWOPI / s1) * s
        if free:
            return np.asarray([g1, g2])[self.free]
        return np.asarray([g1, g2])

    def derivative(self, phases, log10_ens=3, index=0, order=1):
        """ Return the phase gradient (dprim/dphi) at a vector of phases.

            order: order of derivative (1=1st derivative, etc.)

            output : a len(phases) ndarray, dprim/dphi

            NB this will generally be opposite in sign to the gradient of
            the location parameter.
        """
        e, gamma, loc = self._make_p(log10_ens)
        z = TWOPI * (phases - loc)
        s1 = np.sinh(gamma)
        c1 = np.cosh(gamma)
        c = np.cos(z)
        s = np.sin(z)
        f = s1 / (c1 - c)
        f2 = f ** 2
        if order == 1:
            return (-TWOPI / s1) * (f ** 2 * s)
        elif order == 2:
            return (-(TWOPI ** 2) / s1) * f ** 2 * (c - 2 * f * s ** 2 / s1)
        else:
            raise NotImplementedError

    def random(self, n):
        if hasattr(n, "__len__"):
            n = len(n)
        return np.mod(cauchy.rvs(loc=self.p[-1], scale=self.p[0] / TWOPI, size=n), 1)

    def integrate(self, x1, x2, log10_ens=3):
        # NB -- due to the use of tans below, must be careful to use an angle
        # range of -pi/2 to pi/2 rather than 0 to pi as one would want
        # I haven't carefully tested this solution
        e, gamma, loc = self._make_p(log10_ens)
        x1 = PI * (x1 - loc)
        x2 = PI * (x2 - loc)
        t = 1.0 / np.tanh(0.5 * gamma)  # coth(gamma/2)
        v2 = np.arctan(t * tan(x2)) / PI
        v1 = np.arctan(t * tan(x1)) / PI
        return (v2 <= v1) + v2 - v1  # correction for tan wrapping


class LCLorentzian2(LCWrappedFunction):
    """ Represent a (wrapped) two-sided Lorentzian peak.
        Parameters
        Width1    the HWHM of the distribution (left)
        Width2    the HWHM of the distribution (right)
        Location  the mode of the distribution
    """

    def init(self):
        self.p = np.asarray([0.03, 0.03, 0.5])
        self.pnames = ["Width1", "Width2", "Location"]
        self.name = "Lorentzian2"
        self.shortname = "L2"

    def is_two_sided(self):
        return True

    def hwhm(self, right=False):
        return self.p[int(right)]

    def _grad_norm(self, nwraps, log10_ens=3):
        e, gamma1, gamma2, x0 = self._make_p(log10_ens)
        z1 = (-nwraps - x0) / gamma1
        z2 = (nwraps + 1 - x0) / gamma2
        t = gamma2 * np.arctan(z2) - gamma1 * np.arctan(z1)
        t1 = 1.0 / (1 + z1 ** 2)
        t2 = 1.0 / (1 + z2 ** 2)
        k = 2 / (gamma1 + gamma2) / PI
        f = k * t
        g1 = -1.0 / (gamma1 + gamma2) - (np.arctan(z1) - z1 * t1) / t
        g2 = -1.0 / (gamma1 + gamma2) + (np.arctan(z2) - z2 * t2) / t
        g3 = (t1 - t2) / t
        return [-f * g1, -f * g2, -f * g3]

    def base_func(self, phases, log10_ens=3, index=0):
        e, gamma1, gamma2, x0 = self._make_p(log10_ens)
        z = phases + (index - x0)
        z *= np.where(z <= 0, 1.0 / gamma1, 1.0 / gamma2)
        k = 2 / (gamma1 + gamma2) / PI
        return k / (1 + z ** 2)

    def base_grad(self, phases, log10_ens=3, index=0):
        e, gamma1, gamma2, x0 = self._make_p(log10_ens)
        z = phases + (index - x0)
        m = z < 0
        g = np.where(m, 1.0 / gamma1, 1.0 / gamma2)
        t1 = 1 + (z * g) ** 2
        t2 = 2 * (z * g) / t1
        g1 = -1 / (gamma1 + gamma2) + t2 * ((m * z) / gamma1 ** 2)
        g2 = -1 / (gamma1 + gamma2) + t2 * ((~m * z) / gamma2 ** 2)
        g3 = t2 * g
        f = (2.0 / (gamma1 + gamma2) / PI) / t1
        return np.asarray([f * g1, f * g2, f * g3])

    def base_derivative(self, phases, log10_ens=3, index=0, order=1):
        e, gamma1, gamma2, x0 = self._make_p(log10_ens)
        z = phases + (index - x0)
        g = np.where(z < 0, 1.0 / gamma1, 1.0 / gamma2)
        k = 2 / (gamma1 + gamma2) / PI
        z *= g
        f = k / (1 + z ** 2)
        if order == 1:
            return f ** 2 * (-2 / k) * z * g
        elif order == 2:
            fprime_on_z = f ** 2 * (-2 / k) * g
            return fprime_on_z * g + 2 * (fprime_on_z * z) ** 2 / f
        else:
            raise NotImplementedError

    def base_int(self, x1, x2, log10_ens=3, index=0):
        gamma1, gamma2, x0 = self.p
        # the only case where g1 and g2 can be different is if we're on the
        # 0th wrap, i.e. index=0; this also includes the case when we want
        # to use base_int to do a "full" integral
        if index == 0 and (x1 < x0) and (x2 > x0):
            g1, g2 = gamma1, gamma2
        else:
            g1, g2 = [gamma1] * 2 if ((x1 + index) < x0) else [gamma2] * 2
        z1 = (x1 + index - x0) / g1
        z2 = (x2 + index - x0) / g2
        k = 2.0 / (gamma1 + gamma2) / PI
        return k * (g2 * atan(z2) - g1 * atan(z1))

    def random(self, n):
        """ Use multinomial technique to return random photons from
            both components."""
        return two_comp_mc(n, self.p[0], self.p[1], self.p[-1], cauchy.rvs)


class LCVonMises(LCPrimitive):
    """ Represent a peak from the von Mises distribution.  This function is
        used in directional statistics and is naturally wrapped.

        Parameters:
            Width     inverse of the 'kappa' parameter in the std. def.
            Location  the center of the peak in phase
    """

    def init(self):
        self.p = np.asarray([0.05, 0.5])
        self.pnames = ["Width", "Location"]
        self.name = "VonMises"
        self.shortname = "VM"

    def hwhm(self, right=False):
        return 0.5 * np.arccos(self.p[0] * np.log(0.5) + 1) / TWOPI

    def __call__(self, phases, log10_ens=3):
        e, width, loc = self._make_p(log10_ens)
        z = TWOPI * (phases - loc)
        return np.exp(np.cos(z) / width) / i0(1.0 / width)

    def gradient(self, phases, log10_ens=3, free=False):
        e, width, loc = self._make_p(log10_ens)
        my_i0 = i0(1.0 / width)
        my_i1 = i1(1.0 / width)
        z = TWOPI * (phases - loc)
        cz = np.cos(z)
        sz = np.sin(z)
        f = (np.exp(cz) / width) / my_i0
        return np.asarray(
            [-cz / width ** 2 * f, TWOPI * (sz / width + my_i1 / my_i0) * f]
        )


class LCKing(LCWrappedFunction):
    """ Represent a (wrapped) King function peak.

        Parameters
        Sigma     the width parameter
        Gamma     the tail parameter
        Location  the mode of the distribution
    """

    # NOTES -- because we don't integrate over solid angle, the norm
    # integral / jacobean for the usual King function isn't trivial;
    # need to see if this is a show stopper

    def init(self):
        self.p = np.asarray([0.03, 0.5])
        self.pnames = ["Sigma", "Gamma", "Location"]
        self.name = "King"
        self.shortname = "K"

    def hwhm(self, right=False):
        raise NotImplementedError()
        return self.p[0] * (2 * np.log(2)) ** 0.5

    def base_func(self, phases, log10_ens=3, index=0):
        e, s, g, x0 = self._make_p(log10_ens)
        z = phases + index - x0
        u = 0.5 * (z / s) ** 2
        return (g - 1) / g * (1.0 + u / g) ** -g

    def base_grad(self, phases, log10_ens=3, index=0):
        raise NotImplementedError()
        e, width, x0 = self._make_p(log10_ens)
        z = (phases + index - x0) / width
        f = (1.0 / (width * ROOT2PI)) * np.exp(-0.5 * z ** 2)
        return np.asarray([f / width * (z ** 2 - 1.0), f / width * z])

    def base_int(self, x1, x2, log10_ens=3, index=0):
        e, s, g, x0 = self._make_p(log10_ens)
        z1 = x1 + index - x0
        z2 = x2 + index - x0
        u1 = 0.5 * ((x1 + index - x0) / s) ** 2
        u2 = 0.5 * ((x2 + index - x0) / s) ** 2
        f1 = 1 - (1.0 + u1 / g) ** (1 - g)
        f2 = 1 - (1.0 + u2 / g) ** (1 - g)
        if z1 * z2 < 0:  # span the peak
            return 0.5 * (f1 + f2)
        if z1 < 0:
            return 0.5 * (f1 - f2)
        return 0.5 * (f2 - f1)

    def random(self, n):
        raise NotImplementedError()
        if hasattr(n, "__len__"):
            n = len(n)
        return np.mod(norm.rvs(loc=self.p[-1], scale=self.p[0], size=n), 1)


class LCTopHat(LCPrimitive):
    """ Represent a top hat function.

        Parameters:
            Width     right edge minus left edge
            Location  center of top hat
    """

    def init(self):
        self.p = np.asarray([0.03, 0.5])
        self.pnames = ["Width", "Location"]
        self.name = "TopHat"
        self.shortname = "TH"
        self.fwhm_scale = 1

    def hwhm(self, right=False):
        return self.p[0] / 2

    def __call__(self, phases, wrap=True):
        width, x0 = self.p
        return np.where(np.mod(phases - x0 + width / 2, 1) < width, 1.0 / width, 0)

    def random(self, n):
        if hasattr(n, "__len__"):
            n = len(n)
        return np.mod(np.random.rand(n) * self.p[0] + self.p[-1] - self.p[0] / 2, 1)


class LCHarmonic(LCPrimitive):
    """Represent a sinusoidal shape corresponding to a harmonic in a Fourier expansion.

      Parameters:
         Location  the phase of maximum

    """

    def init(self):
        self.p = np.asarray([0.0])
        self.order = 1
        self.pnames = ["Location"]
        self.name = "Harmonic"
        self.shortname = "H"

    def __call__(self, phases, log10_ens=3):
        e, x0 = self._make_p(log10_ens)
        return 1 + 2 * np.cos((TWOPI * self.order) * (phases - x0))

    def integrate(self, x1, x2, log10_ens=3):
        e, x0 = self._make_p(log10_ens)
        t = self.order * TWOPI
        return (x2 - x1) + (np.sin(t * (x2 - x0)) - np.sin(t * (x1 - x0))) / t


class LCEmpiricalFourier(LCPrimitive):
    """ Calculate a Fourier representation of the light curve.
        The only parameter is an overall shift.
        Cannot be used with other LCPrimitive objects!

        Parameters:
           Shift     :     overall shift from original template phase
    """

    def init(self):
        self.nharm = 20
        self.p = np.asarray([0.0])
        self.free = np.asarray([True])
        self.pnames = ["Shift"]
        self.name = "Empirical Fourier Profile"
        self.shortname = "EF"
        self.shift_mode = True
        self.bounds = np.asarray([[0, 1]])

    def __init__(self, phases=None, input_file=None, **kwargs):
        """Must provide either phases or a template input file!"""
        self.init()
        self.__dict__.update(kwargs)
        if input_file is not None:
            self.from_file(input_file)
        if phases is not None:
            self.from_phases(phases)

    def from_phases(self, phases):
        n = float(len(phases))
        harmonics = np.arange(1, self.nharm + 1) * (2 * np.pi)
        self.alphas = np.asarray([(np.cos(k * phases)).sum() for k in harmonics])
        self.betas = np.asarray([(np.sin(k * phases)).sum() for k in harmonics])
        self.alphas /= n
        self.betas /= n
        self.harmonics = harmonics

    def from_file(self, input_file):
        if type(input_file) == type(""):
            toks = [
                line.strip().split()
                for line in open(input_file)
                if len(line.strip()) > 0 and "#" not in line
            ]
        else:
            toks = input_file
        alphas = []
        betas = []
        for tok in toks:
            if len(tok) != 2:
                continue
            try:
                a = float(tok[0])
                b = float(tok[1])
                alphas += [a]
                betas += [b]
            except:
                pass
        n = len(alphas)
        self.alphas = np.asarray(alphas)
        self.betas = np.asarray(betas)
        self.nharm = n
        self.harmonics = np.arange(1, n + 1) * (2 * np.pi)

    def to_file(self, output_file):
        f = open(output_file, "w")
        f.write("# fourier\n")
        for i in range(self.nharm):
            f.write("%s\t%s\n" % (self.alphas[i], self.betas[i]))

    def __call__(self, phases, log10_ens=None):
        """ NB energy-evolution currently not supported."""
        shift = self.p[0]
        harm = self.harmonics
        if shift != 0:
            """ shift theorem, for real coefficients
                It's probably a wash whether it is faster to simply
                subtract from the phases, but it's more fun this way! """
            c = np.cos(harm * shift)
            s = np.sin(harm * shift)
            a = c * self.alphas - s * self.betas
            b = s * self.alphas + c * self.betas
        else:
            a, b = self.alphas, self.betas

        ak = np.asarray([np.cos(phases * k) for k in harm]).transpose()
        bk = np.asarray([np.sin(phases * k) for k in harm]).transpose()
        return 1 + 2 * (a * ak + b * bk).sum(axis=1)

    def integrate(self, x1, x2):
        """ The Fourier expansion by definition includes the entire signal, so
        the norm is always unity."""
        return 1


class LCKernelDensity(LCPrimitive):
    """ Calculate a kernel density estimate of the light curve.
        The bandwidth is empirical, determined from examining several pulsars.
        The only parameter is an overall shift.
        Cannot be used with other LCPrimitive objects!

        Parameters:
            Shift     :     overall shift from original template phase
    """

    def init(self):
        self.bw = None
        self.use_scale = True
        self.max_contrast = 1
        self.resolution = 0.001  # interpolation sampling resolution
        self.p = np.asarray([0.0])
        self.free = np.asarray([True])
        self.pnames = ["Shift"]
        self.name = "Gaussian Kernel Density Estimate"
        self.shortname = "KD"
        self.shift_mode = True

    def __init__(self, phases=None, input_file=None, **kwargs):
        """Must provide either phases or a template input file!"""
        self.init()
        self.__dict__.update(kwargs)
        if input_file is not None:
            self.from_file(input_file)
        if phases is not None:
            self.from_phases(phases)

    def from_phases(self, phases):
        n = len(phases)
        # put in "ideal" HE bins after initial calculation of pulsed fraction
        # estimate pulsed fraction
        h = np.histogram(phases, bins=100)
        o = np.sort(h[0])
        p = (
            float((o[o > o[15]] - o[15]).sum()) / o.sum()
        )  # based on ~30% clean offpulse
        b = o[15]
        if self.bw is None:
            self.bw = (0.5 * (p ** 2 * n) ** -0.2) / (2 * np.pi)
            print(p, self.bw)
            local_p = np.maximum(h[0] - b, 0).astype(float) / h[0]
            print(local_p, b)
            bgbw = ((1 - p) ** 2 * n) ** -0.2 / (2 * np.pi)
            print(bgbw)
            self.bw = np.minimum((local_p ** 2 * h[0]) ** -0.2 / 100.0, bgbw)

        keys = np.searchsorted(h[1], phases)
        keys[keys == len(h[0])] = len(h[0]) - 1
        bw = self.bw[keys]
        print(len(phases), len(bw), type(bw))

        phases = phases.copy()
        self.phases = phases
        self.phases.sort()
        phases = np.asarray(phases)
        self.phases = np.asarray(phases)
        print(type(self.phases), type(phases))
        hi_mask = np.asarray(phases > 0.9)
        lo_mask = np.asarray(phases < 0.1)
        self.num = len(phases)
        self.phases = np.concatenate([phases[hi_mask] - 1, phases])
        self.phases = np.concatenate([self.phases, 1 + phases[lo_mask]])

        print(len(hi_mask), type(hi_mask), type(bw), len(bw))
        self.bw = np.concatenate([bw[hi_mask], bw])
        self.bw = np.concatenate([self.bw, bw[lo_mask]])

        # if self.bw is None:
        #   self.bw = len(phases)**-0.5

        dom = np.linspace(0, 1, int(1.0 / self.resolution))
        vals = self.__all_phases__(dom)
        ip = interp1d(dom, vals)
        mask = (self.phases > 0) & (self.phases < 1)

        """
        # this is a scaling that somehow works very well...
        vals = ip(self.phases[mask])
        scale = vals/(vals.max()-vals.min())*self.max_contrast
        #scale = scale**2
        #scale = (vals/vals.min())**1.5
        if self.use_scale:
         bw = self.bw / scale
        else:
         bw = self.bw * np.ones(len(vals))
        #bw = np.maximum(bw,self.resolution)
        """
        hi_mask = self.phases[mask] > 0.9
        lo_mask = self.phases[mask] < 0.1
        self.bw = np.concatenate([bw[hi_mask], bw])
        self.bw = np.concatenate([self.bw, bw[lo_mask]])

        vals = self.__all_phases__(dom)  # with new bandwidth
        self.interpolator = interp1d(dom, vals)
        self.xvals, self.yvals = dom, vals

    def __all_phases__(self, phases):
        return np.asarray(
            [
                (np.exp(-0.5 * ((ph - self.phases) / self.bw) ** 2) / self.bw).sum()
                for ph in phases
            ]
        ) / ((2 * np.pi) ** 0.5 * self.num)

    def from_file(self, input_file):
        if type(input_file) == type(""):
            toks = [
                line.strip().split()
                for line in open(input_file)
                if len(line.strip()) > 0 and "#" not in line
            ]
        else:
            toks = input_file

        xvals, yvals = np.asarray(toks).astype(float).transpose()
        self.xvals, self.yvals = xvals, yvals
        self.interpolator = interp1d(xvals, yvals)

    def __call__(self, phases):
        shift = self.p[0]
        if shift == 0:
            return self.interpolator(phases)
        # think this sign convention consistent with other classes - check.
        phc = np.mod(phases.copy() - shift, 1)
        """ MTK changed 25 Jul 2011
        if shift >= 0 : phc[phc<0] += 1
        else: phc[phc > 1] -= 1
        """
        return self.interpolator(phc)

    def to_file(self, output_file):
        f = open(output_file, "w")
        f.write("# kernel\n")
        for i in range(len(self.xvals)):
            f.write("%s\t%s\n" % (self.xvals[i], self.yvals[i]))

    def integrate(self, x1=0, x2=1):
        if (x1 == 0) and (x2 == 1):
            return 1.0
        # crude nearest neighbor approximation
        x = self.interpolator.x
        y = self.interpolator.y
        mask = (x >= x0) & (x <= x1)
        return simps(y[mask], x=x[mask])
        # return self.interpolator.y[mask].sum()/len(mask)


def convert_primitive(p1, ptype=LCLorentzian):
    """ Attempt to set the parameters of p2 to give a comparable primitive
        to p1."""
    p2 = ptype()
    p2_scale = p2.p[0] / p2.hwhm()
    # set position
    p2.p[-1] = p1.p[-1]
    # set width
    # default, 2->2 conversion
    p2.p[0] = p2_scale * p1.hwhm()
    # if we are going from 2->1, use mean of widths
    if (len(p2.p) == 2) and (len(p1.p) == 3):
        p2.p[0] = p2_scale * (p1.hwhm(right=False) + p1.hwhm(right=True)) / 2
    # if we are going from 1->2, duplicate
    elif (len(p2.p) == 3) and (len(p1.p) == 2):
        p2.p[1] = p2.p[0]
    # special case of going from gauss to Lorentzian
    # this makes peaks closer in nature than going by equiv HWHM
    if "Gaussian" in str(type(p1)) and "Lorentzian" in str(type(p2)):
        scale = p2(p1.p[-1]) / p1(p1.p[-1])
        p2.p[0] *= scale
        if len(p2.p) == 3:
            p2.p[1] *= scale
    return p2
