"""Handling normalization of light curves with an arbitrary number of primitive components.

This is done by treating each primitive's normalization parameter as
the square of a cartesian variable lying within or on an
n-dimensional ball of unit radius.

author: M. Kerr <matthew.kerr@gmail.com>
"""
from __future__ import absolute_import, division, print_function

from math import acos, asin, cos, pi, sin

import numpy as np

# can some of the code be reduced with inheritance here?
# TODO -- error propagation to norms


class NormAngles(object):
    """Keep track of N angles (0 to pi/2) representing the coordinates inside a unit radius N-ball.

    Generally, the apportionment of the amplitudes of components is
    indicated by the position of the vector, while the overall
    normalization is given by the an additional angle, the sine of
    which provides the (squared) normalization.

    Parameters
    ----------
    norms : tuple or array
        The amplitudes of a set of components; their sum must be <= 1.
    """

    def is_energy_dependent(self):
        return False

    def init(self):
        self.free = np.asarray([True] * self.dim, dtype=bool)
        self.errors = np.zeros(self.dim)
        self.pnames = ["Ang%d" % (i + 1) for i in range(self.dim)]
        self.name = "NormAngles"
        self.shortname = "None"

    def copy(self):
        from copy import deepcopy

        return deepcopy(self)

    def _asarrays(self):
        for key in ["p", "free", "bounds", "errors", "slope", "slope_free"]:
            if hasattr(self, key):
                v = self.__dict__[key]
                if v is not None:
                    self.__dict__[key] = np.asarray(
                        v, dtype=bool if "free" in key else float
                    )

    def __init__(self, norms, **kwargs):
        self.dim = len(norms)
        self.init()
        if not self._check_norms(norms):
            raise ValueError(
                "Provided norms ... \n%s\n ... do not satisfy constraints."
                % (str(norms))
            )
        self.p = self._get_angles(norms)
        self.__dict__.update(**kwargs)
        self._asarrays()

    def __str__(self):
        # IN PROGRESS
        norms = self()
        errs = self.get_errors(free=False, propagate=True)
        dcderiv = 2 * sin(self.p[0]) * cos(self.p[0])
        dcerr = self.errors[0] * abs(dcderiv)

        def norm_string(i):
            fstring = "" if self.free[i] else " [FIXED]"
            return "P%d : %.4f +\- %.4f%s" % (i + 1, norms[i], errs[i], fstring)

        s0 = (
            "\nMixture Amplitudes\n------------------\n"
            + "\n".join([norm_string(i) for i in range(self.dim)])
            + "\nDC : %.4f +\- %.4f" % (1 - self.get_total(), dcerr)
        )
        return s0

    def __len__(self):
        return self.dim

    def _check_norms(self, norms, eps=1e-15):
        ok = True
        for n in norms:
            ok = ok and (n <= (1 + eps))
        return ok and (sum(norms) <= (1 + eps))

    def _get_angles(self, norms):
        """ Determine the n-sphere angles from a set of normalizations."""
        sines = sum(norms) ** 0.5
        if sines > 1:
            if abs(sines - 1) < 1e-12:
                sines = 1
            else:
                raise ValueError("Invalid norm specification")
        angles = [asin(sines)]
        norms = np.asarray(norms) ** 0.5
        for i in range(self.dim - 1):
            t = norms[i] / sines
            if t > 1:
                if abs(t - 1) < 1e-12:
                    t = 1
                else:
                    raise ValueError("Invalid norm specification")
            phi = acos(t)
            sines *= sin(phi)
            angles.append(phi)
        return np.asarray(angles)

    def set_parameters(self, p, free=True):
        if free:
            self.p[self.free] = p
        else:
            self.p[:] = p

    def get_parameters(self, free=True):
        if free:
            return self.p[self.free]
        return self.p

    def get_parameter_names(self, free=True):
        return [p for (p, b) in zip(self.pnames, self.free) if b]

    def set_errors(self, errs):
        """ errs an array with the 1-sigma error estimates with shape
            equal to the number of free parameters."""
        self.errors[:] = 0.0
        self.errors[self.free] = errs

    def get_errors(self, free=True, propagate=True):
        """ Get errors on components.  If specified, propagate errors from
            the internal angle parameters to the external normalizations.
        """
        # TODO -- consider using finite difference instead
        if not propagate:
            return self.errors[self.free] if free else self.errors
        g = self.gradient() ** 2
        g *= self.errors ** 2
        errors = g.sum(axis=1) ** 0.5
        return errors[self.free] if free else errors

    def get_bounds(self):
        """ Angles are always [0,pi/2). """
        return np.asarray([[0, pi / 2] for i in range(self.dim)])[self.free]

    def sanity_checks(self, eps=1e-6):
        t1 = abs(self().sum() - sin(self.p[0]) ** 2) < eps
        return t1

    def __call__(self, log10_ens=3):
        """ Return the squared value of the Cartesian coordinates.

            E.g., for a 4-sphere, return
            x0^2 = sin^2(a)*cos^2(b)
            x1^2 = sin^2(a)*sin^2(b)*cos^2(c)
            x2^2 = sin^2(a)*sin^2(b)*sin^2(c)*cos^2(d)
            x3^2 = sin^2(a)*sin^2(b)*sin^2(c)*sin^2(d)

            Recall that the normalization is *also* given as an angle,
            s.t. the vector lies within the unit sphere.

            These values are guaranteed to satisfy the constraint of
            a sum <= unity and so are suitable for normalizations of
            a light curve.
        """
        p = self.p
        m = sin(p[0])  # normalization
        norms = np.empty(self.dim)
        for i in range(1, self.dim):
            norms[i - 1] = m * cos(p[i])
            m *= sin(p[i])
        norms[self.dim - 1] = m
        return norms ** 2

    def gradient(self, log10_ens=3, free=False):
        """ Return a matrix giving the value of the partial derivative
            of the ith normalization with respect to the jth angle, i.e.

            M_ij = dn_i/dphi_j

            This is the relevant quantity because it is the angles that are
            actually fit for, so this allows the application of the chain
            rule.

            Because of the way the normalizations are defined, the ith
            normalization only depends on the (i+1)th first angles, so
            the upper half of M_ij is zero (see break statement below).

            For taking higher derivatives, it is convenient to express
            the derivative as so:
            -d(cos^2(x)) = d(sin^2(x)) = sin(2x)
            So that any non-zero derivative can be expressed by taking
            the norm, dividing by sin^2/cos^2 as appropriate, and
            multiplying by +/- sin(2x).  Then higher derivatives simply
            pop out a factor of +/-2 and toggle sin/cos.
        """
        m = np.zeros([self.dim, self.dim], dtype=float)
        n = self()
        p = self.p
        s2p = np.sin(2 * p)
        cp = -s2p / np.cos(p) ** 2
        sp = s2p / np.sin(p) ** 2
        # loop over normalizations
        for i in range(self.dim):
            for j in range(self.dim):
                if j > i + 1:
                    break
                if j <= i:
                    # these will always be sin^2 terms
                    m[i, j] = n[i] * sp[j]
                else:
                    # last term is cosine for all but last norm, but we won't
                    # get to it here because j==i is the last term then
                    m[i, j] = n[i] * cp[j]
        if free:
            return m[:, self.free]
        return m

    def hessian(self, log10_ens=3, free=False):
        """ Return a matrix giving the value of the 2nd partial derivative
            of the ith normalization with respect to the jth and kth
            angles,

            M_ijk = dn2_i/dphi_j/dphi_k

            See above notes for gradient.  In general, the cases are

            j < k <= i; just calculate as gradient, getting two sin(2x) terms
            j = k <= i; in this case, pick up a single 2*cos(2x) instead


        """
        m = np.zeros([self.dim, self.dim, self.dim], dtype=float)
        n = self()
        p = self.p
        s2p = np.sin(2 * p)
        c2p = np.cos(2 * p)
        cp = -s2p / np.cos(p) ** 2
        sp = s2p / np.sin(p) ** 2
        # loop over normalizations
        g = self.gradient(free=False)
        for i in range(self.dim):
            for j in range(self.dim):
                if j > i + 1:
                    break
                for k in range(self.dim):
                    if k > i + 1:
                        break
                    if (j <= i) and (k <= i):
                        if j != k:
                            # two separate sines replacing sin^2
                            m[i, j, k] = n[i] * sp[j] * sp[k]
                        else:
                            # diff same sine twice, getting a 2*cos
                            m[i, j, k] = n[i] * 2 * c2p[j] / np.sin(p[j]) ** 2
                    else:
                        # at least one of j, k is a cos^2 term, so we pick up
                        # a negative and need to divide by cos^2
                        if j != k:
                            if j == i + 1:
                                m[i, j, k] = n[i] * cp[j] * sp[k]
                            elif k == i + 1:
                                m[i, j, k] = n[i] * sp[j] * cp[k]
                        else:
                            # both are the cos^2 term, so we get a -2*cos
                            m[i, j, k] = n[i] * (-2) * c2p[j] / np.cos(p[j]) ** 2
        if free:
            return m[:, self.free, self.free]
        return m

    def get_total(self):
        """ Return the amplitude of all norms."""
        return sin(self.p[0]) ** 2

    def set_total(self, val):
        """ Set overall normalization of the represented components."""
        norms = self()
        self.p = self._get_angles(norms * (val / norms.sum()))

    def set_single_norm(self, index, val):
        norms = self()
        norms[index] = val
        if not self._check_norms(norms):
            raise ValueError(
                "Provided norms ... \n%s\n ... do not satisfy constraints."
                % (str(norms))
            )
        self.p = self._get_angles(norms)

    def eval_string(self):
        """ Return a string that can be evaled to instantiate a nearly-
            identical object."""
        t = self()
        if len(t.shape) > 1:
            t = t[:, 0]  # handle e-dep
        return "%s(%s,free=%s,slope=%s,slope_free=%s)" % (
            self.__class__.__name__,
            str(list(t)),
            str(list(self.free)),
            str(list(self.slope)) if hasattr(self, "slope") else None,
            str(list(self.slope_free)) if hasattr(self, "slope_free") else None,
        )

    def dict_string(self):
        """ Round down to avoid input errors w/ normalization."""
        t = self()
        if len(t.shape) > 1:
            t = t[:, 0]  # handle e-dep

        def pretty_list(l, places=6, round_down=True):
            if round_down:
                r = np.round(l, decimals=places)
                r[r > np.asarray(l)] -= 10 ** -places
            else:
                r = l
            fmt = "%." + "%d" % places + "f"
            s = ", ".join([fmt % x for x in r])
            return "[" + s + "]"

        return [
            "name = %s" % self.__class__.__name__,
            "norms = %s" % (pretty_list(t)),
            "free = %s" % (str(list(self.free))),
            "slope = %s"
            % (
                pretty_list(self.slope, round_down=False)
                if hasattr(self, "slope")
                else None
            ),
            "slope_free = %s"
            % (str(list(self.slope_free)) if hasattr(self, "slope_free") else None),
        ]


def numerical_gradient(norms, delta=1e-3):
    """ Check the accuracy of analytic version.  (HINT -- it checks out.)
    """
    rvals = np.empty((norms.dim, norms.dim))
    p = norms.p.copy()
    for i in range(norms.dim):
        norms.p[i] = p[i] + delta
        hi = norms()
        norms.p[i] = p[i] - delta
        lo = norms()
        rvals[:, i] = (hi - lo) / (2 * delta)
        norms.p[:] = p
    return rvals


def numerical_hessian(norms, delta=1e-3):
    rvals = np.empty((norms.dim, norms.dim, norms.dim))
    p = norms.p.copy()
    for i in range(norms.dim):
        for j in range(i, norms.dim):

            norms.p[i] += delta
            norms.p[j] += delta
            hihi = norms()
            norms.p[:] = p

            norms.p[i] += delta
            norms.p[j] -= delta
            hilo = norms()
            norms.p[:] = p

            norms.p[i] -= delta
            norms.p[j] += delta
            lohi = norms()
            norms.p[:] = p

            norms.p[i] -= delta
            norms.p[j] -= delta
            lolo = norms()
            norms.p[:] = p

            rvals[:, i, j] = (hihi + lolo - hilo - lohi) / (4 * delta ** 2)
            rvals[:, j, i] = rvals[:, i, j]
    return rvals
