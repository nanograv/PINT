"""Handling normalization of light curves with an arbitrary number of primitive components.

This is done by treating each primitive's normalization parameter as
the square of a cartesian variable lying within or on an
n-dimensional ball of unit radius.

author: M. Kerr <matthew.kerr@gmail.com>
"""
import numpy as np

# can some of the code be reduced with inheritance here?
# TODO -- error propagation to norms


def isvector(x):
    return len(np.asarray(x).shape) > 0


class NormAngles:
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

    def init(self):
        self.free = np.asarray([True] * self.dim, dtype=bool)
        self.errors = np.zeros(self.dim)
        self.pnames = ["Ang%d" % (i + 1) for i in range(self.dim)]
        self.name = "NormAngles"
        self.shortname = "None"

    def is_energy_dependent(self):
        return False

    def num_parameters(self, free=True):
        return np.sum(self.free) if free else len(self.free)

    def get_free_mask(self):
        """Return a mask with True if parameters are free, else False."""
        return self.free

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

    def __str__(self):
        norms = self()
        errs = self.get_errors(free=False, propagate=True)
        dcderiv = 2 * np.sin(self.p[0]) * np.cos(self.p[0])
        dcerr = self.errors[0] * abs(dcderiv)

        # for energy-dependence
        if self.is_energy_dependent():
            M = self._eindep_gradient(log10_ens=3, free=False)
            eff_slopes = M @ self.slope
            eff_errors = ((M**2) @ self.slope_errors**2) ** 0.5  # CHECK

        def norm_string(i):
            fstring = "" if self.free[i] else " [FIXED]"
            l1 = "P%d : %.4f +\\- %.4f%s" % (i + 1, norms[i], errs[i], fstring)
            if self.is_energy_dependent():
                fstring = "" if self.slope_free[i] else " [FIXED]"
                l1 += "\n (Slope) : %.4f +\\- %.4f%s" % (
                    eff_slopes[i],
                    eff_errors[i],
                    fstring,
                )
            return l1

        s0 = (
            "\nMixture Amplitudes\n------------------\n"
            + "\n".join([norm_string(i) for i in range(self.dim)])
            + "\nDC : %.4f +\\- %.4f" % (1 - self.get_total(), dcerr)
        )
        if self.is_energy_dependent():
            s0 += "\n (Slope) : %.4f" % (-eff_slopes.sum())
        return s0

    def __len__(self):
        return self.dim

    def _make_p(self, log10_ens=3):
        if isvector(log10_ens):
            p = np.empty([self.dim, len(log10_ens)])
            for i in range(self.dim):
                p[i] = self.p[i]
        else:
            p = self.p.copy()
        return None, p

    # TODO -- vectorize
    def _check_norms(self, norms, eps=1e-15):
        ok = True
        for n in norms:
            ok = ok and (n <= (1 + eps))
        return ok and (sum(norms) <= (1 + eps))

    def _get_angles(self, norms):
        """Determine the n-sphere angles from a set of normalizations."""
        sines = sum(norms) ** 0.5
        if sines > 1:
            if abs(sines - 1) < 1e-12:
                sines = 1
            else:
                raise ValueError("Invalid norm specification")
        angles = [np.arcsin(sines)]
        norms = np.asarray(norms) ** 0.5
        for i in range(self.dim - 1):
            t = norms[i] / sines
            if t > 1:
                if abs(t - 1) < 1e-12:
                    t = 1
                else:
                    raise ValueError("Invalid norm specification")
            phi = np.arccos(t)
            sines *= np.sin(phi)
            angles.append(phi)
        return np.asarray(angles)

    def set_parameters(self, p, free=True):
        if free:
            self.p[self.free] = p
        else:
            self.p[:] = p

    def get_parameters(self, free=True):
        return self.p[self.free] if free else self.p

    def get_parameter_names(self, free=True):
        return [p for (p, b) in zip(self.pnames, self.free) if b]

    def set_errors(self, errs):
        """errs an array with the 1-sigma error estimates with shape
        equal to the number of free parameters."""
        self.errors[:] = 0.0
        self.errors[self.free] = errs

    def get_errors(self, free=True, propagate=True):
        """Get errors on components.  If specified, propagate errors from
        the internal angle parameters to the external normalizations.
        """
        if not propagate:
            return self.errors[self.free] if free else self.errors
        g = self.gradient(log10_ens=3, free=free) ** 2
        g *= self.errors**2
        errors = g.sum(axis=1) ** 0.5
        return errors

    def get_bounds(self, free=True):
        """Angles are always [0,pi/2)."""
        PI2 = np.pi * 0.5
        if free:
            return [[0, PI2] for x in self.free if x]
        return [[0, PI2] for _ in self.free]

    def sanity_checks(self, eps=1e-6):
        return np.abs(self().sum() - np.sin(self.p[0]) ** 2) < eps

    def __call__(self, log10_ens=3):
        """Return the squared value of the Cartesian coordinates.

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

        NB this version should work with both scalar (log10_ens=3) and
        vector (log10_ens an array) versions.  The shape is [nparam] for
        scalar and [nparam, nenergy] for vector.
        """
        _, p = self._make_p(log10_ens=log10_ens)
        m = np.sin(p[0])  # normalization
        norms = np.empty_like(p)
        for i in range(1, self.dim):
            norms[i - 1] = m * np.cos(p[i])
            m *= np.sin(p[i])
        norms[self.dim - 1] = m
        norms = norms**2
        # check normalization condition -- can fail numerically
        q = norms.sum(axis=0)
        if np.any(q > 1):
            if len(norms.shape) == 2:
                # for vector case, make sure we don't divide by sum of norms
                # except for cases where the norms actually exceed 1!
                q[q < 1] = 1
                return norms * (1.0 / q)[None, :]
            return norms * (1.0 / q)
        return norms

    def _eindep_gradient(self, log10_ens=3, free=False):
        _, p = self._make_p(log10_ens=log10_ens)
        n = self(log10_ens)
        if len(p.shape) == 1:
            m = np.zeros((self.dim, self.dim), dtype=float)
        else:
            m = np.zeros((self.dim, self.dim, p.shape[1]), dtype=float)
        s2p = np.sin(2 * p)
        # NB -- the gradient is always well defined, and numerical issues
        # here stem from the way the gradient is evaluated by first
        # calculating the normal. I *think* that any matrix element that
        # involves cot(p) where p-->0 will always be 0.  Thus, simply set
        # the places where p is 0 to 0.
        mask = p % (0.5 * np.pi) != 0
        sp = np.zeros(p.shape)
        cp = -s2p / np.cos(p) ** 2  # = -2*tan(p)
        cp[~mask] = 0
        sp = np.zeros_like(cp)
        sp[mask] = s2p[mask] / np.sin(p[mask]) ** 2  # = 2*cot(p)
        # loop over normalizations
        for i in range(self.dim):
            for j in range(self.dim):
                if j > i + 1:
                    break
                # these will always be sin^2 terms if j<=i
                # else, the last term is cosine for all but last norm, but we won't
                # get to it here because j==i is the last term then
                m[i, j] = n[i] * sp[j] if j <= i else n[i] * cp[j]
        return m[:, self.free] if free else m

    def gradient(self, log10_ens=3, free=False):
        """Return a matrix giving the value of the partial derivative
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
        return self._eindep_gradient(log10_ens=log10_ens, free=free)

    def hessian(self, log10_ens=3, free=False):
        """Return a matrix giving the value of the 2nd partial derivative
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
        for i in range(self.dim):
            for j in range(self.dim):
                if j > i + 1:
                    break
                for k in range(self.dim):
                    if k > i + 1:
                        break
                    if (j <= i) and (k <= i):
                        m[i, j, k] = (
                            n[i] * sp[j] * sp[k]
                            if j != k
                            else n[i] * 2 * c2p[j] / np.sin(p[j]) ** 2
                        )
                    elif j != k:
                        if j == i + 1:
                            m[i, j, k] = n[i] * cp[j] * sp[k]
                        elif k == i + 1:
                            m[i, j, k] = n[i] * sp[j] * cp[k]
                    else:
                        # both are the cos^2 term, so we get a -2*cos
                        m[i, j, k] = n[i] * (-2) * c2p[j] / np.cos(p[j]) ** 2
        return m[:, self.free, self.free] if free else m

    def get_total(self):
        """Return the amplitude of all norms."""
        return np.sin(self.p[0]) ** 2

    def set_total(self, val):
        """Set overall normalization of the represented components."""
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

    def delete_component(self, index):
        """Remove a component and return resulting object.

        Pulsed normalization is preserved, and NormAngles or ENormAngles is
        return as appropriate.
        """
        mask = np.ones(self.dim, dtype=bool)
        mask[index] = False
        norms = self()
        newnorms = norms[mask] * norms.sum() / norms[mask].sum()
        n = self.__class__(newnorms)
        n.free[:] = self.free[mask]
        n.errors[:] = self.errors[mask]
        if n.is_energy_dependent():
            print(
                "Warning!  Energy-dependence will be slightly altered when changing components."
            )
            n.slope[:] = self.slope[mask]
            n.slope_free[:] = self.slope_free[mask]
            n.slope_errors[:] = self.slope_errors[mask]
            # get desired energy slopes for new object (approximate)
            M = self._eindep_gradient(log10_ens=3, free=False)
            effective_slopes = (M @ self.slope)[mask]
            M = n._eindep_gradient(log10_ens=3, free=False)
            n.slope[:] = np.linalg.inv(M) @ effective_slopes
        return n

    def add_component(self, norm=0.1):
        """Add a component and return resulting object.

        The normalization is specified as a fraction of the current pulsed
        normalization, such that the resulting NormAngles or ENormAngles
        object will have the same overall pulsed fraction at the pivot en.
        """
        norms = self()
        pulsed = norms.sum()
        newnorms = np.append(norms * (1 - norm), norm * pulsed)
        n = self.__class__(newnorms)
        n.free[:] = np.append(self.free, True)
        n.errors[:] = np.append(self.errors, 0)
        if n.is_energy_dependent():
            n.slope[:] = np.append(self.slope, 0)
            n.slope_free[:] = np.append(self.slope_free, True)
            n.slope_errors[:] = np.append(self.slope_errors, 0)
            ## get desired energy slopes for new object (approximate)
            M = self._eindep_gradient(log10_ens=3, free=False)
            effective_slopes = np.append((M @ self.slope) * (1 - norm), 0)
            M = n._eindep_gradient(log10_ens=3, free=False)
            n.slope[:] = np.linalg.inv(M) @ effective_slopes
        return n

    def reorder_components(self, indices):
        # currently this is only APPROXIMATE, not sure if it's possible
        # to actually re-order them.  Perhaps consider a mapping instead.
        if len(indices) != self.dim:
            raise ValueError("New indices do not match component count.")
        if self.is_energy_dependent():
            print(
                "Warning!  Energy-dependence will be slightly altered when re-ordering components."
            )
            # save the linearized slopes before changing params
            M = self._eindep_gradient(log10_ens=3, free=False)
            effective_slopes = (M @ self.slope)[indices]
        self.p[:] = self._get_angles(self()[indices])
        self.free[:] = self.free[indices]
        self.errors[:] = self.errors[indices]
        if self.is_energy_dependent():
            M = self._eindep_gradient(log10_ens=3, free=False)
            self.slope[:] = np.linalg.inv(M) @ effective_slopes
            self.slope_free[:] = self.slope_free[indices]
            ## TODO -- but I probably don't care about this much
            self.slope_errors[:] = self.slope_errors[indices]
        # TODO -- I don't think just swapping the order will work for
        # slopes, probably need to use the gradient to convert!

    def eval_string(self):
        """Return a string that can be evaluated to instantiate a nearly-
        identical object."""
        t = self()
        if len(t.shape) > 1:
            t = t[:, 0]  # handle e-dep
        return f'{self.__class__.__name__}({list(t)},free={list(self.free)},slope={str(list(self.slope)) if hasattr(self, "slope") else None},slope_free={str(list(self.slope_free)) if hasattr(self, "slope_free") else None})'

    def dict_string(self):
        """Round down to avoid input errors w/ normalization."""
        t = self()
        if len(t.shape) > 1:
            t = t[:, 0]  # handle e-dep

        def pretty_list(l, places=6, round_down=True):
            if round_down:
                r = np.round(l, decimals=places)
                r[r > np.asarray(l)] -= 10**-places
            else:
                r = l
            fmt = "%." + "%d" % places + "f"
            s = ", ".join([fmt % x for x in r])
            return f"[{s}]"

        return [
            f"name = {self.__class__.__name__}",
            f"norms = {pretty_list(t)}",
            f"free = {list(self.free)}",
            "slope = %s"
            % (
                pretty_list(self.slope, round_down=False)
                if hasattr(self, "slope")
                else None
            ),
            f'slope_free = {str(list(self.slope_free)) if hasattr(self, "slope_free") else None}',
        ]


def numerical_gradient(norms, delta=1e-3):
    """Check the accuracy of analytic version.  (HINT -- it checks out.)"""
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

            rvals[:, i, j] = (hihi + lolo - hilo - lohi) / (4 * delta**2)
            rvals[:, j, i] = rvals[:, i, j]
    return rvals
