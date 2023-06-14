from pint.templates.lcnorm import NormAngles
import numpy as np

# can some of the code be reduced with inheritance here?
# TODO -- error propagation to norms


def isvector(x):
    return len(np.asarray(x).shape) > 0


class ENormAngles(NormAngles):
    def __init__(self, norms, slope=None, slope_free=None, **kwargs):
        # TODO -- keyword checking
        """norms -- a tuple or array with the amplitudes of a set of
        components; their sum must be <= 1."""
        super().__init__(norms, **kwargs)
        if slope is None:
            slope = np.zeros(self.dim)
        self.slope = np.asarray(slope, dtype=float)
        if slope_free is None:
            slope_free = np.asarray([False] * self.dim)
        self.slope_free = np.asarray(slope_free, dtype=bool)
        self.slope_errors = np.zeros(self.dim)

    def is_energy_dependent(self):
        return True

    def set_parameters(self, p, free=True):
        if free:
            n = sum(self.free)
            self.p[self.free] = p[:n]
            self.slope[self.slope_free] = p[n:]
        else:
            n = len(self.p)
            self.p[:] = p[:n]
            self.slope[:] = p[n:]

    def get_parameters(self, free=True):
        if free:
            return np.append(self.p[self.free], self.slope[self.slope_free])
        return np.append(self.p, self.slope)

    def num_parameters(self, free=True):
        if free:
            return np.sum(self.free) + np.sum(self.slope_free)
        else:
            return len(self.free) + len(self.slope_free)

    def get_free_mask(self):
        """Return a mask with True if parameters are free, else False."""
        return np.append(self.free, self.slope_free)

    def get_bounds(self, free=True):
        PI2 = np.pi * 0.5
        b1 = np.asarray([[0, PI2] for _ in range(self.dim)])
        b2 = np.asarray([[-PI2, PI2] for _ in range(self.dim)])
        if free:
            return np.concatenate((b1[self.free], b2[self.slope_free]))
        else:
            return np.concatenate((b1, b2))

    def set_errors(self, errs):
        n0 = self.free.sum()
        n1 = self.slope_free.sum()
        self.errors[:] = 0.0
        self.slope_errors[:] = 0.0
        self.errors[self.free] = errs[:n0]
        self.slope_errors[self.slope_free] = errs[n0 : n0 + n1]

    def get_errors(self, free=True, propagate=True):
        """Get errors on components.  If specified, propagate errors from
        the internal angle parameters to the external normalizations.
        """
        if free:
            e = np.append(self.errors[self.free], self.slope_errors[self.slope_free])
        else:
            e = np.append(self.errors, self.slope_errors)
        if not propagate:
            return e
        # what we want here is the conversion between the normalization
        # parameter and the angles, bearing in mind that the slopes are
        # also in angle space if we're using the e-dep model.
        # TODO -- fixed to the median energy, not sure how to propagate,
        # consider later...
        g = self.gradient(log10_ens=3, free=free) ** 2
        g *= e**2
        errors = g.sum(axis=1) ** 0.5
        return errors

    def _make_p(self, log10_ens):
        e, p = super()._make_p(log10_ens)
        de = np.asarray(log10_ens) - 3
        bounds = self.get_bounds()
        if isvector(log10_ens):
            for i in range(self.dim):
                np.clip(p[i] + self.slope[i] * de, bounds[i][0], bounds[i][1], out=p[i])
        else:
            for i in range(self.dim):
                p[i] = np.clip(p[i] + self.slope[i] * de, bounds[i][0], bounds[i][1])
        return de, p

    def gradient(self, log10_ens, free=True):
        # dimension is num_norms x num_params x num_energies
        # TODO -- need to make decision on "clip check" like with prims
        g0 = self._eindep_gradient(log10_ens=log10_ens, free=False)
        e, p = self._make_p(log10_ens)
        if len(p.shape) == 1:
            rvals = np.empty((self.dim, 2 * self.dim))
        else:
            rvals = np.empty((self.dim, 2 * self.dim, p.shape[1]))
        rvals[:, : self.dim] = g0
        rvals[:, self.dim :] = g0 * e
        return rvals[:, np.append(self.free, self.slope_free)] if free else rvals
