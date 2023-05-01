from pint.templates.lcprimitives import *


def isvector(x):
    return len(np.asarray(x).shape) > 0


def edep_gradient(self, grad_func, phases, log10_ens=3, free=False):
    """Return the analytic gradient of a general LCEPrimitive.

    The evaluation is similar to the non-energy dependent version, and
    since this is a linear model, application of the chain rule simply
    returns the gradient at the indicated parameters, but weighted by
    the difference in (log) energy.

    However, there is one complication.  Because of the bounds enforced
    by "_make_p", the gradient for the slope parameters vanishes at
    some energies when the bound has saturated.  These entries should be
    zeroed.
    """
    t = self._make_p(log10_ens)
    e = t[0]
    p = t[1:]
    # NB -- use "False" here to handle case where a parameter might be
    # fixed but the slope free; this isn't really a performance hit
    # because right now the base function computes the gradient for
    # every parameter anyway
    g = grad_func(phases, log10_ens, free=False)
    n = g.shape[0]
    t = np.empty([2 * n, len(phases)])
    t[:n, :] = g
    t[n:, :] = e * g
    # apply correction for parameter clipped
    bounds = self.get_bounds(free=False)
    for i in range(n):
        lo_mask = p[i] <= bounds[i][0]
        hi_mask = p[i] >= bounds[i][1]
        t[n + i, lo_mask | hi_mask] = 0
        t[i, lo_mask | hi_mask] = 0
    return t[np.append(self.free, self.slope_free)] if free else t


class LCEPrimitive(LCPrimitive):
    def is_energy_dependent(self):
        return True

    # TODO -- this is so awkward, fix it?
    def parse_kwargs(self, kwargs):
        # acceptable keyword arguments, can be overridden by children
        recognized_kwargs = ["p", "free", "slope", "slope_free"]
        for key in kwargs.keys():
            if key not in recognized_kwargs:
                raise ValueError(f"kwarg {key} not recognized")
        self.__dict__.update(kwargs)

    def _einit(self):
        """Do setup work common to energy-dependent primitives.
        Should be called after all other common code."""
        n = len(self.p)
        self.slope = np.zeros(n)
        self.slope_free = np.zeros(n, dtype=bool)
        self.slope_bounds = np.asarray([[-0.15, 0.15]] * n, dtype=float)
        self.slope_errors = np.zeros(n)

    def num_parameters(self, free=True):
        if free:
            return np.sum(self.free) + np.sum(self.slope_free)
        return len(self.free) + len(self.slope_free)

    def get_free_mask(self):
        """Return a mask with True if parameters are free, else False."""
        return np.append(self.free, self.slope_free)

    def get_parameters(self, free=True):
        if free:
            return np.append(self.p[self.free], self.slope[self.slope_free])
        return np.append(self.p, self.slope)

    def set_parameters(self, p, free=True):
        if free:
            n = sum(self.free)
            self.p[self.free] = p[:n]
            self.slope[self.slope_free] = p[n:]
        else:
            n = len(self.p)
            self.p[:] = p[:n]
            self.slope[:] = p[n:]
        # adjust position to be between 0 and 1
        self.p[-1] = self.p[-1] % 1
        return np.all(self.p >= 0)

    def set_errors(self, errs):
        n0 = self.free.sum()
        n1 = self.slope_free.sum()
        self.errors[:] = 0.0
        self.slope_errors[:] = 0.0
        self.errors[self.free] = errs[:n0]
        self.slope_errors[self.slope_free] = errs[n0 : n0 + n1]
        return n0 + n1

    def get_bounds(self, free=True):
        if free:
            return np.append(
                np.asarray(self.bounds)[self.free],
                np.asarray(self.slope_bounds)[self.slope_free],
                axis=0,
            )
        return np.append(np.asarray(self.bounds), np.asarray(self.slope_bounds), axis=0)

    def __str__(self):
        m = max([len(n) for n in self.pnames] + [len(" (Slope)")])
        l = []
        for i in range(len(self.pnames)):
            fstring_p = "" if self.free[i] else " [FIXED]"
            fstring_s = "" if self.slope_free[i] else " [FIXED]"
            n = self.pnames[i][:m]
            t_n = n + (m - len(n)) * " "
            l += [t_n + ": %.4f +\\- %.4f%s" % (self.p[i], self.errors[i], fstring_p)]
            n = " (Slope)"
            t_n = n + (m - len(n)) * " "
            l += [
                t_n
                + ": %.4f +\\- %.4f%s"
                % (self.slope[i], self.slope_errors[i], fstring_s)
            ]
        l = [self.name + "\n------------------"] + l
        return "\n".join(l)

    def _make_p(self, log10_ens):
        e = np.asarray(log10_ens, dtype=float) - 3
        # allow saturation at bounds
        return [e] + [
            np.clip(x + y * e, b[0], b[1])
            for x, y, b in zip(self.p, self.slope, self.bounds)
        ]
        # return [e] + [x + y*e for x,y in zip(self.p,self.slope)]

    def get_fixed_energy_version(self, log10_en=3):
        """Return the version of this primitive with parameters
        appropriate for the given energy.  I think this will be
        unnecessary once everything is signature compatible."""
        constructor = self.__class__.__bases__[-1]
        p = self._make_p(log10_en)[1:]
        free = self.free.copy()
        prim = constructor(p=p, free=self.free)
        prim.bounds[:] = self.bounds[:]
        return prim


class LCEWrappedFunction(LCEPrimitive, LCWrappedFunction):
    __doc__ = LCWrappedFunction.__doc__

    def gradient(self, phases, log10_ens=3, free=False):
        g = super().gradient
        return edep_gradient(self, g, phases, log10_ens=log10_ens, free=free)

    def hessian(self, phases, log10_ens=3, free=False):
        """Return the hessian.

        For the linear model, this is a block matrix where the upper-left
        corner is the original hessian, the upper-right/lower-left corners
        are hessians weighted by the energy, and the lower-right by e^2.
        """
        t = self._make_p(log10_ens)
        e = t[0]
        h = super(LCEWrappedFunction, self).hessian(phases, log10_ens, free=False)
        n = h.shape[0]
        H = np.empty((2 * n, 2 * n, len(phases)))
        H[:n, :n] = h
        H[n:, :n] = h * e
        H[:n, n:] = H[n:, :n]  # h is already symmetric
        H[n:, n:] = h * e**2
        assert np.all(np.swapaxes(H, 0, 1) == H)
        if free:
            m = self.get_free_mask()
            return H[m, m]
        return H


class LCEGaussian(LCEWrappedFunction, LCGaussian):
    """Represent a (wrapped) Gaussian peak with linearized energy-
    dependent parameters.

    Parameters
    Norm      :     fraction of photons belonging to peak
    Width     :     the standard deviation parameter of the norm dist.
    Location  :     the mode of the Gaussian distribution

    These parameters are normalized to 1 GeV.  An additional set of
    parameters (included as separate members to not break other
    functionality) give the "slope" of these as a function of
    log10(energy).  Thus, slope=0 implies no energy dependence, while a
    slope of 0.5 implies that, say, the width changes by a factor of 2
    from 0.1 to 1 GeV, etc.
    """

    def init(self):
        super(LCEGaussian, self).init()
        self._einit()
        self.name = "GaussianE"
        self.shortname = "GE"


class LCESkewGaussian(LCEWrappedFunction, LCSkewGaussian):
    """Represent a (wrapped), skew Gaussian peak with linearized energy-
    dependent parameters.

    Parameters
    Norm      :     fraction of photons belonging to peak
    Width     :     the standard deviation parameter of the norm dist.
    Shape     : the degree of skewness, +ve values are right-skewed
    Location  :     the mode of the Gaussian distribution

    These parameters are normalized to 1 GeV.  An additional set of
    parameters (included as separate members to not break other
    functionality) give the "slope" of these as a function of
    log10(energy).  Thus, slope=0 implies no energy dependence, while a
    slope of 0.5 implies that, say, the width changes by a factor of 2
    from 0.1 to 1 GeV, etc.
    """

    def init(self):
        super(LCESkewGaussian, self).init()
        self.name = "SkewGaussianE"
        self.shortname = "GSE"

    def _einit(self):
        super()._einit()
        # broaden boundaries
        self.slope_bounds[1] = [-10, 10]
        # need to allow loc to move more to offset the mode
        self.slope_bounds[2] = [-0.3, 0.3]


class LCELorentzian(LCEWrappedFunction, LCLorentzian):
    """Represent a (wrapped) Lorentzian peak.

    Parameters
    ----------
    Width :
        the width paramater of the wrapped Cauchy distribution, namely HWHM*2PI for narrow distributions
    Location :
        the center of the peak in phase
    """

    def init(self):
        super(LCELorentzian, self).init()
        self.name = "LorentzianE"
        self.shortname = "LE"


class LCELorentzian2(LCEWrappedFunction, LCLorentzian2):
    """Represent a (wrapped) two-sided Lorentzian peak.
    Parameters
    Width1    :  the HWHM of the distribution (left)
    Width2    :  the HWHM of the distribution (right)
    Location  :  the mode of the distribution
    """

    def init(self):
        super(LCELorentzian2, self).init()
        self.name = "Lorentzian2E"
        self.shortname = "L2E"

    def base_int(self, x1, x2, log10_ens, index=0):
        # TODO -- I haven't checked this code
        raise NotImplementedError()
        # abhisrkckl: commented out unreachable code
        # e, gamma1, gamma2, x0 = self._make_p(log10_ens)
        # # the only case where g1 and g2 can be different is if we're on the
        # # 0th wrap, i.e. index=0; this also includes the case when we want
        # # to use base_int to do a "full" integral
        # g1 = np.where((x1 + index) < x0, gamma1, gamma2)
        # g2 = np.where((x2 + index) >= x0, gamma2, gamma1)
        # z1 = (x1 + index - x0) / g1
        # z2 = (x2 + index - x0) / g2
        # k = 2.0 / (gamma1 + gamma2) / PI
        # return k * (g2 * np.arctan(z2) - g1 * np.arctan(z1))

    def random(self, log10_ens):
        """Use multinomial technique to return random photons from
        both components."""
        # TODO -- I haven't checked this code
        raise NotImplementedError()
        # abhisrkckl: commented out unreachable code
        # if not isvector(log10_ens):
        #     n = log10_ens
        #     log10_ens = 3
        # else:
        #     n = len(log10_ens)
        # e, gamma1, gamma2, x0 = self._make_p(log10_ens)  # only change
        # return two_comp_mc(n, gamma1, gamma2, x0, cauchy.rvs)


class LCEGaussian2(LCEWrappedFunction, LCGaussian2):
    """Represent a (wrapped) two-sided Gaussian peak.
    Parameters
    Width1    :  the standard deviation parameter of the norm dist.
    Width2    :  the standard deviation parameter of the norm dist.
    Location  :  the mode of the distribution
    """

    def init(self):
        super(LCEGaussian2, self).init()
        self.name = "Gaussian2E"
        self.shortname = "G2E"

    def base_int(self, x1, x2, log10_ens, index=0):
        # TODO -- I haven't checked this code
        raise NotImplementedError()
        # abhisrkckl: commented out unreachable code
        # e, width1, width2, x0 = self._make_p(log10_ens)
        # w1 = np.where((x1 + index) < x0, width1, width2)
        # w2 = np.where((x2 + index) >= x0, width2, width1)
        # z1 = (x1 + index - x0) / w1
        # z2 = (x2 + index - x0) / w2
        # k1 = 2 * w1 / (width1 + width2)
        # k2 = 2 * w2 / (width1 + width2)
        # return 0.5 * (k2 * erf(z2 / ROOT2) - k1 * erf(z1 / ROOT2))

    def random(self, log10_ens):
        """Use multinomial technique to return random photons from
        both components."""
        # TODO -- I haven't checked this code
        raise NotImplementedError()
        # abhisrkckl: commented out unreachable code
        # if not isvector(log10_ens):
        #     n = log10_ens
        #     log10_ens = 3
        # else:
        #     n = len(log10_ens)
        # e, width1, width2, x0 = self.p
        # return two_comp_mc(n, width1, width2, x0, norm.rvs)


class LCEVonMises(LCEPrimitive, LCVonMises):
    def init(self):
        super(LCEVonMises, self).init()
        self.name = "VonMisesE"
        self.shortname = "VME"

    def gradient(self, phases, log10_ens=3, free=False):
        g = super().gradient
        return edep_gradient(self, g, phases, log10_ens=log10_ens, free=free)
