"""Normalized template representing directional data

Implements a mixture model of LCPrimitives to form a normalized template representing directional data.

author: M. Kerr <matthew.kerr@gmail.com>

"""

import contextlib
import logging
from collections import defaultdict
from copy import deepcopy

import numpy as np

from .lcnorm import NormAngles
from .lcenorm import ENormAngles
from .lceprimitives import *

log = logging.getLogger(__name__)


def isvector(x):
    return len(np.asarray(x).shape) > 0


class LCTemplate:
    """Manage a lightcurve template (collection of LCPrimitive objects).

    IMPORTANT: a constant background is assumed in the overall model,
    so there is no need to furnish this separately.

    Parameters
    ----------
    primitives : list of LCPrimitive
    norms : NormAngles or tuple of float, optional
        If a tuple, they are relative amplitudes for the primitive components.
    """

    def __init__(self, primitives, norms=None, cache_kwargs=None):
        self.primitives = primitives
        self.shift_mode = np.any([p.shift_mode for p in self.primitives])
        if norms is None:
            norms = np.ones(len(primitives)) / len(primitives)
        self.norms = norms if hasattr(norms, "_make_p") else NormAngles(norms)
        self._sanity_checks()
        self._cache = defaultdict(None)
        self._cache_dirty = defaultdict(lambda: True)
        if cache_kwargs is None:
            cache_kwargs = {}
        self.set_cache_properties(**cache_kwargs)

    def __setstate__(self, state):
        # TEMPORARY to handle changed class definition
        self.__dict__.update(state)
        _cache_dirty = defaultdict(lambda: True)
        if not hasattr(self, "_cache_dirty"):
            self._cache = defaultdict(None)
        else:
            # make _cache_dirty a defaultdict from a normal dict
            _cache_dirty.update(self._cache_dirty)
        self._cache_dirty = _cache_dirty
        if not hasattr(self, "ncache"):
            self.ncache = 1000
        if not hasattr(self, "ph_edges"):
            self.ph_edges = np.linspace(0, 1, self.ncache + 1)
        if not hasattr(self, "en_cens"):
            self.en_cens = None
        if not hasattr(self, "en_edges"):
            self.en_edges = None

    def __getstate__(self):
        # transform _cache_dirty into a normal dict, necessary to pickle it
        state = self.__dict__.copy()
        state["_cache_dirty"] = dict(state["_cache_dirty"])
        return state

    def _sanity_checks(self):
        if len(self.primitives) != len(self.norms):
            raise ValueError("Must provide a normalization for each component.")

    def is_energy_dependent(self):
        c1 = np.any([p.is_energy_dependent() for p in self.primitives])
        return c1 or self.norms.is_energy_dependent()

    def has_bridge(self):
        return False

    def __getitem__(self, index):
        if index < 0:
            index += len(self.primitives) + 1
        return self.norms if index == len(self.primitives) else self.primitives[index]

    def __setitem__(self, index, value):
        if index < 0:
            index += len(self.primitives) + 1
        if index == len(self.primitives):
            self.norms = value
        else:
            self.primitives[index] = value

    def __len__(self):
        # sourcery skip: remove-unreachable-code
        raise DeprecationWarning("I'd like to see if this is used.")
        return len(self.primitives)

    def copy(self):
        prims = [deepcopy(x) for x in self.primitives]
        norms = self.norms.copy()
        cache_kwargs = dict(ncache=self.ncache, en_edges=self.en_edges)
        newcopy = self.__class__(prims, norms, cache_kwargs=cache_kwargs)
        for key in self._cache.keys():
            newcopy._cache[key] = self._cache[key]
            newcopy._cache_dirty[key] = self._cache_dirty[key]
        return newcopy

    def set_cache_properties(self, ncache=1000, en_edges=None):
        """Set the granularity and behavior of the cache.

        In all cases, ncache sets the phase resolution.  If it is desired
        to have energy dependence, en_edges must be specified as a set of
        edges in log10 space which fully encompass all possible photon
        energies that wil be used.

        Interpolation is always linear (bilinear in energy if applicable.)
        """
        if hasattr(self, "ncache") and (ncache == self.ncache):
            if (en_edges is None) and (self.en_edges is None):
                return
            elif np.all(en_edges == self.en_edges):
                return
        self.ncache = ncache
        self.ph_edges = np.linspace(0, 1, ncache + 1)
        if en_edges is None:
            self.en_edges = None
            self.en_cens = None
        else:
            en_edges = np.asarray(en_edges)
            if len(en_edges) < 2:
                raise ValueError("len(en_edges) must be >=2.")
            self.en_edges = en_edges
            self.en_cens = 0.5 * (en_edges[1:] + en_edges[:-1])
        self.mark_cache_dirty()

    def mark_cache_dirty(self):
        for k in self._cache_dirty.keys():
            self._cache_dirty[k] = True

    def get_cache(self, order=0):
        if self._cache_dirty[order]:
            self.set_cache(order=order)
        # I don't see how it's possible to have a cache with wrong shape.
        rval = self._cache[order]
        if self.en_edges is not None:
            assert rval.shape[0] == len(self.en_edges)
        assert rval.shape[-1] == (self.ncache + 1)
        return rval

    def set_cache(self, order=0):
        """Populate the cache with values along the bin edges."""
        ncache = self.ncache
        if self.en_edges is None:
            if order == 0:
                new_cache = self(self.ph_edges)
            else:
                new_cache = self.derivative(self.ph_edges, order=order)
        else:
            new_cache = np.empty((len(self.en_edges), len(self.ph_edges)))
            if order == 0:
                for ibin, en in enumerate(self.en_edges):
                    new_cache[ibin] = self(self.ph_edges, log10_ens=en)
            else:
                for ibin, en in enumerate(self.en_edges):
                    new_cache[ibin] = self.derivative(
                        self.ph_edges, log10_ens=en, order=order
                    )
        self._cache[order] = new_cache
        self._cache_dirty[order] = False

    def eval_cache(self, phases, log10_ens=3, order=0):
        """
        Cached values are stored on edges in both phase and, if applicable,
        energy, so lookup is straightforward.
        """
        cache = self.get_cache(order=order)

        dphi = np.atleast_1d(phases) * self.ncache
        phind_lo = dphi.astype(int)
        phind_hi = phind_lo + (phind_lo < self.ncache)  # allows ph==1
        dphi_hi = dphi - phind_lo
        dphi_lo = 1.0 - dphi_hi
        assert np.all(dphi_hi >= 0)
        assert np.all(dphi_hi <= 1)
        assert np.all(dphi_lo >= 0)
        assert np.all(dphi_lo <= 1)

        edges = self.en_edges
        if edges is None:
            return cache[phind_lo] * dphi_lo + cache[phind_hi] * dphi_hi

        de = (log10_ens - edges[0]) / (edges[1] - edges[0])
        eind_lo = de.astype(int)
        eind_hi = eind_lo + 1
        de_hi = de - eind_lo
        de_lo = 1.0 - de_hi
        assert np.all(de_hi >= 0)
        assert np.all(de_hi <= 1)
        assert np.all(de_lo >= 0)
        assert np.all(de_lo <= 1)
        v00 = cache[eind_lo, phind_lo] * (de_lo * dphi_lo)
        v01 = cache[eind_lo, phind_hi] * (de_lo * dphi_hi)
        v10 = cache[eind_hi, phind_lo] * (de_hi * dphi_lo)
        v11 = cache[eind_hi, phind_hi] * (de_hi * dphi_hi)
        return v00 + v01 + v10 + v11

    def set_parameters(self, p, free=True):
        start = 0
        params_ok = True
        for prim in self.primitives:
            n = len(prim.get_parameters(free=free))
            prim.set_parameters(p[start : start + n], free=free)
            start += n
        self.norms.set_parameters(p[start:], free)
        self.mark_cache_dirty()
        return self.check_bounds(free=free)

    def set_errors(self, errs):
        start = 0
        for prim in self.primitives:
            start += prim.set_errors(errs[start:])
        self.norms.set_errors(errs[start:])

    def get_parameters(self, free=True):
        return np.append(
            np.concatenate([prim.get_parameters(free) for prim in self.primitives]),
            self.norms.get_parameters(free),
        )

    def num_parameters(self, free=True):
        """Return the total number of free parameters."""

        nprim = sum((prim.num_parameters(free) for prim in self.primitives))
        return nprim + self.norms.num_parameters(free)

    def _set_all_free_or_fixed(self, freeze=True):
        for prim in self.primitives:
            prim.free[:] = not freeze
        self.norms.free[:] = not freeze

    def free_parameters(self):
        """Free all parameters. Convenience function."""
        self._set_all_free_or_fixed(freeze=False)

    def freeze_parameters(self):
        """Freeze all parameters. Convenience function."""
        self._set_all_free_or_fixed(freeze=True)

    def get_errors(self, free=True):
        return np.append(
            np.concatenate([prim.get_errors(free) for prim in self.primitives]),
            self.norms.get_errors(free),
        )

    def get_free_mask(self):
        """Return a mask with True if parameters are free, else False."""
        m1 = np.concatenate([p.get_free_mask() for p in self.primitives])
        return np.append(m1, self.norms.get_free_mask())

    def get_parameter_names(self, free=True):
        # I will no doubt hate myself in future for below comprehension
        # (or rather lack thereof); this comment will not assuage my rage
        prim_names = [
            "P%d_%s_%s"
            % (
                iprim + 1,
                prim.name[:3] + (prim.name[-1] if prim.name[-1].isdigit() else ""),
                pname[:3] + (pname[-1] if pname[-1].isdigit() else ""),
            )
            for iprim, prim in enumerate(self.primitives)
            for pname in prim.get_parameter_names(free=free)
        ]
        norm_names = [
            "Norm_%s" % pname for pname in self.norms.get_parameter_names(free=free)
        ]
        return prim_names + norm_names
        # return np.append(np.concatenate( [prim.pnames(free) for prim in self.primitives]) , self.norms.get_parameters(free))

    def get_gaussian_prior(self):
        locs, widths, mods, enables = [], [], [], []
        for prim in self.primitives:
            l, w, m, e = prim.get_gauss_prior_parameters()
            locs.append(l)
            widths.append(w)
            mods.append(m)
            enables.append(e)
        t = np.zeros_like(self.norms.get_parameters())
        locs = np.append(np.concatenate(locs), t)
        widths = np.append(np.concatenate(widths), t)
        mods = np.append(np.concatenate(mods), t.astype(bool))
        enables = np.append(np.concatenate(enables), t.astype(bool))
        return GaussianPrior(locs, widths, mods, mask=enables)

    def get_bounds(self, free=True):
        b1 = np.concatenate([prim.get_bounds(free) for prim in self.primitives])
        b2 = self.norms.get_bounds(free)
        return np.concatenate((b1, b2)).tolist()

    def check_bounds(self, free=True):
        bounds = np.asarray(self.get_bounds(free=free))
        x0 = self.get_parameters(free=free)
        return np.all((x0 >= bounds[:, 0]) & (x0 <= bounds[:, 1]))

    def set_overall_phase(self, ph):
        """Put the peak of the first component at phase ph."""
        self.mark_cache_dirty()
        if self.shift_mode:
            self.primitives[0].p[0] = ph
            return
        shift = ph - self.primitives[0].get_location()
        for prim in self.primitives:
            new_location = (prim.get_location() + shift) % 1
            prim.set_location(new_location)

    def get_location(self):
        return self.primitives[0].get_location()

    def get_amplitudes(self, log10_ens=3):
        """Return maximum amplitude of a component."""
        ampls = [p(p.get_location(), log10_ens) for p in self.primitives]
        return self.norms(log10_ens) * np.asarray(ampls)

    def get_code(self):
        """Return a short string encoding the components in the template."""
        return "/".join((p.shortname for p in self.primitives))

    def norm(self):
        return self.norms.get_total()

    def norm_ok(self):
        return self.norm() <= 1

    def integrate(self, phi1, phi2, log10_ens=3, suppress_bg=False):
        """Integrate profile from phi1 to phi2.

        NB that because of the phase modulo ambiguity, it is not uniquely
        definite what the phi2 < phi1 case means:
            integral(0.8,0.2) == -integral(0.2,0.8)
            integral(0.8,1.2) == 1-integral(0.2,0.8)

        To break the ambiguity, we support non-modulo phase here, so you
        can just write integral(0.8,1.2) if that's what you mean.
        """
        phi1 = np.asarray(phi1)
        phi2 = np.asarray(phi2)
        if isvector(log10_ens):
            assert len(log10_ens) == len(phi1)
        with contextlib.suppress(TypeError):
            assert len(phi1) == len(phi2)
        norms = self.norms(log10_ens=log10_ens)
        t = norms.sum(axis=0)
        dphi = phi2 - phi1
        rvals = np.zeros(phi1.shape, dtype=float)
        for n, prim in zip(norms, self.primitives):
            rvals += n * prim.integrate(phi1, phi2, log10_ens=log10_ens)
        return rvals * (1.0 / t) if suppress_bg else (1 - t) * dphi + rvals

    def cdf(self, x, log10_ens=3):
        return self.integrate(np.zeros_like(x), x, log10_ens, suppress_bg=False)

    def max(self, resolution=0.01):
        return self(np.arange(0, 1, resolution)).max()

    def _get_scales(self, phases, log10_ens=3):
        """Method to allow abstraction for setting amplitudes for each
        peak.  Trivial in typical cases, but important for linked
        components, e.g. the bridge pedestal.
        """
        rvals = np.zeros(np.asarray(phases).shape, dtype=float)
        norms = self.norms(log10_ens)
        return rvals, norms, norms.sum(axis=0)

    def __call__(self, phases, log10_ens=3, suppress_bg=False, use_cache=False):
        """Evaluate template at the provided phases and (if provided)
        energies.  If "suppress_bg" is set, ignore the DC component."""
        # TMP -- check phase range.  Add this as a formal check?
        phases = np.asarray(phases)
        log10_ens = np.asarray(log10_ens)
        assert np.all(phases >= 0)
        assert np.all(phases <= 1)
        # end TM
        if use_cache:
            return self.eval_cache(phases, log10_ens=log10_ens, order=0)
        rvals, norms, norm = self._get_scales(phases, log10_ens)
        for n, prim in zip(norms, self.primitives):
            rvals += n * prim(phases, log10_ens=log10_ens)
        return rvals / norm if suppress_bg else (1.0 - norm) + rvals

    def derivative(self, phases, log10_ens=3, order=1, use_cache=False):
        """Return the derivative of the template with respect to pulse
        phase (as opposed to the gradient of the template with respect to
        some of the template parameters)."""

        if use_cache:
            return self.eval_cache(phases, order=order)
        rvals = np.zeros_like(phases)
        norms = self.norms(log10_ens=log10_ens)
        for n, prim in zip(norms, self.primitives):
            rvals += n * prim.derivative(phases, log10_ens=log10_ens, order=order)
        return rvals

    def single_component(self, index, phases, log10_ens=3, add_bg=False):
        """Evaluate a single component of template."""
        n = self.norms(log10_ens=log10_ens)
        rvals = self.primitives[index](phases, log10_ens=log10_ens) * n[index]
        return rvals + n.sum(axis=0) if add_bg else rvals

    def gradient(self, phases, log10_ens=3, free=True, template_too=False):
        r = np.empty((self.num_parameters(free), len(phases)))
        c = 0
        rvals, norms, norm = self._get_scales(phases, log10_ens=log10_ens)
        prim_terms = np.empty((len(self.primitives), len(phases)))
        for i, (nm, prim) in enumerate(zip(norms, self.primitives)):
            n = len(prim.get_parameters(free=free))
            r[c : c + n, :] = nm * prim.gradient(phases, log10_ens=log10_ens, free=free)
            c += n
            prim_terms[i] = prim(phases, log10_ens=log10_ens) - 1

        # handle case where no norm parameters are free
        if c != r.shape[0]:
            # "prim_terms" are df/dn_i and have shape nnorm x nphase
            # the output of gradient is the matrix M_ij or M_ijk, depending
            # on whether or not there is energy dependence, which is
            # dnorm_i/dnorm_angle_j (at energy k).  The sum is over the
            # "internal parameter" norm_j, while the phase axis and
            # norm_angle axis are retained.
            m = self.norms.gradient(log10_ens=log10_ens, free=free)
            if len(m.shape) == 2:
                m = m[..., None]
            np.einsum("ij,ikj->kj", prim_terms, m, out=r[c:])
        if template_too:
            rvals[:] = 1 - norm
            for i in range(len(prim_terms)):
                rvals += (prim_terms[i] + 1) * norms[i]
            return r, rvals
        return r

    def gradient_derivative(self, phases, log10_ens=3, free=False):
        """Return d/dphi(gradient).  This is the derivative with respect
        to pulse phase of the gradient with respect to the parameters.
        """
        # sourcery skip: remove-unreachable-code
        raise NotImplementedError()  # is this used anymore?
        free_mask = self.get_free_mask()
        nparam = len(free_mask)
        nnorm_param = len(self.norms.p)
        nprim_param = nparam - nnorm_param
        rvals = np.empty([nparam, len(phases)])
        prim_terms = np.empty([len(self.primitives), len(phases)])
        norms = self.norms()
        c = 0
        for iprim, prim in enumerate(self.primitives):
            n = len(prim.p)
            rvals[c : c + n] = norms[iprim] * prim.gradient_derivative(phases)
            prim_terms[iprim] = prim.derivative(phases)
            c += n

        norm_grads = self.norms.gradient(phases, free=False)
        for j in range(nnorm_param):
            rvals[nprim_param + j] = 0
            for i in range(nnorm_param):
                rvals[nprim_param + j] += norm_grads[i, j] * prim_terms[i]
        return rvals

    def approx_gradient(self, phases, log10_ens=None, eps=1e-5):
        return approx_gradient(self, phases, log10_ens=log10_ens, eps=eps)

    def approx_hessian(self, phases, log10_ens=None, eps=1e-5):
        return approx_hessian(self, phases, log10_ens=log10_ens, eps=eps)

    def approx_derivative(self, phases, log10_ens=None, order=1, eps=1e-7):
        return approx_derivative(
            self, phases, log10_ens=log10_ens, order=order, eps=eps
        )

    def check_gradient(
        self, atol=1e-7, rtol=1e-5, quiet=False, seed=None, en=None, ph=None
    ):
        if seed is not None:
            # essentially set a known good state of the RNG to avoid any
            # numerical issues interfering with the test
            np.random.seed(seed)
        return check_gradient(self, atol=atol, rtol=rtol, quiet=quiet, en=en, ph=ph)

    def check_derivative(self, atol=1e-7, rtol=1e-5, order=1, eps=1e-7, quiet=False):
        return check_derivative(
            self, atol=atol, rtol=rtol, quiet=quiet, eps=1e-7, order=order
        )

    def hessian(self, phases, log10_ens=3, free=True):
        """Return the hessian of the primitive and normalization angles.

        The primitives components are not coupled due to the additive form
        of the template.  However, because each normalization depends on
        multiple hyper angles, there is in general coupling between the
        normalization components and the primitive components.  The
        general form of the terms is

        (1) block diagonal hessian terms from primitive
        (2 ) for the unmixed derivative of the norms, the sum of the
        hessian of the hyper angles over the primitive terms
        (3) for mixed derivatives, the product gradient of the norm

        In general, this is pretty complicated if some parameters are free
        and some are not, and (currently) this method isn't used in
        fitting, so for ease of implementation, simply evaluate the whole
        hessian, then return only the relevant parts for the free
        parameters.

        """

        free_mask = self.get_free_mask()
        nparam = len(free_mask)
        nnorm_param = self.norms.num_parameters()
        nprim_param = nparam - nnorm_param
        r = np.zeros([nparam, nparam, len(phases)])

        norms = self.norms(log10_ens=log10_ens)
        norm_grads = self.norms.gradient(log10_ens=log10_ens, free=False)
        prim_terms = np.empty([len(self.primitives), len(phases)])

        c = 0
        for i, prim in enumerate(self.primitives):
            h = prim.hessian(phases, log10_ens=log10_ens, free=False)
            pg = prim.gradient(phases, log10_ens=log10_ens, free=False)
            n = len(prim.p)
            # put hessian in diagonal elements
            r[c : c + n, c : c + n, :] = norms[i] * h
            # put cross-terms with normalization; although only one primitive
            # survives in the second derivative, all of the normalization angles
            # feature
            for j in range(n):
                for k in range(nnorm_param):
                    r[nprim_param + k, c + j, :] = pg[j] * norm_grads[i, k]
                    r[c + j, nprim_param + k, :] = r[nprim_param + k, c + j, :]
            prim_terms[i, :] = prim(phases, log10_ens=log10_ens) - 1
            c += n

        # now put in normalization hessian
        hnorm = self.norms.hessian(
            log10_ens=log10_ens
        )  # nnorm_param x nnorm_param x nnorm_param
        for j in range(nnorm_param):
            for k in range(j, nnorm_param):
                for i in range(nnorm_param):
                    r[c + j, c + k, :] += hnorm[i, j, k] * prim_terms[i]
                r[c + k, c + j, :] = r[c + j, c + k, :]

        return r[free_mask][:, free_mask] if free else r

    def delta(self, index=None):
        """Return radio lag -- reckoned by default as the position of the            first peak following phase 0."""
        if (index is not None) and (index <= (len(self.primitives))):
            return self[index].get_location(error=True)
        return self.Delta(delta=True)

    def Delta(self, delta=False):
        """Report peak separation -- reckoned by default as the distance
        between the first and final component locations.

        delta [False] -- if True, return the first peak position"""
        if len(self.primitives) == 1:
            return -1, 0
        prim0, prim1 = self.primitives[0], self.primitives[-1]
        for p in self.primitives:
            if p.get_location() < prim0.get_location():
                prim0 = p
            if p.get_location() > prim1.get_location():
                prim1 = p
        p1, e1 = prim0.get_location(error=True)
        p2, e2 = prim1.get_location(error=True)
        return (p1, e1) if delta else (p2 - p1, (e1**2 + e2**2) ** 0.5)

    def _sorted_prims(self):
        def cmp(p1, p2):
            if p1.p[-1] < p2.p[-1]:
                return -1
            elif p1.p[-1] == p2.p[-1]:
                return 0
            else:
                return 1

        return sorted(self.primitives, cmp=cmp)

    def __str__(self):
        prims = self.primitives
        s0 = str(self.norms)
        s1 = (
            "\n\n"
            + "\n\n".join(
                ["P%d -- " % (i + 1) + str(prim) for i, prim in enumerate(prims)]
            )
            + "\n"
        )
        if len(prims) > 1:
            s1 += "\ndelta   : %.4f +\\- %.4f" % self.delta()
            s1 += "\nDelta   : %.4f +\\- %.4f" % self.Delta()
        return s0 + s1

    def prof_string(self, outputfile=None):
        """Return a string compatible with the format used by pygaussfit.
        Assume all primitives are gaussians."""
        rstrings = []
        dashes = "-" * 25
        norm, errnorm = 0, 0

        for nprim, prim in enumerate(self.primitives):
            phas = prim.get_location(error=True)
            fwhm = 2 * prim.get_width(error=True, hwhm=True)
            ampl = [self.norms()[nprim], 0]
            norm += ampl[0]
            errnorm += ampl[1] ** 2
            for st, va in zip(["phas", "fwhm", "ampl"], [phas, fwhm, ampl]):
                rstrings += ["%s%d = %.5f +/- %.5f" % (st, nprim + 1, va[0], va[1])]
        const = "const = %.5f +/- %.5f" % (1 - norm, errnorm**0.5)
        rstring = [dashes] + [const] + rstrings + [dashes]
        if outputfile is not None:
            f = open(outputfile, "w")
            f.write("# gauss\n")
            for s in rstring:
                f.write(s + "\n")
        return "\n".join(rstring)

    def random(self, n, weights=None, log10_ens=3, return_partition=False):
        """Return n pseudo-random variables drawn from the distribution
        given by this light curve template.

        For simplicity, if weights are not provided, unit weights are
        assumed.  If energies are not provided, a vector of 1 GeV (3)
        is assumed.

        Next, optionally using the weights and the energy vectors, the
        probability for each realization to arise from the primitives or
        the background is determined.  Those probabilities are used in a
        multinomial to determine which component will generate each photon,
        and finally using that partition the correct number of phases are
        simulated from each component.

        Weights ("w") are interpreted as the probability to originate from
        the source, which includes the DC component, so the total prob. to
        be DC is (1-w) (background) + w*sum_prims (unpulsed).
        """

        n = int(round(n))

        if len(self.primitives) == 0:
            return (np.random.rand(n), [n]) if return_partition else np.random.rand(n)

        # check weights
        if weights is None:
            weights = np.ones(n)
        elif len(weights) != n:
            raise ValueError("Provided weight vector does not match requested n.")

        # check energies
        if isvector(log10_ens):
            if len(log10_ens) != n:
                raise ValueError(
                    "Provided log10_ens vector does not match requested n."
                )
        else:
            log10_ens = np.full(n, log10_ens)

        # first, calculate the energy dependent norm of each vector
        norms = self.norms(log10_ens=log10_ens)  # nnorm x nen array
        N = norms.sum(axis=0)
        nDC = weights * N
        pDC = 1 - nDC
        partition_probs = np.append(norms / N * nDC, pDC[None, :], axis=0)
        # now, draw a component for each bit of the partition
        cpp = np.cumsum(partition_probs, axis=0)
        assert np.allclose(cpp[-1], 1)
        comps = np.full(n, len(self.primitives))
        Q = np.random.rand(n)
        for i in np.arange(len(self.primitives))[::-1]:
            mask = Q < cpp[i]
            comps[mask] = i
        total = 0
        rvals = np.empty(n)
        rvals[:] = np.nan  # TMP

        total = 0
        for iprim, prim in enumerate(self.primitives):
            mask = comps == iprim
            total += mask.sum()
            rvals[mask] = prim.random(mask.sum(), log10_ens=log10_ens[mask])

        # DC component
        mask = comps == len(self.primitives)
        total += mask.sum()
        rvals[mask] = np.random.rand(mask.sum())

        assert not np.any(np.isnan(rvals))  # TMP

        return (rvals, comps) if return_partition else rvals

    def swap_primitive(self, index, ptype=LCLorentzian):
        """Swap the specified primitive for a new one with the parameters
        that match the old one as closely as possible."""
        self.primitives[index] = convert_primitive(self.primitives[index], ptype)

    def delete_primitive(self, index, inplace=False):
        """Return a new LCTemplate with the primitive at index removed.

        The flux is renormalized to preserve the same pulsed ratio (in the
        case of an energy-dependent template, at the pivot energy).
        """
        norms, prims = self.norms, self.primitives
        if len(prims) == 1:
            raise ValueError("Template only has a single primitive.")
        if index < 0:
            index += len(prims)
        newprims = [deepcopy(p) for ip, p in enumerate(prims) if index != ip]
        newnorms = self.norms.delete_component(index)
        if not inplace:
            return LCTemplate(newprims, newnorms)
        self.primitives = newprims
        self.norms = newnorms

    def add_primitive(self, prim, norm=0.1, inplace=False):
        """[Convenience] -- return a new LCTemplate with the specified
        LCPrimitive added and renormalized."""
        norms, prims = self.norms, self.primitives
        if len(prims) == 0:
            # special case of an empty profile
            return LCTemplate([prim], [1])
        nprims = [deepcopy(prims[i]) for i in range(len(prims))] + [prim]
        nnorms = self.norms.add_component(norm)
        if not inplace:
            return LCTemplate(nprims, nnorms)
        self.norms = nnorms
        self.primitives = nprims

    def order_primitives(self, order=0):
        """Re-order components in place.

        order == 0: order by ascending position
        order == 1: order by descending maximum amplitude
        order == 2: order by descending normalization
        """
        if order == 0:
            indices = np.argsort([p.get_location() for p in self.primitives])
        elif order == 1:
            indices = np.argsort(self.get_amplitudes())[::-1]
        elif order == 2:
            indices = np.argsort(self.norms())[::-1]
        else:
            raise NotImplementedError("Specified order not supported.")
        self.primitives = [self.primitives[i] for i in indices]
        self.norms.reorder_components(indices)

    def get_fixed_energy_version(self, log10_en=3):
        return self

    def add_energy_dependence(self, index, slope_free=True):
        comp = self[index]
        if comp.is_energy_dependent():
            return
        if comp.name == "NormAngles":
            # normalization
            newcomp = ENormAngles(self.norms())
        else:
            # primitive
            if comp.name == "Gaussian":
                constructor = LCEGaussian
            elif comp.name == "VonMises":
                constructor = LCEVonMises
            else:
                raise NotImplementedError(f"{comp.name} not supported.")
            newcomp = constructor(p=comp.p)
        newcomp.free[:] = comp.free
        newcomp.slope_free[:] = slope_free
        self[index] = newcomp

    def get_eval_string(self):
        """Return a string that can be "eval"ed to make a cloned set of
        primitives and template."""
        ps = "\n".join(
            ("p%d = %s" % (i, p.eval_string()) for i, p in enumerate(self.primitives))
        )
        prims = f'[{",".join("p%d" % i for i in range(len(self.primitives)))}]'
        ns = f"norms = {self.norms.eval_string()}"
        return f"{self.__class__.__name__}({prims},norms)"

    def closest_to_peak(self, phases):
        return min((p.closest_to_peak(phases) for p in self.primitives))

    def mean_value(self, phases, log10_ens=None, weights=None, bins=20):
        """Compute the mean value of the profile over the codomain of
        phases.  Mean is taken over energy and is unweighted unless
        a set of weights are provided."""
        if (log10_ens is None) or (not self.is_energy_dependent()):
            return self(phases)
        if weights is None:
            weights = np.ones_like(log10_ens)
        edges = np.linspace(log10_ens.min(), log10_ens.max(), bins + 1)
        w = np.histogram(log10_ens, weights=weights, bins=edges)
        rvals = np.zeros_like(phases)
        for weight, en in zip(w[0], (edges[:-1] + edges[1:]) / 2):
            rvals += weight * self(phases, en)
        rvals /= w[0].sum()
        return rvals

    def mean_single_component(
        self, index, phases, log10_ens=None, weights=None, bins=20, add_pedestal=True
    ):
        prim = self.primitives[index]
        if (log10_ens is None) or (not self.is_energy_dependent()):
            n = self.norms()
            return prim(phases) * n[index] + add_pedestal * (1 - n.sum())
        if weights is None:
            weights = np.ones_like(log10_ens)
        edges = np.linspace(log10_ens.min(), log10_ens.max(), bins + 1)
        w = np.histogram(log10_ens, weights=weights, bins=edges)
        rvals = np.zeros_like(phases)
        for weight, en in zip(w[0], (edges[:-1] + edges[1:]) / 2):
            rvals += weight * prim(phases, en) * self.norms(en)[index]
        rvals /= w[0].sum()
        return rvals

    def rotate(self, dphi):
        """Adjust the template by dphi."""
        self.mark_cache_dirty()
        log.info(f"Shifting template by {dphi}.")
        for prim in self.primitives:
            new_location = (prim.get_location() + dphi) % 1
            prim.set_location(new_location)

    def get_display_point(self, do_rotate=False):
        # TODO -- need to fix this to scan all the way around, either
        # from -0.5 to 0 or from 0.5 to 1.0, whichever -- see J0102
        """Return phase shift which optimizes the display of the profile.

        This is determined by finding the 60% window which contains the
        most flux and returning the left edge.  Rotating the profile such
        that this edge is at phi=0.20 would then center this interval, so
        the resulting phase shift would do that.
        """
        N = 50
        dom = np.linspace(0, 1, 2 * N + 1)[:-1]
        cod = self.integrate(dom, dom + 0.6)
        dphi = 0.20 - dom[np.argmax(cod)]
        if do_rotate:
            self.rotate(dphi)
        return dphi

    def write_profile(self, fname, nbin, integral=False, suppress_bg=False):
        """Write out a two-column tabular profile to file fname.

        The first column indicates the left edge of the phase bin, while
        the right column indicates the profile value.

        Parameters
        ----------
        integral : bool
            if True, integrate the profile over the bins.  Otherwise, differential
            value at indicated bin phase.
        suppress_bg : bool
            if True, do not include the unpulsed (DC) component

        """

        if not integral:
            bin_phases = np.linspace(0, 1, nbin + 1)[:-1]
            bin_values = self(bin_phases, suppress_bg=suppress_bg)
            bin_values *= 1.0 / bin_values.mean()

        else:
            phases = np.linspace(0, 1, 2 * nbin + 1)
            values = self(phases, suppress_bg=suppress_bg)
            hi = values[2::2]
            lo = values[:-1:2]
            mid = values[1::2]
            bin_phases = phases[:-1:2]
            bin_values = 1.0 / (6 * nbin) * (hi + 4 * mid + lo)

        bin_values *= 1.0 / bin_values.mean()
        open(fname, "w").write(
            "".join(("%.6f %.6f\n" % (x, y) for x, y in zip(bin_phases, bin_values)))
        )


def get_gauss2(
    pulse_frac=1,
    x1=0.1,
    x2=0.55,
    ratio=1.5,
    width1=0.01,
    width2=0.02,
    lorentzian=False,
    bridge_frac=0,
    skew=False,
):
    """Return a two-gaussian template.  Convenience function."""
    n1, n2 = np.asarray([ratio, 1.0]) * (1 - bridge_frac) * (pulse_frac / (1.0 + ratio))
    if skew:
        prim = LCLorentzian2 if lorentzian else LCGaussian2
        p1, p2 = [width1, width1 * (1 + skew), x1], [width2 * (1 + skew), width2, x2]
    else:
        if lorentzian:
            prim = LCLorentzian
            width1 *= 2 * np.pi
            width2 *= 2 * np.pi
        else:
            prim = LCGaussian
        p1, p2 = [width1, x1], [width2, x2]
    if bridge_frac > 0:
        nb = bridge_frac * pulse_frac
        b = LCGaussian(p=[0.1, (x2 + x1) / 2])
        return LCTemplate([prim(p=p1), b, prim(p=p2)], [n1, nb, n2])
    return LCTemplate([prim(p=p1), prim(p=p2)], [n1, n2])


def get_gauss1(pulse_frac=1, x1=0.5, width1=0.01):
    """Return a one-gaussian template.  Convenience function."""
    return LCTemplate([LCGaussian(p=[width1, x1])], [pulse_frac])


def get_2pb(pulse_frac=0.9, lorentzian=False):
    """Convenience function to get a 2 Lorentzian + Gaussian bridge template."""
    prim = LCLorentzian if lorentzian else LCGaussian
    p1 = prim(p=[0.03, 0.1])
    b = LCGaussian(p=[0.15, 0.3])
    p2 = prim(p=[0.03, 0.55])
    return LCTemplate(
        primitives=[p1, b, p2],
        norms=[0.3 * pulse_frac, 0.4 * pulse_frac, 0.3 * pulse_frac],
    )


def make_twoside_gaussian(one_side_gaussian):
    """Make a two-sided gaussian with the same initial shape as the
    input one-sided gaussian."""
    g2 = LCGaussian2()
    g1 = one_side_gaussian
    g2.p[0] = g2.p[1] = g1.p[0]
    g2.p[-1] = g1.p[-1]
    return g2


def adaptive_samples(func, npt, log10_ens=3, nres=200):
    """func should have a .cdf method.

    NB -- log10_ens needs to be a scalar!

    The idea is to return a set of points on [0,1] which are approximately
    distributed uniformly in F(phi) and thus more densely sample the
    peaks.  First, the cdf is evaluated on nres points.  Then npt estimates
    of the inverse cdf are obtained by linear interpolation.
    """
    assert np.isscalar(log10_ens)
    x = np.linspace(0, 1, nres)
    F = func.cdf(x, log10_ens=log10_ens)
    Y = np.linspace(0, 1, npt)
    idx = np.searchsorted(F, Y[1:-1])
    assert idx.min() > 0
    F1 = F[idx]
    F0 = F[idx - 1]
    m = (F1 - F0) * (1.0 / (x[1] - x[0]))
    X1 = x[idx]
    X0 = x[idx - 1]
    Y[1:-1] = X0 + (Y[1:-1] - F0) / (F1 - F0) * (x[1] - x[0])
    return Y


class GaussianPrior:
    def __init__(self, locations, widths, mod, mask=None):
        self.x0 = np.where(mod, np.mod(locations, 1), locations)
        self.s0 = np.asarray(widths) * 2**0.5
        self.mod = np.asarray(mod)
        if mask is None:
            self.mask = np.asarray([True] * len(locations))
        else:
            self.mask = np.asarray(mask)
            self.x0 = self.x0[self.mask]
            self.s0 = self.s0[self.mask]
            self.mod = self.mod[self.mask]

    def __len__(self):
        """Return number of parameters with a prior."""
        return self.mask.sum()

    def __call__(self, parameters):
        if not np.any(self.mask):
            return 0
        parameters = parameters[self.mask]
        parameters = np.where(self.mod, np.mod(parameters, 1), parameters)
        return np.sum(((parameters - self.x0) / self.s0) ** 2)

    def gradient(self, parameters):
        if not np.any(self.mask):
            return np.zeros_like(parameters)
        parameters = parameters[self.mask]
        parameters = np.where(self.mod, np.mod(parameters, 1), parameters)
        rvals = np.zeros(len(self.mask))
        rvals[self.mask] = 2 * (parameters - self.x0) / self.s0**2
        return rvals


def prim_io(template, bound_eps=1e-5):
    """Read files and build LCPrimitives."""

    def read_gaussian(toks):
        primitives = []
        norms = []
        for i, tok in enumerate(toks):
            if tok[0].startswith("phas"):
                g = LCGaussian()
                g.p[-1] = float(tok[2])
                g.errors[-1] = float(tok[4])
                primitives += [g]
            elif tok[0].startswith("fwhm"):
                g = primitives[-1]
                g.p[0] = float(tok[2]) / 2.3548200450309493  # kluge for now
                g.errors[0] = float(tok[4]) / 2.3548200450309493
            elif tok[0].startswith("ampl"):
                norms.append(float(tok[2]))
        # check to that bounds are OK
        for iprim, prim in enumerate(primitives):
            if prim.check_bounds():
                continue
            for ip, p in enumerate(prim.p):
                lo, hi = prim.bounds[ip]
                if (p < lo) and (abs(p - lo) < bound_eps):
                    prim.p[ip] = lo
                if (p > hi) and (abs(p - hi) < bound_eps):
                    prim.p[ip] = hi
            if not prim.check_bounds():
                raise ValueError("Unrecoverable bounds errors on input.")
        # check norms
        norms = np.asarray(norms)
        n = norms.sum()
        if (n > 1) and (abs(n - 1) < bounds_eps):
            norms *= 1.0 / n
        return primitives, list(norms)

    lines = None
    try:
        with open(template, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = template.split("\n")
    if lines is None:
        raise ValueError("Could not load lines from template.")
    toks = [line.strip().split() for line in lines if len(line.strip()) > 0]
    label, toks = toks[0], toks[1:]
    if "gauss" in label:
        return read_gaussian(toks)
    elif "kernel" in label:
        return [LCKernelDensity(input_file=toks)], None
    elif "fourier" in label:
        return [LCEmpiricalFourier(input_file=toks)], None
    raise ValueError("Template format not recognized!")


def check_gradient_derivative(templ):
    dom = np.linspace(0, 1, 10001)
    pcs = 0.5 * (dom[:-1] + dom[1:])
    ngd = templ.gradient(dom)
    ngd = (ngd[:, 1:] - ngd[:, :-1]) / (dom[1] - dom[0])
    gd = templ.gradient_derivative(templ, pcs)
    for i in range(gd.shape[0]):
        print(np.max(np.abs(gd[i] - ngd[i])))
    return pcs, gd, ngd


def isvector(x):
    return len(np.asarray(x).shape) > 0
