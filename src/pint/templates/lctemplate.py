"""Normalized template representing directional data

Implements a mixture model of LCPrimitives to form a normalized template representing directional data.

author: M. Kerr <matthew.kerr@gmail.com>

"""
from __future__ import absolute_import, division, print_function

from collections import defaultdict
from copy import deepcopy

import numpy as np
from astropy import log

from .lcnorm import NormAngles
from .lcprimitives import *


class LCTemplate(object):
    """Manage a lightcurve template (collection of LCPrimitive objects).

    IMPORTANT: a constant background is assumed in the overall model,
    so there is no need to furnish this separately.

    Parameters
    ----------
    primitives : list of LCPrimitive
    norms : NormAngles or tuple of float, optional
        If a tuple, they are relative amplitudes for the primitive components.
    """

    def __init__(self, primitives, norms=None):
        self.primitives = primitives
        self.shift_mode = np.any([p.shift_mode for p in self.primitives])
        if norms is None:
            norms = np.ones(len(primitives)) / len(primitives)
        self.norms = norms if isinstance(norms, NormAngles) else NormAngles(norms)
        self._sanity_checks()
        self._cache = defaultdict(None)
        self._cache_dirty = defaultdict(lambda: True)
        self.set_cache_properties()

    def __setstate__(self, state):
        # TEMPORARY to handle changed class definition
        self.__dict__.update(state)
        _cache_dirty = defaultdict(lambda: True)
        if not hasattr(self, "_cache_dirty"):
            self._cache = defaultdict(None)
            self._cache_dirty = _cache_dirty
        else:
            # make _cache_dirty a defaultdict from a normal dict
            _cache_dirty.update(self._cache_dirty)
            self._cache_dirty = _cache_dirty
        if not hasattr(self, "ncache"):
            self.ncache = 1000
            self.interpolation = 1

    def __getstate__(self):
        # transform _cache_dirty into a normal dict, necessary to pickle it
        self._cache_dirty = dict(self._cache_dirty)
        return self.__dict__

    def _sanity_checks(self):
        if len(self.primitives) != len(self.norms):
            raise ValueError("Must provide a normalization for each component.")

    def is_energy_dependent(self):
        return False

    def has_bridge(self):
        return False

    def __getitem__(self, index):
        return self.primitives[index]

    def __setitem__(self, index, value):
        self.primitives[index] = value

    def __len__(self):
        return len(self.primitives)

    def copy(self):
        prims = [deepcopy(x) for x in self.primitives]
        return self.__class__(prims, self.norms.copy())

    def set_cache_properties(self, ncache=1000, interpolation=1):
        self.ncache = ncache
        self.interpolation = interpolation
        self.mark_cache_dirty()

    def mark_cache_dirty(self):
        for k in self._cache_dirty.keys():
            self._cache_dirty[k] = True

    def get_cache(self, order=0):
        ncache = self.ncache
        if self._cache_dirty[order]:
            self.set_cache(order=order)
        if len(self._cache[order]) != ncache + 1:
            self.set_cache(order=order)
        return self._cache[order]

    def set_cache(self, order=0):
        """ Populate the cache with *point* values.

        Previous implementation used an average (poor person's integration)
        but I think it makes more sense to use points and interpolation as
        necessary.
        """
        ncache = self.ncache
        if order == 0:
            t = self(np.linspace(0, 1, ncache + 1))
            # self._cache[0] = 0.5*(t[1:]+t[:-1])
            # NB store the wrapped values to make evaluation faster
            self._cache[0] = t
            self._cache_dirty[0] = False
        else:
            t = self.derivative(np.linspace(0, 1, ncache + 1), order=order)
            # self._cache[order] = 0.5*(t[1:]+t[:-1])
            self._cache[order] = t
            self._cache_dirty[order] = False

    def eval_cache(self, phases, order=0):
        # NB, cached values are stored on a grid of points from [0..1).  In
        # order to find the closest point, add half a bin before floor

        ncache = self.ncache
        interpolation = self.interpolation
        cached_values = self.get_cache(order=order)

        if interpolation == 0:
            indices = np.array(phases * ncache + 0.5, dtype=int)
            try:
                return cached_values[indices]
            except IndexError as e:
                nanphases = np.sum(np.isnan(phases))
                if nanphases > 0:
                    print("%d phases were NaN!" % (nanphases))
                    indices[np.isnan(phases)] = 0
                    return cached_values[indices]
                else:
                    raise e

        if interpolation == 1:
            indices = phases * ncache
            indices_lo = indices.astype(int)
            dhi = indices - indices_lo
            dlo = 1.0 - dhi
            return cached_values[indices_lo] * dlo + cached_values[indices_lo + 1] * dhi

        else:
            raise NotImplementedError(
                "interpolation=%d not implemented" % (interpolation)
            )

    def set_parameters(self, p, free=True):
        start = 0
        params_ok = True
        for prim in self.primitives:
            n = len(prim.get_parameters(free=free))
            params_ok = (
                prim.set_parameters(p[start : start + n], free=free) and params_ok
            )
            start += n
        self.norms.set_parameters(p[start:], free)
        self.mark_cache_dirty()
        return params_ok

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

    def get_errors(self, free=True):
        return np.append(
            np.concatenate([prim.get_errors(free) for prim in self.primitives]),
            self.norms.get_errors(free),
        )

    def get_free_mask(self):
        """ Return a mask with True if parameters are free, else False.
        """
        return np.append(
            np.concatenate([prim.free for prim in self.primitives]), self.norms.free
        )

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

    def get_bounds(self):
        b1 = np.concatenate([prim.get_bounds() for prim in self.primitives])
        b2 = self.norms.get_bounds()
        return np.concatenate((b1, b2)).tolist()

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
        """ Return maximum amplitude of a component."""
        ampls = [p(p.get_location(), log10_ens) for p in self.primitives]
        return self.norms(log10_ens) * np.asarray(ampls)

    def get_code(self):
        """ Return a short string encoding the components in the template."""
        return "/".join((p.shortname for p in self.primitives))

    def norm(self):
        return self.norms.get_total()

    def integrate(self, phi1, phi2, log10_ens=3, suppress_bg=False):
        norms = self.norms(log10_ens)
        t = norms.sum(axis=0)
        dphi = phi2 - phi1
        rvals = np.zeros_like(t)
        for n, prim in zip(norms, self.primitives):
            rvals += n * prim.integrate(phi1, phi2, log10_ens)
        rvals.sum(axis=0)
        if suppress_bg:
            return rvals / t
        return (1 - t) * dphi + rvals

    def cdf(self, x, log10_ens=3):
        return self.integrate(0, x, log10_ens, suppress_bg=False)

    def max(self, resolution=0.01):
        return self(np.arange(0, 1, resolution)).max()

    def _get_scales(self, phases, log10_ens=3):
        """ Method to allow abstraction for setting amplitudes for each
            peak.  Trivial in typical cases, but important for linked
            components, e.g. the bridge pedestal.
        """
        rvals = np.zeros_like(phases)
        norms = self.norms(log10_ens)
        return rvals, norms, norms.sum(axis=0)

    def __call__(self, phases, log10_ens=3, suppress_bg=False, use_cache=False):
        """ Evaluate template at the provided phases and (if provided)
            energies.  If "suppress_bg" is set, ignore the DC component."""
        if use_cache:
            return self.eval_cache(phases, order=0)
        rvals, norms, norm = self._get_scales(phases, log10_ens)
        for n, prim in zip(norms, self.primitives):
            rvals += n * prim(phases, log10_ens)
        if suppress_bg:
            return rvals / norm
        return (1.0 - norm) + rvals

    def derivative(self, phases, log10_ens=3, order=1, use_cache=False):
        """ Return the derivative of the template with respect to pulse
        phase (as opposed to the gradient of the template with respect to
        some of the template parameters)."""

        if use_cache:
            return self.eval_cache(phases, order=order)
        rvals = np.zeros_like(phases)
        norms = self.norms()
        for n, prim in zip(norms, self.primitives):
            rvals += n * prim.derivative(phases, log10_ens, order=order)
        return rvals

    def single_component(self, index, phases, log10_ens=3):
        """ Evaluate a single component of template."""
        n = self.norms(log10_ens)[index]
        return self.primitives[index](phases, log10_ens) * n

    def single_component(self, index, phases, log10_ens=3, add_bg=False):
        """ Evaluate a single component of template."""
        n = self.norms(log10_ens)
        rvals = self.primitives[index](phases, log10_ens) * n[index]
        if add_bg:
            return rvals + n.sum(axis=0)
        return rvals

    def gradient(self, phases, log10_ens=3, free=True):
        r = np.empty([len(self.get_parameters(free=free)), len(phases)])
        c = 0
        norms = self.norms()
        prim_terms = np.empty([len(phases), len(self.primitives)])
        for i, (norm, prim) in enumerate(zip(norms, self.primitives)):
            n = len(prim.get_parameters(free=free))
            r[c : c + n, :] = norm * prim.gradient(phases, free=free)
            c += n
            prim_terms[:, i] = prim(phases) - 1
        # handle case where no norm parameters are free
        if c == r.shape[0]:
            return r
        m = self.norms.gradient(free=free)
        # r[c:,:] = (prim_terms*m).sum(axis=1)
        for j in range(m.shape[0]):
            r[c, :] = (prim_terms * m[:, j]).sum(axis=1)
            c += 1
        return r

    def gradient_derivative(self, phases, log10_ens=3, free=False):
        """ Return d/dphi(gradient).  This is the derivative with respect
            to pulse phase of the gradient with respect to the parameters.
        """
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
        return approx_gradient(self, phases, log10_ens, eps=eps)

    def approx_hessian(self, phases, log10_ens=None, eps=1e-5):
        return approx_hessian(self, phases, log10_ens, eps=eps)

    def check_gradient(self, atol=1e-8, rtol=1e-5, quiet=False):
        return check_gradient(self, atol=atol, rtol=rtol, quiet=quiet)

    def hessian(self, phases, log10_ens=3, free=True):
        """ Return the hessian of the primitive and normaliation angles.

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
        nnorm_param = len(self.norms.p)
        nprim_param = nparam - nnorm_param
        r = np.zeros([nparam, nparam, len(phases)])

        norms = self.norms()
        norm_grads = self.norms.gradient(phases, free=False)
        prim_terms = np.empty([len(self.primitives), len(phases)])

        c = 0
        for i, prim in enumerate(self.primitives):
            h = prim.hessian(phases, free=False)
            pg = prim.gradient(phases, free=False)
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
            prim_terms[i, :] = prim(phases) - 1
            c += n

        # now put in normalization hessian
        hnorm = self.norms.hessian()  # nnorm_param x nnorm_param x nnorm_param
        for j in range(nnorm_param):
            for k in range(j, nnorm_param):
                for i in range(nnorm_param):
                    r[c + j, c + k, :] += hnorm[i, j, k] * prim_terms[i]
                r[c + k, c + j, :] = r[c + j, c + k, :]

        if free:
            return r[free_mask][:, free_mask]
        return r

    def delta(self, index=None):
        """ Return radio lag -- reckoned by default as the posittion of the            first peak following phase 0."""
        if (index is not None) and (index <= (len(self.primitives))):
            return self[index].get_location(error=True)
        return self.Delta(delta=True)

    def Delta(self, delta=False):
        """ Report peak separation -- reckoned by default as the distance
            between the first and final component locations.
            
            delta [False] -- if True, return the first peak position"""
        if len(self.primitives) == 1:
            return -1, 0
        prim0, prim1 = self[0], self[-1]
        for p in self.primitives:
            if p.get_location() < prim0.get_location():
                prim0 = p
            if p.get_location() > prim1.get_location():
                prim1 = p
        p1, e1 = prim0.get_location(error=True)
        p2, e2 = prim1.get_location(error=True)
        if delta:
            return p1, e1
        return (p2 - p1, (e1 ** 2 + e2 ** 2) ** 0.5)

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
            s1 += "\ndelta   : %.4f +\- %.4f" % self.delta()
            s1 += "\nDelta   : %.4f +\- %.4f" % self.Delta()
        return s0 + s1

    def prof_string(self, outputfile=None):
        """ Return a string compatible with the format used by pygaussfit.
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
        const = "const = %.5f +/- %.5f" % (1 - norm, errnorm ** 0.5)
        rstring = [dashes] + [const] + rstrings + [dashes]
        if outputfile is not None:
            f = open(outputfile, "w")
            f.write("# gauss\n")
            for s in rstring:
                f.write(s + "\n")
        return "\n".join(rstring)

    def random(self, n, weights=None, return_partition=False, randomize_partition=True):
        """ Return n pseudo-random variables drawn from the distribution 
            given by this light curve template.

            Uses a mulitinomial to divvy the n phases up amongs the various
            components, which then each generate MC phases from their own
            distributions.

            If a vector of weights is provided, the weight in interpreted
            as the probability that a photon comes from the template.  A
            random check is done according to this probability to
            determine whether to draw the photon from the template or from
            a uniform distribution.

            By default, the partition of light curve components is
            randomized, i.e. there is no time or weight dependence.  If not
            randomized, the partition have components sorted by their
            their overall amplitude (decreasing).
        """

        # compatibility with energy-dependent calls
        if hasattr(n, "__len__"):
            n = len(n)

        # edge case of uniform template
        if len(self.primitives) == 0:
            if return_partition:
                return np.random.rand(n), [n]
            return np.random.rand(n)

        n = int(round(n))
        norms = self.norms()
        norms = np.append(norms, [1 - sum(norms)])
        rvals = np.empty(n)
        partition = np.empty(n)  # record the index of each component
        if weights is not None:
            # perform a trial to see if each photon is from bg or template
            if len(weights) != n:
                raise ValueError(
                    "Provided weight vector does not provide a weight for each photon."
                )
            t = np.random.rand(n)
            m = t <= weights
            t_indices = np.arange(n)[m]  # indices to draw from templates (not bg)
            n = m.sum()
            rvals[~m] = np.random.rand(
                len(m) - n
            )  # assign uniform phases to the bg photons
            partition[~m] = len(norms) - 1  # set partition to bg component (last)
        else:
            t_indices = np.arange(n)

        # TODO -- faster implementation for single component case

        if randomize_partition:
            t_indices = np.random.permutation(t_indices)

        # multinomial implementation -- draw from the template components
        a = np.argsort(norms)[::-1]
        boundaries = np.cumsum(norms[a])
        components = np.searchsorted(boundaries, np.random.rand(n))
        counter = 0
        for mapped_comp, comp in zip(a, np.arange(len(norms))):
            n = (components == comp).sum()
            idx = t_indices[counter : counter + n]
            if mapped_comp == len(norms) - 1:  # uniform
                rvals[idx] = np.random.rand(n)
            else:
                rvals[idx] = self.primitives[mapped_comp].random(n)
            partition[idx] = mapped_comp
            counter += n
        if return_partition:
            return rvals, partition
        return rvals

    def swap_primitive(self, index, ptype=LCLorentzian):
        """ Swap the specified primitive for a new one with the parameters
           that match the old one as closely as possible."""
        self.primitives[index] = convert_primitive(self.primitives[index], ptype)

    def delete_primitive(self, index):
        """ [Convenience] -- return a new LCTemplate with the specified
            LCPrimitive removed and renormalized."""
        norms, prims = self.norms, self.primitives
        if len(prims) == 1:
            raise ValueError("Template only has a single primitive.")
        if index < 0:
            index += len(prims)
        nprims = [deepcopy(prims[i]) for i in range(len(prims)) if i != index]
        nnorms = np.asarray([norms()[i] for i in range(len(prims)) if i != index])
        norms_free = [norms.free[i] for i in range(len(prims)) if i != index]
        lct = LCTemplate(nprims, nnorms * norms().sum() / nnorms.sum())
        lct.norms.free[:] = norms_free
        return lct

    def add_primitive(self, prim, norm=0.1):
        """ [Convenience] -- return a new LCTemplate with the specified
            LCPrimitive added and renormalized."""
        norms, prims = self.norms, self.primitives
        nprims = [prim] + [deepcopy(prims[i]) for i in range(len(prims))]
        nnorms = np.asarray([norm] + [norms()[i] for i in range(len(prims))])
        norms_free = [True] + [norms.free[i] for i in range(len(prims))]
        lct = LCTemplate(nprims, nnorms * norms().sum() / nnorms.sum())
        lct.norms.free[:] = norms_free
        return lct

    def order_primitives(self, indices, zeropt=0, order_by_amplitude=False):
        """ Order the primitives specified by the indices by position.
            x0 specifies an alternative zeropt.
            
            If specified, the peaks will instead be ordered by descending 
            maximum amplitude."""

        def dist(index):
            p = self.primitives[index]
            if order_by_amplitude:
                a = self.get_amplitudes()
                return a.max() - a[index]
            d = p.get_location() - zeropt
            return d if (d > 0) else d + 1

        if not hasattr(indices, "__len__"):
            raise TypeError("indices must specify a list or array of indices")
        if len(indices) < 2:
            print("Found fewer than 2 indices, returning.")
            return
        norms, prims = self.norms(), self.primitives
        norms_free = self.norms.free.copy()
        for i in indices:
            x0 = dist(i)
            x1 = x0
            swapj = i
            for j in indices[i + 1 :]:
                dj = dist(j)
                if dj < x1:
                    x1 = dj
                    swapj = j
            if x1 < x0:
                j = swapj
                prims[i], prims[j] = prims[j], prims[i]
                norms[i], norms[j] = norms[j], norms[i]
                norms_free[i], norms_free[j] = norms_free[j], norms_free[i]
        self.norms = NormAngles(norms)  # this may be fragile
        self.norms.free[:] = norms_free

    def get_fixed_energy_version(self, log10_en=3):
        return self

    def get_eval_string(self):
        """ Return a string that can be "eval"ed to make a cloned set of
            primitives and template. """
        ps = "\n".join(
            ("p%d = %s" % (i, p.eval_string()) for i, p in enumerate(self.primitives))
        )
        prims = "[%s]" % (",".join(("p%d" % i for i in range(len(self.primitives)))))
        ns = "norms = %s" % (self.norms.eval_string())
        s = "%s(%s,norms)" % (self.__class__.__name__, prims)
        return s

    def closest_to_peak(self, phases):
        return min((p.closest_to_peak(phases) for p in self.primitives))

    def mean_value(self, phases, log10_ens=None, weights=None, bins=20):
        """ Compute the mean value of the profile over the codomain of 
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

    def align_peak(self, phi=0, dphi=0.001):
        """ Adjust such that template peak arrives within dphi of phi."""
        self.mark_cache_dirty()
        nbin = int(1.0 / dphi) + 1
        # This shifts the first primitive to peak at phase 0.0
        # Could instead use tallest primitive or some other feature
        shift = -1.0 * self.primitives[0].get_location()
        log.info("Shifting profile peak by {0}".format(shift))
        for prim in self.primitives:
            new_location = (prim.get_location() + shift) % 1
            prim.set_location(new_location)

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
            lo = values[0:-1:2]
            mid = values[1::2]
            bin_phases = phases[0:-1:2]
            bin_values = 1.0 / (6 * nbin) * (hi + 4 * mid + lo)

        bin_values *= 1.0 / bin_values.mean()
        open(fname, "w").write(
            "".join(("%.6f %.6f\n" % (x, y) for x, y in zip(bin_phases, bin_values)))
        )


class LCBridgeTemplate(LCTemplate):
    """ A light curve template specialized to the "typical" shape of a
        gamma-ray pulsar, viz. two peaks linked by a bridge.  The bridge
        is implemented as a pedestal connecting the modes of two specific
        peaks, and the only free parameter associated with it is its
        normalization (from which its amplituded is determimned).
    """

    def has_bridge(self):
        return True

    def __init__(self, primitives, norms=None):
        """ primitives -- a list of LCPrimitive instances of len >= 2; the
                last two components are interpreted as P1 and P2
            norms -- either an instance of NormAngles, or a tuple of
                relative amplitudes for the primitive components; should
                have one extra parameter for the bridge component; the
                first component is interpreted as the pedestal nor
        """
        if norms is None:
            norms = np.ones(len(primitives) + 1) / (len(primitives + 1))
        super(LCBridgeTemplate, self).__init__(primitives, norms)
        self.p1 = self.primitives[-2]
        self.p2 = self.primitives[-1]

    def _sanity_checks(self):
        if len(self.primitives) != len(self.norms) - 1:
            raise ValueError("Require n_primitive+1 norm components.")

    def _get_scales(self, phases, log10_ens=3):
        """ Return the scale factor for p1, p2, and the pedestal such that
            the pedestal has the correct normalization and p1, p2, and the
            pedestal form a smooth, continuous curve.
        """
        all_norms = self.norms(log10_ens)
        nped, norms = all_norms[0], all_norms[1:]
        n1, n2 = norms[-2:]
        p1, p2 = self.p1, self.p2
        # NB -- location need to be made "energy aware"
        l1, l2 = p1.get_location(), p2.get_location()
        # can't really enforce l1 < l2, so need to allow for wrapping
        delta = (l2 - l1) + (l2 < l1)
        # evaluate each peak at change points
        f11, f12, f21, f22 = (
            p1(l1, log10_ens),
            p1(l2, log10_ens),
            p2(l1, log10_ens),
            p2(l2, log10_ens),
        )
        d = f11 * f22 - f12 * f21
        if l2 > l1:
            i1 = p1.integrate(l1, l2, log10_ens)
            i2 = p2.integrate(l1, l2, log10_ens)
        else:
            i1 = 1 - p1.integrate(l2, l1, log10_ens)
            i2 = 1 - p2.integrate(l2, l1, log10_ens)
        # coefficient for pedestal
        k = nped * (1.0 - (i1 * (f22 - f21) + i2 * (f11 - f12)) / (d * delta)) ** -1
        # rescaling for peaks over pedestal
        dn1 = k / (delta * d) * (f21 - f22)
        dn2 = k / (delta * d) * (f12 - f11)
        # make the mask inclusive for ease of testing
        if l2 > l1:
            mask = (phases >= l1) & (phases <= l2)
        else:
            mask = (phases >= l1) | (phases <= l2)
        rvals = k / delta * mask  # pedestal
        norm_list = [norms[i] for i in range(0, len(norms) - 2)] + [
            n1 + dn1 * mask,
            n2 + dn2 * mask,
        ]
        return rvals, norm_list, all_norms.sum(axis=0)

    def random(self, n, weights=None, return_partition=False):
        # note -- this wouldn't be that hard to do, just do multinomial as
        # usual, then an additional step to determine which part of the peaks
        # the photons should come from
        raise NotImplementedError()

    def __str__(self):
        s = super(LCBridgeTemplate, self).__str__()
        return "TODO: Add pedestal stuff! \n\n" + s
        # prims = self.primitives
        # s0 = str(self.norms)
        # s1 = '\n\n'+'\n\n'.join( ['P%d -- '%(i+1)+str(prim) for i,prim in enumerate(prims)] ) + '\n'
        # s1 +=  '\ndelta   : %.4f +\- %.4f'%self.delta()
        # s1 +=  '\nDelta   : %.4f +\- %.4f'%self.Delta()
        # return s0+s1

    def single_component(self, index, phases, log10_ens=3):
        """ Evaluate a single component of template."""
        # this needs to be done in some sane way, not sure if ideal exists
        # so best guess is to compute the pedestal offset and add that on
        # to the inner peak, trusting that it will itself be plotted to
        # give the sense that this component isn't independnet
        # raise NotImplementedError()
        rvals, norms, norm = self._get_scales(phases, log10_ens)
        if index < len(self.primitives):
            np.add(
                rvals, norms[index] * self.primitives[index](phases, log10_ens), rvals
            )
        return rvals

    def mean_single_component(
        self, index, phases, log10_ens=None, weights=None, bins=20, add_pedestal=False
    ):
        # if add_pedestal:
        # print('No add pedestal.')
        # this needs to be done in some sane way, not sure if ideal exists
        # raise NotImplementedError()
        if (log10_ens is None) or (not self.is_energy_dependent()):
            return self.single_component(index, phases)
        if weights is None:
            weights = np.ones_like(log10_ens)
        edges = np.linspace(log10_ens.min(), log10_ens.max(), bins + 1)
        w = np.histogram(log10_ens, weights=weights, bins=edges)
        rvals = np.zeros_like(phases)
        for weight, en in zip(w[0], (edges[:-1] + edges[1:]) / 2):
            rvals += weight * self.single_component(index, phases, en)
        rvals /= w[0].sum()
        return rvals


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
    """ Convenience function to get a 2 Lorentzian + Gaussian bridge template."""
    prim = LCLorentzian if lorentzian else LCGaussian
    p1 = prim(p=[0.03, 0.1])
    b = LCGaussian(p=[0.15, 0.3])
    p2 = prim(p=[0.03, 0.55])
    return LCTemplate(
        primitives=[p1, b, p2],
        norms=[0.3 * pulse_frac, 0.4 * pulse_frac, 0.3 * pulse_frac],
    )


def make_twoside_gaussian(one_side_gaussian):
    """ Make a two-sided gaussian with the same initial shape as the
        input one-sided gaussian."""
    g2 = LCGaussian2()
    g1 = one_side_gaussian
    g2.p[0] = g2.p[1] = g1.p[0]
    g2.p[-1] = g1.p[-1]
    return g2


class GaussianPrior(object):
    def __init__(self, locations, widths, mod, mask=None):
        self.x0 = np.where(mod, np.mod(locations, 1), locations)
        self.s0 = np.asarray(widths) * 2 ** 0.5
        self.mod = np.asarray(mod)
        if mask is None:
            self.mask = np.asarray([True] * len(locations))
        else:
            self.mask = np.asarray(mask)
            self.x0 = self.x0[self.mask]
            self.s0 = self.s0[self.mask]
            self.mod = self.mod[self.mask]

    def __len__(self):
        """ Return number of parameters with a prior."""
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
        rvals[self.mask] = 2 * (parameters - self.x0) / self.s0 ** 2
        return rvals


def prim_io(template):
    """ Read files and build LCPrimitives. """

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
        return primitives, norms

    toks = [line.strip().split() for line in open(template) if len(line.strip()) > 0]
    if "gauss" in toks[0]:
        return read_gaussian(toks[1:])
    elif "kernel" in toks[0]:
        return [LCKernelDensity(input_file=toks[1:])], None
    elif "fourier" in toks[0]:
        return [LCEmpiricalFourier(input_file=toks[1:])], None
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
