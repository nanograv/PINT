"""
A module implementing binned and unbinned likelihood for weighted and
unweighted sets of photon phases.  The model is encapsulated in LCTemplate,
a mixture model.

LCPrimitives are combined to form a light curve (LCTemplate).
LCFitter then performs a maximum likelihood fit to determine the
light curve parameters.

LCFitter also allows fits to subsets of the phases for TOA calculation.

$Header: /nfs/slac/g/glast/ground/cvs/pointlike/python/uw/pulsar/lcfitters.py,v 1.54 2017/03/24 18:48:51 kerrm Exp $

author: M. Kerr <matthew.kerr@gmail.com>

"""
import numpy as np
import scipy
from pint.eventstats import hm, hmw
from scipy.optimize import fmin, fmin_tnc, leastsq

SECSPERDAY = 86400.0


def isvector(x):
    return len(np.asarray(x).shape) > 0


def shifted(m, delta=0.5):
    """Produce a copy of a binned profile shifted in phase by delta."""
    f = np.fft.fft(m, axis=-1)
    n = f.shape[-1]
    arg = np.fft.fftfreq(n) * (n * np.pi * 2.0j * delta)
    return np.real(np.fft.ifft(np.exp(arg) * f, axis=-1))


def weighted_light_curve(nbins, phases, weights, normed=False, phase_shift=0):
    """Return a set of bins, values, and errors to represent a
    weighted light curve."""
    bins = np.linspace(0 + phase_shift, 1 + phase_shift, nbins + 1)
    counts = np.histogram(phases, bins=bins, density=False)[0]
    w1 = (np.histogram(phases, bins=bins, weights=weights, density=False)[0]).astype(
        float
    )
    w2 = (
        np.histogram(phases, bins=bins, weights=weights**2, density=False)[0]
    ).astype(float)
    errors = np.where(counts > 1, w2**0.5, counts)
    norm = w1.sum() / nbins if normed else 1.0
    return bins, w1 / norm, errors / norm


class LCFitter:
    def __init__(
        self,
        template,
        phases,
        weights=None,
        log10_ens=3,
        times=1,
        binned_bins=1000,
        binned_ebins=8,
        phase_shift=0,
    ):
        """Class for fitting light curves.

        Arguments:
        template -- an instance of LCTemplate
        phases   -- list of photon phases

        Keyword arguments:
        weights      [None] optional photon weights
        log10_ens    [None] optional photon energies (log10(E/MeV))
        times        [None] optional photon arrival times
        binned_bins  [100]  phase bins to use in binned likelihood
        binned_ebins [8]    energy bins to use in binned likelihood
        phase_shift  [0]    set this if a phase shift has been applied
        """
        self.template = template
        self.phases = np.asarray(phases)
        if weights is None:
            weights = np.ones(len(phases), dtype=float)
        self.weights = weights
        self.log10_ens = np.asarray(log10_ens)
        self.times = times
        self.binned_bins = binned_bins
        self.binned_ebins = binned_ebins  # TODO?
        self.phase_shift = phase_shift
        self.loglikelihood = self.unbinned_loglikelihood
        self.gradient = self.unbinned_gradient

        self._binned_setup()

    def __str__(self):
        if "ll" in self.__dict__.keys():
            return "\nLog Likelihood for fit: %.2f\n" % (self.ll) + str(self.template)
        return str(self.template)

    def __getstate__(self):
        """Cannot pickle self.loglikelihood and self.gradient since
        these are instancemethod objects.
        See: http://mail.python.org/pipermail/python-list/2000-October/054610.html"""
        result = self.__dict__.copy()
        # del result["loglikelihood"]
        # del result["gradient"]
        result.pop("loglikelihood")
        result.pop("gradient")
        return result

    def __setstate__(self, state):
        self.__dict__ = state
        self.loglikelihood = self.unbinned_loglikelihood
        self.gradient = self.unbinned_gradient

    def is_energy_dependent(self):
        return self.template.is_energy_dependent()

    def _hist_setup(self):
        """Setup binning for a quick chi-squared fit."""
        h = hmw(self.phases, self.weights)
        nbins = 25
        if h > 100:
            nbins = 50
        if h > 1000:
            nbins = 100
        bins, counts, errors = weighted_light_curve(
            nbins, self.phases, self.weights, phase_shift=self.phase_shift
        )
        mask = counts > 0
        N = counts.sum()
        self.bg_level = 1 - (self.weights**2).sum() / N
        x = (bins[1:] + bins[:-1]) / 2
        y = counts / N * nbins
        yerr = errors / N * nbins
        self.chistuff = x[mask], y[mask], yerr[mask]
        # now set up binning for binned likelihood

    def _binned_setup(self):
        nbins = self.binned_bins
        bins = np.linspace(0 + self.phase_shift, 1 + self.phase_shift, nbins + 1)
        # TODO -- revisit this slice approach and implement in a way that
        # doesn't require sorting.  It seems very fragile.  Looking at the
        # binned likelihood, we could keep the masks, select the weights,
        # and broadcast the single template term.
        a = np.argsort(self.phases)
        self.phases = self.phases[a]
        self.weights = self.weights[a]
        if isvector(self.log10_ens):
            self.log10_ens = self.log10_ens[a]
        self.counts_centers = []
        self.slices = []
        indices = np.arange(len(self.weights))
        for i in range(nbins):
            mask = (self.phases >= bins[i]) & (self.phases < bins[i + 1])
            if mask.sum() > 0:
                w = self.weights[mask]
                if w.sum() == 0:
                    continue
                p = self.phases[mask]
                self.counts_centers.append((w * p).sum() / w.sum())
                # self.counts_centers.append(0.5*(bins[i]+bins[i+1]))
                self.slices.append(slice(indices[mask].min(), indices[mask].max() + 1))
        self.counts_centers = np.asarray(self.counts_centers)

    def unbinned_loglikelihood(self, p, *args, **kwargs):
        t = self.template
        params_ok = t.set_parameters(p)
        if not t.norm_ok():
            return 2e20
        if (not params_ok) and ("skip_bounds_check" not in kwargs):
            return 2e20
        # TODO -- keep this formulation??
        arg = 1 + self.weights * (
            t(self.phases, log10_ens=self.log10_ens, use_cache=False) - 1
        )
        arg[arg <= 0] = 1e-300
        return -np.log(arg).sum()
        # return -np.log(1 + self.weights * (t(self.phases,log10_ens=self.log10_ens,use_cache=False) - 1)).sum()

    def binned_loglikelihood(self, p, *args, **kwargs):
        t = self.template
        params_ok = t.set_parameters(p)
        if not t.norm_ok():
            return 2e20
        if (not params_ok) and ("skip_bounds_check" not in kwargs):
            return 2e20
        template_terms = (
            t(self.counts_centers, log10_ens=self.log10_ens, use_cache=False) - 1
        )
        phase_template_terms = np.empty_like(self.weights)
        for tt, sl in zip(template_terms, self.slices):
            phase_template_terms[sl] = tt
        # TODO -- keep this formulation??
        arg = 1 + self.weights * phase_template_terms
        arg[arg <= 0] = 1e-300
        return -np.log(arg).sum()
        # return -np.log(1 + self.weights * phase_template_terms).sum()

    def unbinned_gradient(self, p, *args, **kwargs):
        t = self.template
        params_ok = t.set_parameters(p)
        if not t.norm_ok():
            return np.full(p.shape, 2e20)
        if (not params_ok) and ("skip_bounds_check" not in kwargs):
            return np.full(p.shape, 2e20)
        g, tmpl = t.gradient(self.phases, log10_ens=self.log10_ens, template_too=True)
        # numer = self.weights * t.gradient(self.phases,log10_ens=self.log10_ens)
        # denom = 1 + self.weights * (t(self.phases,log10_ens=self.log10_ens,use_cache=False) - 1)
        numer = self.weights * g
        denom = 1 + self.weights * (tmpl - 1)
        return -np.sum(numer / denom, axis=1)

    def binned_gradient(self, p, *args, **kwargs):
        t = self.template
        params_ok = t.set_parameters(p)
        if not t.norm_ok():
            return np.full(p.shape, 2e20)
        if (not params_ok) and ("skip_bounds_check" not in kwargs):
            return np.full(p.shape, 2e20)
        nump = len(p)
        template_terms = (
            t(self.counts_centers, log10_ens=self.log10_ens, use_cache=False) - 1
        )
        gradient_terms = t.gradient(self.counts_centers, log10_ens=self.log10_ens)
        phase_template_terms = np.empty_like(self.weights)
        phase_gradient_terms = np.empty([nump, len(self.weights)])
        # distribute the central values to the unbinned phases/weights
        for tt, gt, sl in zip(template_terms, gradient_terms.transpose(), self.slices):
            phase_template_terms[sl] = tt
            for j in range(nump):
                phase_gradient_terms[j, sl] = gt[j]
        numer = self.weights * phase_gradient_terms
        denom = 1 + self.weights * (phase_template_terms)
        return -(numer / denom).sum(axis=1)

    def chi(self, p, *args):
        x, y, yerr = self.chistuff
        bg = self.bg_level
        if not self.template.shift_mode and np.any(p < 0):
            return 2e100 * np.ones_like(x) / len(x)
        args[0].set_parameters(p)
        return (bg + (1 - bg) * self.template(x) - y) / yerr

    def quick_fit(self):
        t = self.template
        p0 = t.get_parameters().copy()
        chi0 = (self.chi(t.get_parameters(), t) ** 2).sum()
        f = leastsq(self.chi, t.get_parameters(), args=(t))
        chi1 = (self.chi(t.get_parameters(), t) ** 2).sum()
        print(chi0, chi1, " chi numbers")
        if chi1 > chi0:
            print(self)
            print("Failed least squares fit -- reset and proceed to likelihood.")
            t.set_parameters(p0)

    def _fix_state(self, restore_state=None):
        old_state = []
        counter = 0
        for p in self.template.primitives:
            for i in range(len(p.p)):
                old_state.append(p.free[i])
                if restore_state is not None:
                    p.free[i] = restore_state[counter]
                elif i < (len(p.p) - 1):
                    p.free[i] = False
                counter += 1
        return old_state

    def _set_unbinned(self, unbinned=True):
        if unbinned:
            self.loglikelihood = self.unbinned_loglikelihood
            self.gradient = self.unbinned_gradient
        else:
            self.loglikelihood = self.binned_loglikelihood
            self.gradient = self.binned_gradient

    def fit(
        self,
        quick_fit_first=False,
        unbinned=True,
        use_gradient=True,
        ftol=1e-5,
        overall_position_first=False,
        positions_first=False,
        estimate_errors=False,
        prior=None,
        unbinned_refit=True,
        try_bootstrap=True,
        quiet=False,
    ):
        # NB use of priors currently not supported by quick_fit, positions first, etc.
        self._set_unbinned(unbinned)
        if (prior is not None) and (len(prior) > 0):
            fit_func = lambda x: self.loglikelihood(x) + prior(x)
            grad_func = lambda x: self.gradient(x) + prior.gradient(x)
        else:
            fit_func = self.loglikelihood
            grad_func = self.gradient

        if overall_position_first:
            """do a brute force scan over profile down to <1mP."""

            def logl(phase):
                self.template.set_overall_phase(phase)
                return self.loglikelihood(self.template.get_parameters())

            # coarse grained
            dom = np.linspace(0, 1, 101)
            cod = [logl(x) for x in 0.5 * (dom[1:] + dom[:-1])]
            idx = np.argmin(cod)
            # fine grained
            dom = np.linspace(dom[idx], dom[idx + 1], 101)
            cod = [logl(x) for x in dom]
            # set to best fit phase shift
            ph0 = dom[np.argmin(cod)]
            self.template.set_overall_phase(ph0)

        if positions_first:
            print("Running positions first")
            restore_state = self._fix_state()
            self.fit(
                quick_fit_first=quick_fit_first,
                estimate_errors=False,
                unbinned=unbinned,
                use_gradient=use_gradient,
                positions_first=False,
                quiet=quiet,
            )
            self._fix_state(restore_state)

        # an initial chi squared fit to find better seed values
        if quick_fit_first:
            self.quick_fit()

        ll0 = -fit_func(self.template.get_parameters())
        p0 = self.template.get_parameters().copy()
        if use_gradient:
            f = self.fit_tnc(fit_func, grad_func, ftol=ftol, quiet=quiet)
        else:
            f = self.fit_fmin(fit_func, ftol=ftol)
        if (ll0 > self.ll) or (ll0 == 2e20) or (np.isnan(ll0)):
            if (
                unbinned_refit
                and np.isnan(ll0)
                and (not unbinned)
                and (self.binned_bins * 2) < 400
            ):
                print(
                    "Did not converge using %d bins... retrying with %d bins..."
                    % (self.binned_bins, self.binned_bins * 2)
                )
                self.template.set_parameters(p0)
                self.ll = ll0
                self.fitvals = p0
                self.binned_bins *= 2
                self._hist_setup()
                return self.fit(
                    quick_fit_first=quick_fit_first,
                    unbinned=unbinned,
                    use_gradient=use_gradient,
                    positions_first=positions_first,
                    estimate_errors=estimate_errors,
                    prior=prior,
                )
            self.bad_p = self.template.get_parameters().copy()
            self.bad_ll = self.ll
            print("Failed likelihood fit -- resetting parameters.")
            if np.isnan(ll0):
                print("   (Condition: LL = NaN)")
            if ll0 > self.ll:
                print("   (Condition: LL did not improve)")
            if ll0 == -2e20:
                print("   (Condition: LL set to -infty)")
            self.template.set_parameters(p0)
            self.ll = ll0
            self.fitvals = p0
            return False
        if (
            estimate_errors
            and not self.hess_errors(use_gradient=use_gradient)
            and try_bootstrap
        ):
            self.bootstrap_errors(set_errors=True)
        if not quiet:
            print("Improved log likelihood by %.2f" % (self.ll - ll0))
        return True

    def fit_position(self, unbinned=True, track=False, skip_coarse=False):
        """Fit overall template position.  Return shift and its error.

        Parameters
        ----------
        unbinned : bool
            Use unbinned likelihood; will be very slow for many photons.
        track : bool
            Limit best-fit solution to +/- 0.2 periods of zero phase.
            Helps to avoid 0.5period ambiguity for two-peaked profiles.

        Returns
        -------
        delta_phi : float
            overall phase shift from template
        delta_phi_err : float
            estimated uncertainty on phase shift from likelihood hessian

        """

        if not self.template.check_bounds():
            raise ValueError("Template does not satisfy parameter bounds.")

        self._set_unbinned(unbinned)
        ph0 = self.template.get_location()

        def logl(phase):
            self.template.set_overall_phase(phase)
            return self.loglikelihood(self.template.get_parameters())

        # coarse grained search
        if track:
            dom = np.append(np.linspace(0.0, 0.2, 25), np.linspace(0.8, 1.0, 25)[:-1])
        else:
            if skip_coarse:
                dom = np.linspace(ph0 - 0.01, ph0 + 0.01, 21)
            else:
                dom = np.linspace(0, 1, 101)
        x0 = min(dom, key=logl)
        ph1 = fmin(logl, x0, full_output=True, disp=0, xtol=1e-6)[0][0]
        self.template.set_overall_phase(ph1)

        # estimate error by computing d2logl/dphi2
        f0 = self.template(self.phases)
        f1 = self.template.derivative(self.phases, order=1)
        f2 = self.template.derivative(self.phases, order=2)
        w = self.weights
        den = 1 + w * (f0 - 1)
        d2 = np.sum(((w * f1) / den) ** 2 - w * f2 / den)
        return ph1 - ph0, d2**-0.5

    def fit_background(self, unbinned=True):
        """Fit the background level, holding the ratios of the pulsed
        components fixed but varying their total normalization."""

        self._set_unbinned(unbinned)

        def logl(p):
            if np.isscalar(p):
                self.template.norms.set_total(p)
            else:
                self.template.norms.set_total(p[0])
            return self.loglikelihood(self.template.get_parameters())

        old_total = self.template.norms.get_total()

        grid = np.linspace(0, 1, 51)[1:]
        gridbest = min(grid, key=logl)
        gmax = min(grid[-1], gridbest + 0.02)
        gmin = max(grid[0], gridbest - 0.02)
        grid = np.linspace(gmin, gmax, 51)
        gridbest = min(grid, key=logl)
        self.template.norms.set_total(gridbest)
        return gridbest

    def fit_fmin(self, fit_func, ftol=1e-5):
        x0 = self.template.get_parameters()
        fit = fmin(fit_func, x0, disp=0, ftol=ftol, full_output=True)
        self.fitval = fit[0]
        self.ll = -fit[1]
        return fit

    def fit_cg(self):
        from scipy.optimize import fmin_cg

        return fmin_cg(
            self.loglikelihood,
            self.template.get_parameters(),
            fprime=self.gradient,
            args=(self.template,),
            full_output=1,
            disp=1,
        )

    def fit_bfgs(self):
        from scipy.optimize import fmin_bfgs

        # bounds = self.template.get_bounds()
        fit = fmin_bfgs(
            self.loglikelihood,
            self.template.get_parameters(),
            fprime=self.gradient,
            args=(self.template,),
            full_output=1,
            disp=1,
            gtol=1e-5,
            norm=2,
        )
        self.template.set_errors(np.diag(fit[3]) ** 0.5)
        self.fitval = fit[0]
        self.ll = -fit[1]
        self.cov_matrix = fit[3]
        return fit

    def fit_tnc(self, fit_func, grad_func, ftol=1e-5, quiet=False):
        x0 = self.template.get_parameters()
        bounds = self.template.get_bounds()
        fit = fmin_tnc(
            fit_func,
            x0,
            fprime=grad_func,
            ftol=ftol,
            pgtol=1e-5,
            bounds=bounds,
            maxfun=5000,
            messages=8,
            disp=0 if quiet else 1,
        )
        self.fitval = fit[0]
        self.ll = -fit_func(self.template.get_parameters())
        return fit

    def fit_l_bfgs_b(self):
        from scipy.optimize import fmin_l_bfgs_b

        x0 = self.template.get_parameters()
        bounds = self.template.get_bounds()
        return fmin_l_bfgs_b(
            self.loglikelihood, x0, fprime=self.gradient, bounds=bounds, factr=1e-5
        )

    def hess_errors(self, use_gradient=True):
        """Set errors from hessian.  Fit should be called first..."""
        p = self.template.get_parameters()
        nump = len(p)
        self.cov_matrix = np.zeros([nump, nump], dtype=float)
        logl = lambda p: self.loglikelihood(p, skip_bounds_check=True)
        grad = lambda p: self.gradient(p, skip_counts_check=True)
        ss = calc_step_size(logl, p.copy())
        if use_gradient:
            h1 = hess_from_grad(grad, p.copy(), step=ss)
            c1 = scipy.linalg.inv(h1)
            if np.all(np.diag(c1) > 0):
                self.cov_matrix = c1
            else:
                print("Could not estimate errors from hessian.")
                return False
        else:
            h1 = hessian(self.template, logl, delta=ss)
            try:
                c1 = scipy.linalg.inv(h1)
            except scipy.linalg.LinAlgError:
                print("Hessian matrix was singular!  Aborting.")
                return False
            d = np.diag(c1)
            if np.all(d > 0):
                self.cov_matrix = c1
                # attempt to refine
                h2 = hessian(self.template, logl, delt=d**0.5)
                try:
                    c2 = scipy.linalg.inv(h2)
                except scipy.linalg.LinAlgError:
                    print("Second try at hessian matrix was singular!  Aborting.")
                    return False
                if np.all(np.diag(c2) > 0):
                    self.cov_matrix = c2
            else:
                print("Could not estimate errors from hessian.")
                return False
        self.template.set_errors(np.diag(self.cov_matrix) ** 0.5)
        return True

    def bootstrap_errors(self, nsamp=100, fit_kwargs={}, set_errors=False):
        p0 = self.phases
        w0 = self.weights
        param0 = self.template.get_parameters().copy()
        n = len(p0)
        results = np.empty([nsamp, len(self.template.get_parameters())])
        fit_kwargs["estimate_errors"] = False  # never estimate errors
        if "unbinned" not in fit_kwargs.keys():
            fit_kwargs["unbinned"] = True
        counter = 0
        for i in range(nsamp * 2):
            if counter == nsamp:
                break
            if i == (2 * nsamp - 1):
                self.phases = p0
                self.weights = w0
                raise ValueError("Could not construct bootstrap sample.  Giving up.")
            a = (np.random.rand(n) * n).astype(int)
            self.phases = p0[a]
            if w0 is not None:
                self.weights = w0[a]
            if not fit_kwargs["unbinned"]:
                self._hist_setup()
            if self.fit(**fit_kwargs):
                results[counter, :] = self.template.get_parameters()
                counter += 1
            self.template.set_parameters(param0)
        if set_errors:
            self.template.set_errors(np.std(results, axis=0))
        self.phases = p0
        self.weights = w0
        return results

    def write_template(self, outputfile="template.gauss"):
        s = self.template.prof_string(outputfile=outputfile)

    def plot(
        self,
        nbins=50,
        fignum=2,
        axes=None,
        plot_components=False,
        template=None,
        line_color="blue",
        comp_color=None,
        plot_eavg=True,
        log10_erange=None,
        optimize_display_phase=True,
    ):
        import pylab as pl

        if comp_color is None:
            comp_color = line_color

        weights = self.weights
        log10_ens = self.log10_ens
        phases = self.phases

        if template is None:
            template = self.template

        if optimize_display_phase:
            template = template.copy()
            dphi = template.get_display_point(do_rotate=True)
            phases = np.mod(phases + dphi, 1)

        plot_log_en = 3
        if (log10_erange is not None) and (log10_ens is not None):
            lo, hi = log10_erange
            mask = (log10_ens >= lo) & (log10_ens < hi)
            if weights is not None:
                weights = weights[mask]
            phases = phases[mask]
            log10_ens = log10_ens[mask]
            plot_log_en = 0.5 * (lo + hi)

        if axes is None:
            fig = pl.figure(fignum)
            axes = fig.add_subplot(111)

        axes.hist(
            phases,
            bins=np.linspace(0, 1, nbins + 1),
            histtype="step",
            ec="C3",
            density=True,
            lw=1,
            weights=weights,
        )

        if weights is not None:
            bg_level = 1 - (weights**2).sum() / weights.sum()
            axes.axhline(bg_level, color="k")
            x, w1, errors = weighted_light_curve(nbins, phases, weights, normed=True)
            x = (x[:-1] + x[1:]) / 2
            axes.errorbar(x, w1, yerr=errors, capsize=0, marker="", ls=" ", color="red")
        else:
            bg_level = 0
            h = np.histogram(phases, bins=np.linspace(0, 1, nbins + 1))
            x = (h[1][:-1] + h[1][1:]) / 2
            n = float(h[0].sum()) / nbins
            axes.errorbar(
                x,
                h[0] / n,
                yerr=h[0] ** 0.5 / n,
                capsize=0,
                marker="",
                ls=" ",
                color="C3",
            )

        def avg_energy(func):
            if not plot_eavg:
                return func(dom)
            if not isvector(log10_ens):
                return func(dom)
            h = np.histogram(log10_ens, weights=weights, bins=20)
            wt = h[0] * (1.0 / h[0].sum())
            hc = 0.5 * (h[1][:-1] + h[1][1:])
            rvals = np.zeros_like(dom)
            for x, w in zip(hc, wt):
                rvals += w * func(dom, log10_ens=x)
            return rvals

        dom = np.linspace(0, 1, 1000)

        cod = avg_energy(template) * (1 - bg_level) + bg_level
        axes.plot(dom, cod, color=line_color, lw=1)
        if plot_components:
            for i in range(len(template.primitives)):

                def f(ph, log10_ens=3):
                    return template.single_component(i, dom, log10_ens=log10_ens)

                cod = avg_energy(f) * (1 - bg_level) + bg_level
                axes.plot(dom, cod, color=comp_color, lw=1, ls="--")
        pl.axis([0, 1, pl.axis()[2], max(pl.axis()[3], cod.max() * 1.05)])
        axes.set_ylabel("Normalized Profile")
        axes.set_xlabel("Phase")
        axes.grid(True)
        return bg_level

    def plot_ebands(
        self, nband=4, fignum=2, plot_zoom=True, equalize_y=False, **plot_kwargs
    ):
        import pylab as pl

        pl.close(fignum)
        fig = pl.figure(fignum, (4.5 + 4.5 * int(plot_zoom), 7))
        if "fignum" in plot_kwargs:
            plot_kwargs.pop("fignum")

        log10_ebands = self.get_ebands(nband)
        if len(log10_ebands) == 0:
            raise ValueError("No energy information available.")
        toggle = int(plot_zoom)
        axes = []
        axzooms = []
        maxy = 0
        for i in range(nband):
            lo, hi = log10_ebands[i : i + 2]
            ax = pl.subplot(nband, 1 + toggle, i * (1 + toggle) + 1)
            axes.append(ax)
            plot_kwargs["log10_erange"] = [lo, hi]
            plot_kwargs["axes"] = ax
            self.plot(**plot_kwargs)
            maxy = max(ax.axis()[-1], maxy)
            ax.text(
                0.03,
                ax.axis()[-1] * 0.88,
                "%.2f--%.2f GeV" % (10**lo * 1e-3, 10**hi * 1e-3),
            )
            if plot_zoom:
                axzoom = pl.subplot(nband, 1 + toggle, i * (1 + toggle) + 2)
                plot_kwargs["axes"] = axzoom
                bg_level = self.plot(**plot_kwargs)
                axzoom.axis([0, 1, bg_level - 0.2, bg_level + 1.0])
                axzoom.set_ylabel("")
                axzoom.tick_params(
                    labelleft=False, labelright=True, labelbottom=i == (nband - 1)
                )
                if i >= nband - 1:
                    axzoom.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
                axzoom.set_xlabel("")
                axzoom.set_ylabel("")
                axzooms.append(axzoom)
            if i < (nband - 1):
                ax.tick_params(labelbottom=False)
            ax.set_xlabel("")
            ax.set_ylabel("")
        if equalize_y:
            for ax in axes:
                old_axis = list(ax.axis())
                old_axis[-1] = maxy
                ax.axis(tuple(old_axis))

        try:
            fig.supylabel("Normalized Profile")
            fig.supxlabel("Phase")
        except AttributeError:
            axes[-1].set_xlabel("Phase")
            if axzooms:
                axzooms[-1].set_xlabel("Phase")
            for ax in axes:
                ax.set_ylabel("Normalized Profile")
        pl.tight_layout()
        pl.subplots_adjust(hspace=0)
        pl.subplots_adjust(wspace=0)

    def get_ebands(self, nband=3):
        if not isvector(self.log10_ens):
            return []
        t = self.template(self.phases, log10_ens=self.log10_ens)
        if self.weights is None:
            logl = np.log(t)
        else:
            logl = np.log(1 + self.weights * (t - 1))
        a = np.argsort(self.log10_ens)
        logl = logl[a]
        logl = np.cumsum(logl)
        logl *= 1.0 / logl[-1]
        indices = np.searchsorted(logl, np.arange(nband)[1:] / nband)
        indices = np.append(0, indices)
        indices = np.append(indices, -1)
        log10_ens = self.log10_ens[a][indices]
        # round boundaries down/up to nearest 0.1
        log10_ens[0] = np.floor(log10_ens[0] * 10) * 0.1
        log10_ens[-1] = np.ceil(log10_ens[-1] * 10) * 0.1
        return log10_ens

    def plot_residuals(self, nbins=50, fignum=3):
        import pylab as pl

        edges = np.linspace(0, 1, nbins + 1)
        lct = self.template
        cod = np.asarray(
            [lct.integrate(e1, e2) for e1, e2 in zip(edges[:-1], edges[1:])]
        ) * len(self.phases)
        pl.figure(fignum)
        counts = np.histogram(self.phases, bins=edges)[0]
        pl.errorbar(
            x=(edges[1:] + edges[:-1]) / 2,
            y=counts - cod,
            yerr=counts**0.5,
            ls=" ",
            marker="o",
            color="red",
        )
        pl.axhline(0, color="blue")
        pl.ylabel("Residuals (Data - Model)")
        pl.xlabel("Phase")
        pl.grid(True)

    def rotate_for_display(self):
        """Rotate both internal phases and template to a nice phase.

        NB this will potentially break zero-phase references and will also
        necessitate re-binning if using binned mode.

        """
        dphi = self.template.get_display_point(do_rotate=True)
        self.phases = np.mod(self.phases + dphi, 1)
        self._binned_setup()

    def aic(self, template=None):
        """Return the Akaike information criterion for the current state.

        Note the sense of the statistic is such that more negative
        implies a better fit."""
        if template is not None:
            template, self.template = self.template, template
        else:
            template = self.template
        nump = len(template.get_parameters())
        ts = 2 * (nump + self())
        self.template = template
        return ts

    def bic(self, template=None):
        """Return the Bayesian information criterion for the current state.

        Note the sense of the statistic is such that more negative
        implies a better fit.

        This should work for energy-dependent templates provided the
        template and fitter match types."""
        if template is not None:
            template, self.template = self.template, template
        else:
            template = self.template
        nump = len(self.template.get_parameters())
        n = len(self.phases) if self.weights is None else self.weights.sum()
        ts = nump * np.log(n) + 2 * self()
        self.template = template
        return ts


def hessian(m, mf, *args, **kwargs):
    """Calculate the Hessian; mf is the minimizing function, m is the model,args additional arguments for mf."""
    p = m.get_parameters().copy()
    p0 = p.copy()  # sacrosanct copy
    delta = kwargs.get("delt", [0.01] * len(p))
    hessian = np.zeros([len(p), len(p)])
    for i in range(len(p)):
        delt = delta[i]
        for j in range(
            i, len(p)
        ):  # Second partials by finite difference; could be done analytically in a future revision
            xhyh, xhyl, xlyh, xlyl = p.copy(), p.copy(), p.copy(), p.copy()
            xdelt = delt if p[i] >= 0 else -delt
            ydelt = delt if p[j] >= 0 else -delt

            xhyh[i] *= 1 + xdelt
            xhyh[j] *= 1 + ydelt

            xhyl[i] *= 1 + xdelt
            xhyl[j] *= 1 - ydelt

            xlyh[i] *= 1 - xdelt
            xlyh[j] *= 1 + ydelt

            xlyl[i] *= 1 - xdelt
            xlyl[j] *= 1 - ydelt

            hessian[i][j] = hessian[j][i] = (
                mf(xhyh, m, *args)
                - mf(xhyl, m, *args)
                - mf(xlyh, m, *args)
                + mf(xlyl, m, *args)
            ) / (p[i] * p[j] * 4 * delt**2)

    mf(
        p0, m, *args
    )  # call likelihood with original values; this resets model and any other values that might be used later
    return hessian


def get_errors(template, total, n=100):
    """This is, I think, for making MC estimates of TOA errors."""
    from scipy.optimize import fmin

    ph0 = template.get_location()

    def logl(phi, *args):
        phases = args[0]
        template.set_overall_phase(phi % 1)
        return -np.log(template(phases)).sum()

    errors = np.empty(n)
    fitvals = np.empty(n)
    errors_r = np.empty(n)
    delta = 0.01
    mean = 0
    for i in range(n):
        template.set_overall_phase(ph0)
        ph = template.random(total)
        results = fmin(logl, ph0, args=(ph,), full_output=1, disp=0)
        phi0, fopt = results[0], results[1]
        fitvals[i] = phi0
        mean += logl(phi0 + delta, ph) - logl(phi0, ph)
        errors[i] = (
            logl(phi0 + delta, ph) - fopt * 2 + logl(phi0 - delta, ph)
        ) / delta**2
        my_delta = errors[i] ** -0.5
        errors_r[i] = (
            logl(phi0 + my_delta, ph) - fopt * 2 + logl(phi0 - my_delta, ph)
        ) / my_delta**2
    print("Mean: %.2f" % (mean / n))
    return fitvals - ph0, errors**-0.5, errors_r**-0.5


def make_err_plot(template, totals=[10, 20, 50, 100, 500], n=1000):
    import pylab as pl

    fvals = []
    errs = []
    bins = np.arange(-5, 5.1, 0.25)
    for tot in totals:
        f, e = get_errors(template, tot, n=n)
        fvals += [f]
        errs += [e]
        pl.hist(
            f / e,
            bins=np.arange(-5, 5.1, 0.5),
            histtype="step",
            density=True,
            label="N = %d" % tot,
        )
    g = lambda x: (np.pi * 2) ** -0.5 * np.exp(-(x**2) / 2)
    dom = np.linspace(-5, 5, 101)
    pl.plot(dom, g(dom), color="k")
    pl.legend()
    pl.axis([-5, 5, 0, 0.5])


def approx_gradient(fitter, eps=1e-6):
    """Numerically approximate the gradient of an instance of one of the
    light curve fitters.

    TODO -- potentially merge this with the code in lcprimitives"""
    func = fitter.template
    orig_p = func.get_parameters(free=True).copy()
    g = np.zeros([len(orig_p)])
    weights = np.asarray([-1, 8, -8, 1]) / (12 * eps)

    def do_step(which, eps):
        p0 = orig_p.copy()
        p0[which] += eps
        # func.set_parameters(p0,free=False)
        return fitter.loglikelihood(p0)

    for i in range(len(orig_p)):
        # use a 4th-order central difference scheme
        for j, w in zip([2, 1, -1, -2], weights):
            g[i] += w * do_step(i, j * eps)

    func.set_parameters(orig_p, free=True)
    return g


def hess_from_grad(grad, par, step=1e-3, iterations=2):
    """Use gradient to compute hessian.  Proceed iteratively to take steps
    roughly equal to the 1-sigma errors.

    The initial step can be:
        [scalar] use the same step for the initial iteration
        [array] specify a step for each parameters.
    """

    def mdet(M):
        """Return determinant of M.
        Use a Laplace cofactor expansion along first row."""
        n = M.shape[0]
        if n == 2:
            return M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if n == 1:
            return M[0, 0]
        rvals = np.zeros(1, dtype=M.dtype)
        toggle = 1.0
        for i in range(n):
            minor = np.delete(np.delete(M, 0, 0), i, 1)
            rvals += M[0, i] * toggle * mdet(minor)
            toggle *= -1
        return rvals

    def minv(M):
        """Return inverse of M, using cofactor expansion."""
        n = M.shape[0]
        C = np.empty_like(M)
        for i in range(n):
            for j in range(n):
                m = np.delete(np.delete(M, i, 0), j, 1)
                C[i, j] = (-1) ** (i + j) * mdet(m)
        det = (M[0, :] * C[0, :]).sum()
        return C.transpose() / det

    # why am I using a co-factor expansion?  Reckon this would better be
    # done as Cholesky in any case
    minv = scipy.linalg.inv

    def make_hess(p0, steps):
        npar = len(par)
        hess = np.empty([npar, npar], dtype=p0.dtype)
        for i in range(npar):
            par[i] = p0[i] + steps[i]
            gup = grad(par)
            par[i] = p0[i] - steps[i]
            gdn = grad(par)
            par[:] = p0
            hess[i, :] = (gup - gdn) / (2 * steps[i])
        return hess

    p0 = par.copy()  # sacrosanct
    if not (hasattr(step, "__len__") and len(step) == len(p0)):
        step = np.ones_like(p0) * step
    hessians = [make_hess(p0, step)]

    for i in range(iterations):
        steps = np.diag(minv(hessians[-1])) ** 0.5
        mask = np.isnan(steps)
        if np.any(mask):
            steps[mask] = step[mask]
        hessians.append(make_hess(p0, steps))

    g = grad(p0)  # reset parameters
    for i in range(iterations, -1, -1):
        if not np.any(np.isnan(np.diag(minv(hessians[i])) ** 0.5)):
            return hessians[i].astype(float)
    return hessians[0].astype(float)


def calc_step_size(logl, par, minstep=1e-5, maxstep=1e-1):
    from scipy.optimize import bisect

    rvals = np.empty_like(par)
    p0 = par.copy()
    ll0 = logl(p0)

    def f(x, i):
        p0[i] = par[i] + x
        delta_ll = logl(p0) - ll0 - 0.5
        p0[i] = par[i]
        return 0 if abs(delta_ll) < 0.05 else delta_ll

    for i in range(len(par)):
        if f(maxstep, i) <= 0:
            rvals[i] = maxstep
        else:
            try:
                rvals[i] = bisect(f, minstep, maxstep, args=(i))
            except ValueError as e:
                print("Unable to compute a step size for parameter %d." % i)
                rvals[i] = maxstep
    logl(par)  # reset parameters
    return rvals
