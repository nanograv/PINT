from scipy.stats import uniform, norm
from pint.models.priors import UniformUnboundedRV, Prior
from pint.residuals import Residuals
from pint.logging import log

import numpy as np


class BayesianTiming:
    """A wrapper around the PINT API that provides lnprior, prior_transform, lnlikelihood, and lnposterior functions.
    This interface can be used to draw posterior samples using the sampler of your choice.

    Parameters
    ----------
    model : :class:`pint.models.TimingModel`
        The best-fit values stored in this object are not used.
    toas : a :class:`pint.toa.TOAs` instance. Contains the input toas.
    use_pulse_numbers : bool, optional
        How to handle phase wrapping. If True, will use the pulse numbers from the toas object
        while creating :class:`pint.residuals.Residuals` objects. Otherwise will use the nearest integer.
    prior_info : dict, optional
        A dict containing the prior information on free parameters. This parameter supersedes any priors
        present in the model.

    Notes
    -----
    * The `prior` attribute of each free parameter in the `model` object should be set to an instance of
      :class:`pint.models.priors.Prior`.
    * Sampling over white noise parameters is supported, but sampling red noise parameters is not yet implemented.
    """

    def __init__(self, model, toas, use_pulse_numbers=False, prior_info=None):
        self.model = model
        self.toas = toas

        self.param_labels = self.model.free_params
        self.params = [getattr(model, par) for par in self.param_labels]
        self.nparams = len(self.param_labels)

        if prior_info is not None:
            for par in prior_info.keys():
                distr = prior_info[par]["distr"]
                if distr == "uniform":
                    pmax, pmin = prior_info[par]["pmax"], prior_info[par]["pmin"]
                    getattr(self.model, par).prior = Prior(uniform(pmin, pmax - pmin))
                elif distr == "normal":
                    mu, sigma = prior_info[par]["mu"], prior_info[par]["sigma"]
                    getattr(self.model, par).prior = Prior(norm(mu, sigma))

        self._validate_priors()

        # Simple weighted least-squares (wls), Weighted least squares with normalization term for white noise (wls_wn),
        # or Generalized least-squares with normalization term (gls). gls is not implemented yet.
        self.likelihood_method = self._decide_likelihood_method()

        self.track_mode = "use_pulse_numbers" if use_pulse_numbers else "nearest"

    def _validate_priors(self):
        for param in self.params:
            if not hasattr(param, "prior") or param.prior is None:
                raise AttributeError(f"Prior is not set for parameter {param.name}.")
            if isinstance(param.prior._rv, UniformUnboundedRV):
                raise NotImplementedError(
                    f"Unbounded uniform priors are not supported. (param : {param.name})"
                )

    def _decide_likelihood_method(self):
        if "NoiseComponent" not in self.model.component_types:
            return "wls"
        else:
            correlated_errors_present = np.any(
                [
                    nc.introduces_correlated_errors
                    for nc in self.model.NoiseComponent_list
                ]
            )
            if not correlated_errors_present:
                return "wls"
            else:
                raise NotImplementedError(
                    "Likelihood function for correlated noise is not implemented yet."
                )

    def lnprior(self, params):
        """Basic implementation of a factorized log prior.
        More complex priors must be separately implemented

        Args:
            params (array-like): Parameters

        Returns:
            float: Value of the log-prior at params
        """
        if len(params) != self.nparams:
            raise IndexError(
                f"The number of input parameters ({len(params)}) should be the same as the number of free parameters ({self.nparams})."
            )

        lnsum = 0.0
        for param_val, param in zip(params, self.params):
            lnpr = param.prior_pdf(param_val, logpdf=True)
            if lnpr in (np.nan, -np.inf):
                return -np.inf
            else:
                lnsum += lnpr

        return lnsum

    def prior_transform(self, cube):
        """Basic implementation of prior transform for a factorized prior.
        More complex prior transforms must be separately implemented

        Args:
            cube (array-like): Sample drawn from a uniform distribution defined in a nparams-dimensional unit hypercube.

        Returns:
            ndarray : Sample drawn from the prior distribution
        """
        result = np.array(
            [param.prior._rv.ppf(x) for x, param in zip(cube, self.params)]
        )
        return result

    def lnlikelihood(self, params):
        """The Log-likelihood function. If the model does not contain any noise components or
        if the model contains only uncorrelated noise components, this is equal to -chisq/2 plus the
        normalization term containing the noise parameters. If the the model contains
        correlated noise, this is equal to -chisq/2 plus the normalization term where chisq
        is the generalized least-squares metric (Not Implemented yet). For reference, see, e.g.,
            https://doi.org/10.1093/mnras/stt2122

        Args:
            params (array-like): Parameters

        Returns:
            float: The value of the log-likelihood at params
        """
        if self.likelihood_method == "wls":
            return self._wls_lnlikelihood(params)
        elif self.likelihood_method == "gls":
            raise NotImplementedError(
                f"Likelihood function for method gls not implemented yet."
            )
        else:
            raise ValueError(f"Unknown likelihood method '{self.likelihood_method}'.")

    def lnposterior(self, params):
        """Log-posterior function. If the prior evaluates to zero, the likelihood is not evaluated.

        Args:
            params (array-like): Parameters

        Returns:
            float: The value of the log-posterior at params
        """
        lnpr = self.lnprior(params)
        if np.isnan(lnpr):
            return -np.inf
        else:
            return lnpr + self.lnlikelihood(params)

    def _wls_lnlikelihood(self, params):
        params_dict = dict(zip(self.param_labels, params))
        self.model.set_param_values(params_dict)
        res = Residuals(self.toas, self.model, track_mode=self.track_mode)
        chi2 = res.calc_chi2()
        sigmas = self.model.scaled_toa_uncertainty(self.toas).to("s").value
        return -chi2 / 2 - np.sum(np.log(sigmas))

    def scaled_lnprior(self, cube):
        return self.lnprior(self.prior_transform(cube))

    def scaled_prior_transform(self, cube):
        return cube

    def scaled_lnlikelihood(self, cube):
        return self.lnlikelihood(self.prior_transform(cube))

    def scale_samples(self, cubes):
        return np.array(list(map(self.prior_transform, cubes)))
