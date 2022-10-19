"""Bayesian interface providing the pulsar timing likelihood, prior and posterior functions."""

from copy import deepcopy

import numpy as np
from scipy.stats import norm, uniform

from pint.models.priors import Prior, UniformUnboundedRV
from pint.residuals import Residuals


class BayesianTiming:
    """A wrapper around the PINT API that provides lnprior, prior_transform,
    lnlikelihood, and lnposterior functions. This interface can be used to
    draw posterior samples using the sampler of your choice.

    Parameters
    ----------
    model : :class:`pint.models.timing_model.TimingModel`
        Contains the input timing model. The best-fit values stored in this object
        are not used.
    toas : :class:`pint.toa.TOAs`
        Contains the input toas.
    use_pulse_numbers : bool, optional
        How to handle phase wrapping. If True, will use the pulse numbers
        from the toas object while creating :class:`pint.residuals.Residuals`
        objects. Otherwise will use the nearest integer.
    prior_info : dict, optional
        A dict containing the prior information on free parameters. This parameter
        supersedes any priors present in the model.

    Notes
    -----
    1. The `prior` attribute of each free parameter in the `model` object should be set to
       an instance of :class:`pint.models.priors.Prior`.

    2. The parameters of BayesianTiming.model will change for every likelihood function call.
       These parameters in general will not be the best-fit values. Hence, it is NOT a good
       idea to save it as a par file.

    3. Only narow-band TOAs are supported at present.

    4. Currently, only uniform and normal distributions are supported in prior_info. More
       general priors should be set directly in the TimingModel object before creating the
       BayesianTiming object. Here is an example prior_info object:

    `prior_info = { "F0" : {"distr" : "normal", "mu" : 1, "sigma" : 0.00001}, "EFAC1" : {"distr" : "uniform", "pmin" : 0.5, "pmax" : 2.0} }`

    See examples/bayesian-example-NGC6440E.py for detailed example.
    """

    def __init__(self, model, toas, use_pulse_numbers=False, prior_info=None):
        # Make a deep copy to not mess up the original model.
        self.model = deepcopy(model)
        self.toas = toas

        if toas.is_wideband():
            raise NotImplementedError("Wideband TOAs are not yet supported.")

        self.param_labels = self.model.free_params
        self.params = [getattr(self.model, par) for par in self.param_labels]
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
                else:
                    raise NotImplementedError(
                        "Only uniform and normal distributions are supported in prior_info."
                    )

        self._validate_priors()

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
        """Weighted least squares with normalization term (wls), or Generalized
        least-squares with normalization term (gls)."""

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
                    "GLS likelihood for correlated noise is not yet implemented."
                )
                # return "gls"

    def lnprior(self, params):
        """Basic implementation of a factorized log prior.
        More complex priors must be separately implemented.

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
        More complex prior transforms must be separately implemented.

        Args:
            cube (array-like): Sample drawn from a uniform distribution defined in an
            nparams-dimensional unit hypercube.

        Returns:
            ndarray : Sample drawn from the prior distribution
        """
        result = np.array(
            [param.prior._rv.ppf(x) for x, param in zip(cube, self.params)]
        )
        return result

    def lnlikelihood(self, params):
        """The Log-likelihood function. If the model does not contain any noise components or
        if the model contains only uncorrelated noise components, this is equal to -chisq/2
        plus the normalization term containing the noise parameters. If the the model contains
        correlated noise, this is equal to -chisq/2 plus the normalization term where chisq
        is the generalized least-squares metric. For reference, see, e.g., Lentati+ 2013.

        Args:
            params (array-like): Parameters

        Returns:
            float: The value of the log-likelihood at params
        """
        if self.likelihood_method == "wls":
            return self._wls_lnlikelihood(params)
        elif self.likelihood_method == "gls":
            raise NotImplementedError(
                "GLS likelihood for correlated noise is not yet implemented."
            )
        else:
            raise ValueError(f"Unknown likelihood method '{self.likelihood_method}'.")

    def lnposterior(self, params):
        """Log-posterior function. If the prior evaluates to zero, the likelihood
        is not evaluated.

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
        """Implementation of Log-Likelihood function for uncorrelated noise only.
        `wls' stands for weighted least squares. Also includes the normalization
        term to enable sampling over white noise parameters (EFAC and EQUAD).

        Args:
            params (array-like): Parameters

        Returns:
            float: The value of the log-likelihood at params
        """
        params_dict = dict(zip(self.param_labels, params))
        self.model.set_param_values(params_dict)
        res = Residuals(self.toas, self.model, track_mode=self.track_mode)
        chi2 = res.calc_chi2()
        sigmas = self.model.scaled_toa_uncertainty(self.toas).si.value
        return -chi2 / 2 - np.sum(np.log(sigmas))
