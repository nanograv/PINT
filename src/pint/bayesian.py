from pint.models.priors import UniformUnboundedRV
import numpy as np
from scipy.stats import rv_continuous, uniform, norm
from scipy.special import ndtri
from pint.residuals import Residuals


class BayesianTiming:
    def __init__(self, model, toas):
        self.model = model
        self.toas = toas

        self.param_labels = self.model.free_params
        self.params = [getattr(model, par) for par in self.param_labels]
        self.nparams = len(self.param_labels)

        self._validate_priors()

        self.likelihood_method = self._decide_likelihood_method()

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
                return "wls_wn"
            else:
                raise NotImplementedError(
                    "Likelihood function for correlated noise is not implemented yet."
                )

    def lnprior(self, params):
        """Basic implementation of a factorized log prior."""
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
        """Basic implementation of prior transform for a factorized prior."""
        result = np.array(
            [param.prior._rv.ppf(x) for x, param in zip(cube, self.params)]
        )
        return result

    def lnlikelihood(self, params):
        if self.likelihood_method == "wls":
            return self._wls_lnlikelihood(params)
        elif self.likelihood_method == "wls_wn":
            return self._wls_wn_lnlikelihood(params)
        else:
            raise NotImplementedError(
                f"Likelihood function for method {self.likelihood_method} not implemented yet."
            )

    def _wls_lnlikelihood(self, params):
        params_dict = dict(zip(self.param_labels, params))
        self.model.set_param_values(params_dict)
        res = Residuals(self.toas, self.model)
        return -res.calc_chi2() / 2

    def _wls_wn_lnlikelihood(self, params):
        params_dict = dict(zip(self.param_labels, params))
        self.model.set_param_values(params_dict)
        res = Residuals(self.toas, self.model)
        chi2 = res.calc_chi2()
        sigmas = self.model.scaled_toa_uncertainty(self.toas).to("s").value
        return -chi2 / 2 - np.sum(np.log(sigmas))

    def scaled_lnprior(self, cube):
        return self.lnprior(self.prior_transform(cube))

    def scaled_prior_transform(self, cube):
        return cube

    def scaled_lnlikelihood(self, cube):
        return self.lnlikelihood(self.prior_transform(cube))
