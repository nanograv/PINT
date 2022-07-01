import numpy as np

import pint.models
import pint.toa
import pint.residuals

import scipy.special

class Prior:
    def __init__(self, model : pint.models.TimingModel, method : str, **kwargs):
        self.params = model.free_params
        self.method = method
        if self.method in ['uniform', 'normal']:
            values = np.array([getattr(model,param).value for param in self.params])
            uncertainties = np.array([getattr(model,param).uncertainty.value for param in self.params])
            
            if np.any(uncertainties==0):
                raise ValueError("All fitting parameters must have non-zero uncertainties for this method to work. Update uncertainties in the par file or use method='custom'.")
            
            if 'sigma' in kwargs:
                self.sigma = kwargs['sigma']  
            else:
                self.sigma = 1

            if self.method == 'uniform':
                self.mins = values - self.sigma*uncertainties
                self.maxs = values + self.sigma*uncertainties
                self.spans = self.maxs - self.mins
            else:
                self.means = values
                self.stds  = self.sigma*uncertainties
                                    
        elif self.method == 'custom':
            self.info = kwargs['info']
            self._validate_custom_info(self.info)

        else: 
            raise ValueError("Invalid method.")
    
    def lnprior(self, params):
        if self.method == 'uniform':
            return self._uniform_lnprior(params)
        elif self.method == 'normal':
            return self._normal_lnprior(params)
        elif self.method == 'custom':
            return self._custom_lnprior(params)

    def _uniform_lnprior(self, params):
        if np.all(params>=self.mins) and np.all(params<=self.maxs):
            return 0
        else:
            return -np.inf
    
    def _normal_lnprior(self, params):
        return -0.5*((params - self.means)/self.stds)**2

    def _custom_lnprior(self, params):
        lnp = 0
        for param_label, param_val in zip(self.params, params):
            if self.info[param_label]['method'] == 'uniform':
                param_min = self.info[param_label]['min']
                param_max = self.info[param_label]['max']
                if not (param_val>=param_min and param_val<=param_max):
                    return -np.inf
            elif self.info[param_label]['method'] == 'normal':
                mean = self.info[param_label]['mean']
                std = self.info[param_label]['std']
                lnp += -0.5*((param_val - mean)/std)**2
            else:
                raise ValueError("Invalid method.")
        
        return lnp

    def prior_transform(self, cube):
        if self.method == 'uniform':
            return self._uniform_prior_transform(cube)
        elif self.method == 'normal':
            return self._normal_prior_transform(cube)
        elif self.method == 'custom':
            return self._custom_prior_transform(cube)
    
    def _uniform_prior_transform(self, cube):
        return self.mins + cube*self.spans
    
    def _normal_prior_transform(self, cube):
        return self.means + self.stds * scipy.special.ndtri(cube)

    def _custom_prior_transform(self, cube):
        param_vals = np.empty_like(cube)
        for idx, param_label in enumerate(self.params):
            if self.info[param_label]['method'] == 'uniform':
                param_min = self.info[param_label]['min']
                param_max = self.info[param_label]['max']
                param_span = param_max - param_min
                param_vals[idx] = param_min + param_span*cube[idx]
            elif self.info[param_label]['method'] == 'normal':
                mean = self.info[param_label]['mean']
                std = self.info[param_label]['std']
                param_vals[idx] = mean + std * scipy.special.ndtri(cube[idx])
            else:
                raise ValueError("Invalid method.")
        return param_vals

class SPNTA:
    def __init__(self, model : pint.models.TimingModel, toas : pint.toa.TOAs, prior_method='uniform', **kwargs):
        self.model = model
        self.toas = toas

        self._model = model
        self.free_params = model.free_params

        self.likelihood_method = self._decide_likelihood_method()
        
        if prior_method is None:
            self.prior = None
        else:
            self.prior = Prior(model, prior_method, **kwargs)
            self.lnprior = self.prior.lnprior
            self.prior_transform = self.prior.prior_transform

    def _decide_likelihood_method(self):
        if 'NoiseComponent' not in self.model.component_types:
            return 'wls'
        else:
            correlated_errors_present = np.any([nc.introduces_correlated_errors for nc in self.model.NoiseComponent_list])
            if not correlated_errors_present:
                return 'wls_wn'
            else:
                raise NotImplementedError("Likelihood function for correlated noise is not implemented yet.")
    
    def _wls_lnlikelihood(self, params):
        params_dict = dict(zip(self.free_params, params))
        self.model.set_param_values(params_dict)
        res = pint.residuals.Residuals(self.toas, self.model)
        return -res.calc_chi2()/2
    
    def _wls_wn_lnlikelihood(self, params):
        params_dict = dict(zip(self.free_params, params))
        self.model.set_param_values(params_dict)
        res = pint.residuals.Residuals(self.toas, self.model)
        chi2 = res.calc_chi2()
        sigmas = self.model.scaled_toa_uncertainty(self.toas).to('s').value
        return -chi2/2 - np.log(sigmas)
    
    def lnlikelihood(self, params):
        if self.likelihood_method == 'wls':
            return self._wls_lnlikelihood(params)
        elif self.likelihood_method == 'wls_wn':
            return self._wls_wn_lnlikelihood(params)
        else:
            raise NotImplementedError(f"Likelihood function for method {self.likelihood_method} not implemented yet.")

