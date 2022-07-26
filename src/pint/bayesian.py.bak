import numpy as np
import scipy.special

import pint.models
import pint.toa
import pint.residuals

class Prior:
    def __init__(self, model : pint.models.TimingModel, dist : str, width=1, info=None):
        self.params = model.free_params
        self.dist = dist
        if self.dist in ['uniform', 'normal']:
            values = np.array([getattr(model,param).value for param in self.params])
            uncertainties = np.array([getattr(model,param).uncertainty.value for param in self.params])
            
            if np.any(uncertainties==0):
                raise ValueError("All fitting parameters must have non-zero uncertainties for this method to work. Update uncertainties in the par file or use dist='custom'.")
            
            self.width = width

            if self.dist == 'uniform':
                self.mins = values - self.width*uncertainties
                self.maxs = values + self.width*uncertainties
                self.spans = self.maxs - self.mins
            else:
                self.means = values
                self.stds  = self.width*uncertainties
                                    
        elif self.dist == 'custom':
            self.info = info
            #self._validate_custom_info(self.info)

        else: 
            raise ValueError("Invalid dist.")
    
    def lnprior(self, params):
        if self.dist == 'uniform':
            return self._uniform_lnprior(params)
        elif self.dist == 'normal':
            return self._normal_lnprior(params)
        elif self.dist == 'custom':
            return self._custom_lnprior(params)

    def _uniform_lnprior(self, params):
        if np.all(params>=self.mins) and np.all(params<=self.maxs):
            return 0
        else:
            return -np.inf
    
    def _normal_lnprior(self, params):
        return np.sum(-0.5*((params - self.means)/self.stds)**2)

    def _custom_lnprior(self, params):
        lnp = 0
        for param_label, param_val in zip(self.params, params):
            if self.info[param_label]['dist'] == 'uniform':
                param_min = self.info[param_label]['min']
                param_max = self.info[param_label]['max']
                if not (param_val>=param_min and param_val<=param_max):
                    return -np.inf
            elif self.info[param_label]['dist'] == 'normal':
                mean = self.info[param_label]['mean']
                std = self.info[param_label]['std']
                lnp += -0.5*((param_val - mean)/std)**2
            else:
                raise ValueError("Invalid dist.")
        
        return lnp

    def prior_transform(self, cube):
        if self.dist == 'uniform':
            return self._uniform_prior_transform(cube)
        elif self.dist == 'normal':
            return self._normal_prior_transform(cube)
        elif self.dist == 'custom':
            return self._custom_prior_transform(cube)
    
    def _uniform_prior_transform(self, cube):
        return self.mins + cube*self.spans
    
    def _normal_prior_transform(self, cube):
        return self.means + self.stds * scipy.special.ndtri(cube)

    def _custom_prior_transform(self, cube):
        param_vals = np.empty_like(cube)
        for idx, param_label in enumerate(self.params):
            if self.info[param_label]['dist'] == 'uniform':
                param_min = self.info[param_label]['min']
                param_max = self.info[param_label]['max']
                param_span = param_max - param_min
                param_vals[idx] = param_min + param_span*cube[idx]
            elif self.info[param_label]['dist'] == 'normal':
                mean = self.info[param_label]['mean']
                std = self.info[param_label]['std']
                param_vals[idx] = mean + std * scipy.special.ndtri(cube[idx])
            else:
                raise ValueError("Invalid dist.")
        return param_vals

    def sample(self):
        cube = np.random.rand(len(self.params))
        return self.prior_transform(cube)
    
    def get_info_dict(self):
        if self.dist == 'custom':
            return self.info
        else:
            info_dict = dict()
            if self.dist == 'uniform':
                for par, min_val, max_val in zip(self.params, self.mins, self.maxs):
                    info_dict[par] = {  'dist' : 'uniform',
                                        'min'  : min_val,
                                        'max'  : max_val
                                    }
            elif self.dist == 'normal':
                for par, mean, std in zip(self.params, self.means, self.stds):
                    info_dict[par] = {  'dist' : 'normal',
                                        'mean' : mean,
                                        'std'  : std
                                    }
            
            return info_dict

class SPNTA:
    def __init__(self, model : pint.models.TimingModel, toas : pint.toa.TOAs, prior_dist='uniform', prior_width=1, prior_info=None):
        self.model = model
        self.toas = toas

        self._model = model
        self.free_params = model.free_params
        self.initial_params = np.array([getattr(model,param).value for param in self.free_params])
        self.ndim = len(self.free_params)

        self.likelihood_method = self._decide_likelihood_method()

        if prior_dist is None:
            self.prior = None
        else:
            self.prior = Prior(model, prior_dist, width=prior_width, info=prior_info)
            self.lnprior = self.prior.lnprior
            self.prior_transform = self.prior.prior_transform

        # eager initialization.
        _ = self.lnlikelihood(self.initial_params)

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
        return -chi2/2 - np.sum(np.log(sigmas))
    
    def lnlikelihood(self, params):
        if self.likelihood_method == 'wls':
            return self._wls_lnlikelihood(params)
        elif self.likelihood_method == 'wls_wn':
            return self._wls_wn_lnlikelihood(params)
        else:
            raise NotImplementedError(f"Likelihood function for method {self.likelihood_method} not implemented yet.")

    def lnposterior(self, params):
        lnp = self.lnprior(params)

        if np.isfinite(lnp):
            return lnp + self.lnlikelihood(params)
        else:
            return lnp