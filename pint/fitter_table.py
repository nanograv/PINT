import numpy as np
import astropy.table as tb
import scipy.optimize as opt
import copy 

class fitter_table(object):
    """
    A class do fitting
    """
    def __init__(self,resids,model,toa, method = None):
        self.method = method
        self.resids = resids
        self.model = model
        self.toa = toa
        self.chi2 = 0.0
        self.rms = 0.0
        self.fit_params = []
        self.new_model = copy.deepcopy(model)
         
    def fit_func(self,params0):
        """
        Fit_funcion for minimize. 
        Params0 is a list of initial value of parameters
        Return is the Chi^2 for residuls.  
        """
        for pars,valus in zip(self.fit_params,params0):
            getattr(self.new_model, pars).value = valus     
        pt = self.new_model.phase_table(self.toa)
        chi2 = np.sum(pt.frac**2)
        return chi2

    def do_fit(self,method = None):
        
        # Check fitting parameters
        fit_value = []
        for par in self.model.params:
            if getattr(self.new_model,par).frozen == False:
                self.fit_params.append(par)
                fit_value.append(np.longdouble(getattr(self.model,par).value)) 
        if self.fit_params == []:
            print "No parameters need to be fitted." 
            return None

        # Do fitting, using the scipy optimize minimize function
        if method == None:
            res = opt.minimize(self.fit_func,fit_value)
        else:     
            res = opt.minimize(self.fit_func,fit_value,method = method, options={'gtol': 1e-6, 'disp': True})
        print res
        for par,new_value in zip(self.model.params,res.x):
            getattr(self.new_model,par).value = new_value
        return 
