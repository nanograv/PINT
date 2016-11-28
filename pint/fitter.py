# fitter.py
# Defines the basic TOA fitter class

from fitting_methods import FittingMethod
 #Make fitter class a wrapper class, allows different fitting method


class fitter(object):
    """This is upper level fitter class. By giving method argument, fitter will call
    different fitting method.
    Parameters
    ----------
    toas : a pint TOAs instance
        The input toas.
    model : a pint timing model instance
        The initial timing model for fitting.
    method : str or FittingMethod class.
        The method user select or defined for fitting. 
    """
    def __init__(self, toas, model, method='wls'):
        if isinstance(method, str):
            if method in FittingMethod._method_list.keys():
                self._fitter = FittingMethod._method_list[method](toas, model)
            else:
                raise NameError("Fitting '" + method + "' is not implemented yet.")
        else:
            if issubclass(method, FittingMethod):
                self._fitter = method(toas, model)
            else:
                raise TypeError("'" + method + "' has to be a FittingMethod type.")

    @property
    def model(self):
        return self._fitter.model

    @model.setter
    def model(self, m):
        self._fitter.model = m
        self._fitter.update_resids()

    @property
    def toas(self):
        return self._fitter.toas

    @toas.setter
    def toas(self, t):
        self._fitter.toas = t
        self._fitter.update_resids()

    def __getattr__(self, name):
        if name in dir(self._fitter):
            return getattr(self._fitter, name)
        else:
            return super(fitter, self).__getattribute__(name)
