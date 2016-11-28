# fitter.py
# Defines the basic TOA fitter class

from fitting_methods import FITTING_METHOD, fitter_cls
 #Make fitter class a wrapper class, allows different fitting method


class fitter(object):
    """This is wrapper fitter class. By giving method argument, fitter will call
    different fitter_cls.
    """
    def __init__(self, toas, model, method='wls'):
        if isinstance(method, str):
            if method in FITTING_METHOD.keys():
                self._fitter = FITTING_METHOD[method](toas, model)
            else:
                raise NameError("'" + method + "' is not in the built-in fitting methods.")
        else:
            if issubclass(method, fitter_cls):
                self._fitter = fitter_cls(toas, model)
            else:
                raise TypeError("'" + method + "' has to be a fitter_cls type.")

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
        if name in self._fitter.__dict__.keys():
            return getattr(self._fitter, name)
        else:
            return getattr(self, name)
