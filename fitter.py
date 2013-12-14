# fitter.py
# Defines the basic TOA fitter class
import numpy

class resids(object):
    """
    resids(toas=None,model=None)

    """

    def __init__(self, toas=None, model=None):
        self.toas=toas
        self.model=model

    def get_phase(self):
        
        # Compute predicted phase for given input model and TOAs. Right now, the
        # function compute_phase doesn't return an array for an array of input
        # TOAs, so we must loop. We might want to change this.
        return numpy.array([self.model.compute_phase(t.mjd) for t in self.toas.get_mjds()])

    def intPhase(self,ph):
        
        # Convert decimal phases to nearest integer phase
        return numpy.round(ph)

    def get_PSR_freq(self):

        # All residuals require the model pulsar frequency to be defined
        F0names=['F0','nu'] # recognized parameter names
        nF0=0;
        for n in F0names:
            if n in self.model.params:
                F0=getattr(self.model,n).value
                nF0+=1
        
        if nF0==0:
            raise ValueError('no PSR frequency parameter found; ' +
                             'valid names are %s' % F0names)
            
        if nF0>1:
            raise ValueError('more than one PSR frequency parameter found; ' +
                             'should be only one from %s' % F0names)

        return F0

    def calc_resids(self):

        ph = self.get_phase();
        
        return (ph - self.intPhase(ph))/self.get_PSR_freq();

    def calc_chi2(self):
        
        # Residual units are in seconds. Error units are in microseconds.
        return sum(self.calc_resids()/(self.toas.get_errors()*1e-6))

    def get_dof(self):
        
        # Compute number of degrees of freedom
        dof=len(self.toas.toas)
        for p in self.model.params:
            dof -= bool(not getattr(self.model,p).frozen)
            
        return dof

    def get_reduced_chi2(self,chi2,dof):

        return chi2/dof

    

        
    
        
    

    

        
        
            
        
        

    
    
    

        
        
