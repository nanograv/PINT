# fitter.py
# Defines the basic TOA fitter class

class resids(object):
    """
    resids(toas,model=None)

    """

    def __init__(self, toas=None, model=None):
        self.toas=toas
        self.model=model

    def get_resids(self):
        
        # Compute predicted phase for given input model and TOAs

