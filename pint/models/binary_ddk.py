from pint import ls,GMsun,Tsun
from .stand_alone_psr_binaries.DDK_model import DDKmodel
from .binary_dd import BinaryDD
from . import parameter as p
from .timing_model import MissingParameter
import astropy.units as u
from astropy import log


class BinaryDDK(BinaryDD):
    """This is a PINT pulsar binary ddk model class a subclass of DD model.
    It is a wrapper for independent DDKmodel class defined in
    ./stand_alone_psr_binary/DDK_model.py
    All the detailed calculations are in the independent DDKmodel.
    The aim for this class is to connect the independent binary model with PINT platform
    DDKmodel special parameters:
    KIN inclination angle
    KOM the longitude of the ascending node, Kopeikin (1995) Eq 9. OMEGA
    Reference
    ---------
    KOPEIKIN. 1995, 1996
    """
    register = True
    def __init__(self,):
        super(BinaryDDK, self).__init__()
        self.binary_model_name = 'DDK'
        self.binary_model_class = DDKmodel

        self.add_param(p.floatParameter(name='KIN', value=0.0, units="deg",
                       description="Inclination angle"))
        self.add_param(p.floatParameter(name='KOM', value=0.0, units="deg",
                       description="The longitude of the ascending node"))
        self.interal_params += ['PMRA_DDK', 'PMDEC_DDK']

    @property
    def PMRA_DDK(self):
        if 'PMRA' in self._parent.params:
            return self.PMRA
        elif 'PMELONG' in self._parent.params:
            return self.PMELONG
        else:
            raise MissingParameter("DDK", "DDK model needs proper motion parameters.")

    @property
    def PMDEC_DDK(self):
        if 'PMDEC' in self._parent.params:
            return self.PMDEC
        elif 'PMELAT' in self._parent.params:
            return self.PMELAT
        else:
            raise MissingParameter("DDK", "DDK model needs proper motion parameters.")


    def setup(self):
        """Check out parameters setup.
        """
        super(BinaryDDK,self).setup()

        if 'PMRA' not in self._parent.params or 'PMDEC' not in self._parent.params:
            # Check ecliptic coordinates proper motion.
            if 'PMELONG' not in self._parent.params or 'PMELAT' not in self._parent.params:
                raise MissingParameter("DDK", "DDK model needs proper motion parameters.")
            else:
                log.info("Using ecliptic coordinate. The parameter KOM is"
                         " measured respect to ecliptic North.")

        else:
            log.info("Using equatorial coordinate. The parameter KOM is"
                     " measured respect to equatorial North.")
