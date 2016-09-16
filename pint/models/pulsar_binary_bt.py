
"""This model provides the BT (Blandford & Teukolsky 1976, ApJ, 205, 580) model.
    """
from pint import ls,GMsun,Tsun
from .stand_alone_psr_binaries.BT_model import BTmodel
from .pulsar_binary import PulsarBinary
import parameter as p
from .timing_model import Cache, TimingModel, MissingParameter
import astropy.units as u

class BinaryBT(PulsarBinary):
    """This class provides an implementation of the BT model
        by Blandford & Teukolsky 1976, ApJ, 205, 580.
    """
    def __init__(self):
        super(BinaryBT, self).__init__()
        self.binary_model_name = 'BT'
        self.binary_model_class = BTmodel

        self.add_param(p.floatParameter(name="GAMMA", value=0.0,
             units="second",
             description="Time dilation & gravitational redshift"),
             binary_param = True)

        self.delay_funcs['L2'] += [self.BT_delay,]

    def setup(self):
        super(BinaryBT, self).setup()

        # If any necessary parameter is missing, raise MissingParameter.
        # This will probably be updated after ELL1 model is added.
        for p in ("PB", "T0", "A1"):
            if getattr(self, p).value is None:
                raise MissingParameter("BT", p,
                                       "%s is required for BT" % p)

        # If any *DOT is set, we need T0
        for p in ("PBDOT", "OMDOT", "EDOT", "A1DOT"):
            if getattr(self, p).value is None:
                getattr(self, p).set("0")
                getattr(self, p).frozen = True

            if getattr(self, p).value is not None:
                if self.T0.value is None:
                    raise MissingParameter("BT", "T0",
                        "T0 is required if *DOT is set")

        if self.GAMMA.value is None:
            self.GAMMA.set("0")
            self.GAMMA.frozen = True

        # If eccentricity is zero, freeze some parameters to 0
        # OM = 0 -> T0 = TASC
        if self.ECC.value == 0 or self.ECC.value is None:
            for p in ("E", "OM", "OMDOT", "EDOT"):
                getattr(self, p).set("0")
                getattr(self, p).frozen = True

    @Cache.use_cache
    def get_bt_object(self, toas):
        """
        Obtain the BTmodel object for this set of parameters/toas
        """
        # Don't need to fill P0 and P1. Translate all the others to the format
        # that is used in bmodel.py
        bt_params = ['PB', 'PBDOT', 'ECC', 'EDOT', 'OM', 'OMDOT', 'A1', \
                'A1DOT', 'T0', 'GAMMA']
        aliases = {'A1DOT':'XDOT', 'ECC':'E'}

        pardict = {}
        for par in bt_params:
            key = par if not par in aliases else aliases[par]

            # T0 needs to be converted to long double
            parobj = getattr(self, key)
            pardict[par] = parobj.value

        # Apply all the delay terms, except for the binary model itself
        tt0 = toas['tdbld'] * SECS_PER_DAY
        for df in self.delay_funcs['L1']:
            if df != self.BT_delay:
                tt0 -= df(toas)

        # Return the BTmodel object
        return BTmodel(tt0/SECS_PER_DAY, **pardict)

    @Cache.use_cache
    def BT_delay(self, toas):
        """Return the BT timing model delay"""
        btob = self.get_bt_object(toas)

        return btob.delay()

    @Cache.use_cache
    def d_delay_d_A1(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_A1()

    @Cache.use_cache
    def d_delay_d_XDOT(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_A1DOT()

    @Cache.use_cache
    def d_delay_d_OM(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_OM()

    @Cache.use_cache
    def d_delay_d_OMDOT(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_OMDOT()

    @Cache.use_cache
    def d_delay_d_E(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_ECC()

    @Cache.use_cache
    def d_delay_d_EDOT(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_EDOT()

    @Cache.use_cache
    def d_delay_d_PB(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_PB()

    @Cache.use_cache
    def d_delay_d_PBDOT(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_PBDOT()

    @Cache.use_cache
    def d_delay_d_T0(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_T0()

    @Cache.use_cache
    def d_delay_d_GAMMA(self, toas):
        btob = self.get_bt_object(toas)

        return btob.d_delay_d_GAMMA()
