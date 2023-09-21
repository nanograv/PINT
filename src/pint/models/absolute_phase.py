"""Timing model absolute phase (TZRMJD, TZRSITE ...)"""
import astropy.units as u
from loguru import logger as log

import pint.toa as toa
from pint.models.parameter import MJDParameter, floatParameter, strParameter
from pint.models.timing_model import MissingParameter, PhaseComponent


class AbsPhase(PhaseComponent):
    """Absolute phase model.

    The model defines the absolute phase's reference time and observatory.

    Parameters supported:

    .. paramtable::
        :class: pint.models.absolute_phase.AbsPhase

    Note
    ----
    Although this class is considered as a phase component, it does not
    provide the phase_func
    """

    register = True
    category = "absolute_phase"

    def __init__(self):
        super().__init__()
        self.add_param(
            MJDParameter(
                name="TZRMJD",
                description="Epoch of the zero phase TOA.",
                time_scale="utc",
            )
        )
        self.add_param(
            strParameter(
                name="TZRSITE", description="Observatory of the zero phase TOA."
            )
        )
        self.add_param(
            floatParameter(
                name="TZRFRQ",
                units=u.MHz,
                description="The frequency of the zero phase TOA.",
            )
        )
        self.tz_cache = None

    def setup(self):
        super().setup()

    def validate(self):
        super().validate()
        # Make sure the cached TOA is cleared
        self.tz_cache = None
        # Check input Parameters
        if self.TZRMJD.value is None:
            raise MissingParameter(
                "AbsPhase",
                "TZRMJD",
                "TZRMJD is required to compute the absolute phase.",
            )
        if self.TZRSITE.value is None:
            self.TZRSITE.value = "ssb"
            # update the TZRMJD to new time scale
            self.TZRMJD.time_scale = "tdb"
            log.info("The TZRSITE is set at the solar system barycenter.")

        if (self.TZRFRQ.value is None) or (self.TZRFRQ.value == 0.0):
            self.TZRFRQ.quantity = float("inf") * u.MHz
            log.info("TZRFRQ was 0.0 or None. Setting to infinite frequency.")

    def get_TZR_toa(self, toas):
        """Get the TOAs class for the TZRMJD.

        We are treating the TZRMJD as a special TOA.
        Note that any observatory clock corrections will be applied
        to this TOA, as with any other TOA. This does not affect the
        value of the TZRMJD parameter, however.
        """
        clkc_info = toas.clock_corr_info
        # If we have cached the TZR TOA and all the TZR* and clock info has not changed, then don't rebuild it
        if self.tz_cache is not None and (
            self.tz_clkc_info["include_bipm"] == clkc_info["include_bipm"]
            and self.tz_clkc_info["include_gps"] == clkc_info["include_gps"]
            and self.tz_planets == toas.planets
            and self.tz_ephem == toas.ephem
            and self.tz_hash
            == hash((self.TZRMJD.value, self.TZRSITE.value, self.TZRFRQ.value))
        ):
            return self.tz_cache
        # Otherwise we have to build the TOA and apply clock corrections
        # NOTE: Using TZRMJD.quantity.jd[1,2] so that the time scale can be properly
        # set to the TZRSITE default timescale (e.g. UTC for TopoObs and TDB for SSB)
        log.debug("Creating and dealing with the single TZR_toa for absolute phase")
        # TZR_toa = toa.TOA(
        #     (self.TZRMJD.quantity.jd1 - 2400000.5, self.TZRMJD.quantity.jd2),
        #     obs=self.TZRSITE.value,
        #     freq=self.TZRFRQ.quantity,
        # )
        # tz = toa.get_TOAs_list(
        #     [TZR_toa],
        #     include_bipm=clkc_info["include_bipm"],
        #     include_gps=clkc_info["include_gps"],
        #     ephem=toas.ephem,
        #     planets=toas.planets,
        # )
        tz = toa.get_TOAs_array(
            (self.TZRMJD.quantity.jd1 - 2400000.5, self.TZRMJD.quantity.jd2),
            obs=self.TZRSITE.value,
            freqs=self.TZRFRQ.quantity,
            include_bipm=clkc_info["include_bipm"],
            include_gps=clkc_info["include_gps"],
            ephem=toas.ephem,
            planets=toas.planets,
            tzr=True,
        )
        log.debug("Done with TZR_toa")
        self.tz_cache = tz
        self.tz_hash = hash((self.TZRMJD.value, self.TZRSITE.value, self.TZRFRQ.value))
        self.tz_clkc_info = clkc_info
        self.tz_planets = toas.planets
        self.tz_ephem = toas.ephem
        return tz

    def make_TZR_toa(self, toas):
        """Calculate the TZRMJD if one not given.

        TZRMJD = first toa after PEPOCH.
        """
        PEPOCH = self._parent.PEPOCH.quantity.mjd
        # TODO: add warning for PEPOCH far away from center of data?
        later = [i for i in toas.get_mjds() if i > PEPOCH * u.d]
        earlier = [i for i in toas.get_mjds() if i <= PEPOCH * u.d]
        TZRMJD = min(later) if later else max(earlier)
        self.TZRMJD.quantity = TZRMJD.value
        self.setup()
