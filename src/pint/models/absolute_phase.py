"""Timing model absolute phase (TZRMJD, TZRSITE ...)"""
from __future__ import absolute_import, division, print_function

import astropy.units as u
from astropy import log

import pint.toa as toa
from pint.models.parameter import MJDParameter, floatParameter, strParameter
from pint.models.timing_model import MissingParameter, PhaseComponent


class AbsPhase(PhaseComponent):
    """Absolute phase model.

    The model defines the absolute phase's reference time and observatory.

    Note
    ----
    Although this class is condisder as a phase component, it does not
    provide the phase_func
    """

    register = True
    category = "absolute_phase"

    def __init__(self):
        super(AbsPhase, self).__init__()
        self.add_param(
            MJDParameter(
                name="TZRMJD", description="Epoch of the zero phase.", time_scale="utc"
            )
        )
        self.add_param(
            strParameter(
                name="TZRSITE", description="Observatory of the zero phase measured."
            )
        )
        self.add_param(
            floatParameter(
                name="TZRFRQ",
                units=u.MHz,
                description="The frequency of the zero phase mearsured.",
            )
        )

    def setup(self):
        super(AbsPhase, self).setup()

    def validate(self):
        super(AbsPhase, self).validate()
        # Check input Parameters
        if self.TZRMJD.value is None:
            raise MissingParameter(
                "AbsPhase",
                "TZRMJD",
                "TZRMJD is required " "to compute the absolute phase. ",
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
        value of the TZRMJD parmeter, however.
        """
        # NOTE: Using TZRMJD.quantity.jd[1,2] so that the time scale can be properly
        # set to the TZRSITE default timescale (e.g. UTC for TopoObs and TDB for SSB)
        TZR_toa = toa.TOA(
            (self.TZRMJD.quantity.jd1 - 2400000.5, self.TZRMJD.quantity.jd2),
            obs=self.TZRSITE.value,
            freq=self.TZRFRQ.quantity,
        )
        clkc_info = toas.clock_corr_info
        tz = toa.get_TOAs_list(
            [TZR_toa],
            include_bipm=clkc_info["include_bipm"],
            include_gps=clkc_info["include_gps"],
            ephem=toas.ephem,
            planets=toas.planets,
        )
        return tz

    def make_TZR_toa(self, toas):
        """ Calculate the TZRMJD if one not given.
        TZRMJD = first toa after PEPOCH.
        """
        PEPOCH = self.PEPOCH.quantity.value
        # TODO: add warning for PEPOCH far away from center of data?
        TZRMJD = min([i for i in toas.get_mjds() if i > PEPOCH * u.d])
        self.TZRMJD.quantity = TZRMJD.value
        self.setup()
