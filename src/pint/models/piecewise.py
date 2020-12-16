"""Pulsar timing piecewise solution."""
# piecewise.py
# Defines piecewise spindown timing model class
from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np
from astropy import log

from pint.models.parameter import prefixParameter, MJDParameter
from pint.models.timing_model import MissingParameter, PhaseComponent
from pint.utils import split_prefixed_name, taylor_horner, taylor_horner_deriv


class PiecewiseSpindown(PhaseComponent):
    """Pulsar spin-down piecewise solution."""

    register = True
    category = "piecewise"

    def __init__(self):
        super(PiecewiseSpindown, self).__init__()

        self.add_param(
            prefixParameter(
                name="PWSTART_1",
                units="MJD",
                unit_template=lambda x: "MJD",
                description_template=lambda x: "Epoch of solution piece %d" % x,
                parameter_type="MJD",
                time_scale="tdb",
            )
        )

        self.add_param(
            prefixParameter(
                name="PWSTOP_1",
                units="MJD",
                unit_template=lambda x: "MJD",
                description_template=lambda x: "Epoch of solution piece %d" % x,
                parameter_type="MJD",
                time_scale="tdb",
            )
        )
        self.add_param(
            prefixParameter(
                name="PWEP_1",
                units="MJD",
                unit_template=lambda x: "MJD",
                description_template=lambda x: "Epoch of solution piece %d" % x,
                parameter_type="MJD",
                time_scale="tdb",
            )
        )
        self.add_param(
            prefixParameter(
                name="PWPH_1",
                units="",
                value=0.0,
                description_template=lambda x: "Starting phase of solution piece %d"
                % x,
                unit_template=lambda x: "",
                type_match="float",
                uncertainty=1,
            )
        )
        self.add_param(
            prefixParameter(
                name="PWF0_1",
                units="Hz",
                value=0.0,
                description_template=lambda x: "Frequency of solution piece %d" % x,
                unit_template=lambda x: "Hz",
                type_match="float",
            )
        )
        self.add_param(
            prefixParameter(
                name="PWF1_1",
                units="Hz/s",
                value=0.0,
                description_template=lambda x: "Frequency-derivative of solution piece"
                " %d " % x,
                unit_template=lambda x: "Hz/s",
            )
        )
        self.add_param(
            prefixParameter(
                name="PWF2_1",
                units="Hz/s^2",
                value=0.0,
                description_template=lambda x: "Second frequency-"
                "derivative of solution piece"
                " %d " % x,
                unit_template=lambda x: "Hz/s^2",
            )
        )

        self.phase_funcs_component += [self.piecewise_phase]
        # self.phase_derivs_wrt_delay += [self.d_piecewise_phase_d_delay]

    def setup(self):
        super(PiecewiseSpindown, self).setup()
        self.pwsol_prop = [
            "PWEP_",
            "PWSTART_",
            "PWSTOP_",
            "PWPH_",
            "PWF0_",
            "PWF1_",
            "PWF2_",
        ]
        self.pwsol_indices = [
            getattr(self, y).index
            for x in self.pwsol_prop
            for y in self.params
            if x in y
        ]
        # for idx in set(self.pwsol_indices):
        #     for param in self.pwsol_prop:
        #         if not hasattr(self, param + "%d" % idx):
        #             param0 = getattr(self, param + "1")
        #             self.add_param(param0.new_param(idx))
        #             getattr(self, param + "%d" % idx).value = 0.0
        #         self.register_deriv_funcs(
        #             getattr(self, "d_phase_d_" + param[0:-1]), param + "%d" % idx
        #         )
        for idx in set(self.pwsol_indices):
            for param in self.pwsol_prop:
                if param.startswith("PWF") or param.startswith("PWPH"):
                    self.register_deriv_funcs(self.d_phase_d_F, f"{param}{idx}")

    def validate(self):
        """Validate parameters input."""
        super(PiecewiseSpindown, self).validate()
        for idx in set(self.pwsol_indices):
            if not hasattr(self, "PWEP_%d" % idx):
                msg = "Piecewise solution Epoch is needed for Piece %d." % idx
                raise MissingParameter("PiecewiseSpindown", "PWEP_%d" % idx, msg)
            if not hasattr(self, "PWSTART_%d" % idx):
                msg = "Piecewise solution starting epoch is needed for Piece %d." % idx
                raise MissingParameter("PiecewiseSpindown", "PWSTART_%d" % idx, msg)
            if not hasattr(self, "PWSTOP_%d" % idx):
                msg = "Piecewise solution end epoch is needed for Piece %d." % idx
                raise MissingParameter("PiecewiseSpindown", "PWSTOP_%d" % idx, msg)

    def print_par(self):
        result = ""
        for idx in set(self.pwsol_indices):
            for param in self.pwsol_prop:
                par = getattr(self, param + "%d" % idx)
                result += par.as_parfile_line()
        return result

    def get_dt_and_affected(self, toas, delay, glepnm):
        tbl = toas.table
        glep = getattr(self, glepnm)
        idx = glep.index
        start = getattr(self, "PWSTART_%d" % idx).value
        stop = getattr(self, "PWSTOP_%d" % idx).value
        affected = (tbl["tdbld"] >= start) & (tbl["tdbld"] < stop)
        phsepoch_ld = glep.quantity.tdb.mjd_long
        dt = (tbl["tdbld"][affected] - phsepoch_ld) * u.day - delay[affected]
        return dt, affected

    def piecewise_phase(self, toas, delay):
        """Glitch phase function.
        delay is the time delay from the TOA to time of pulse emission
        at the pulsar, in seconds.
        returns an array of phases in long double
        """
        phs = u.Quantity(np.zeros(toas.ntoas, dtype=np.longdouble))
        glepnames = [x for x in self.params if x.startswith("PWEP_")]
        for glepnm in glepnames:
            glep = getattr(self, glepnm)
            idx = glep.index
            # dPH = getattr(self, "PWPH_%d" % idx).quantity
            # dF0 = getattr(self, "PWF0_%d" % idx).quantity
            # dF1 = getattr(self, "PWF1_%d" % idx).quantity
            # dF2 = getattr(self, "PWF2_%d" % idx).quantity
            dt, affected = self.get_dt_and_affected(toas, delay, glepnm)
            # fterms = [dPH, dF0, dF1, dF2]
            fterms = self.get_spin_terms(idx)
            phs[affected] += taylor_horner(dt.to(u.second), fterms)
        return phs.to(u.dimensionless_unscaled)

    # def d_piecewise_phase_d_delay(self, toas, param, delay):
    #     par = getattr(self, param)
    #     unit = par.units
    #     tbl = toas.table
    #     ders = u.Quantity(np.zeros(toas.ntoas, dtype=np.longdouble) * (1 / u.second))
    #     glepnames = [x for x in self.params if x.startswith("PWEP_")]
    #     for glepnm in glepnames:
    #         glep = getattr(self, glepnm)
    #         idx = glep.index
    #         dF0 = getattr(self, "PWF0_%d" % idx).quantity
    #         dF1 = getattr(self, "PWF1_%d" % idx).quantity
    #         dF2 = getattr(self, "PWF2_%d" % idx).quantity
    #         dt, affected = self.get_dt_and_affected(toas, delay, glepnm)
    #         fterms = [0.0 * u.Unit("")] + [dF0, dF1, dF2]
    #         d_pphs_d_delay = taylor_horner_deriv(dt.to(u.second), fterms)
    #         ders[affected] = -d_pphs_d_delay.to(1 / u.second)
    #
    #     return ders.to(1 / unit)

    def get_spin_terms(self, order):
        return [getattr(self, f"PWPH_{order}").quantity] + [
            getattr(self, f"PWF{ii}_{order}").quantity for ii in range(3)
        ]

    def d_phase_d_F(self, toas, param, delay):
        """Calculate the derivative wrt to an spin term."""
        par = getattr(self, param)
        unit = par.units
        pn, idxf, idxv = split_prefixed_name(param)
        if param.startswith("PWF"):
            order = split_prefixed_name(param[:4])[2] + 1
        else:
            order = 0
        # order = idxv + 1
        fterms = self.get_spin_terms(idxv)
        # make the choosen fterms 1 others 0
        fterms = [ft * np.longdouble(0.0) / unit for ft in fterms]
        fterms[order] += np.longdouble(1.0)
        glepnm = f"PWEP_{idxf}"
        res = u.Quantity(np.zeros(toas.ntoas, dtype=np.longdouble)) * (1 / unit)
        dt, affected = self.get_dt_and_affected(toas, delay, glepnm)
        d_pphs_d_f = taylor_horner(dt.to(u.second), fterms)
        res[affected] = d_pphs_d_f.to(1 / unit)
        return res
