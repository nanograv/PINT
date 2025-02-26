"""Pulsar timing glitches."""

import astropy.units as u
import numpy as np

from pint.exceptions import MissingParameter
from pint.models.parameter import prefixParameter
from pint.models.timing_model import PhaseComponent
from pint.utils import split_prefixed_name


class Glitch(PhaseComponent):
    """Pulsar spin-down glitches.

    Parameters supported:

    .. paramtable::
        :class: pint.models.glitch.Glitch
    """

    @classmethod
    def _description_glitch_phase(cls, x):
        return f"Phase change for glitch {x}"

    @classmethod
    def _description_glitch_epoch(cls, x):
        return f"Epoch of glitch {x}"

    @classmethod
    def _description_glitch_frequencychange(cls, x):
        return (f"Permanent frequency change for glitch {x}",)

    @classmethod
    def _description_glitch_frequencyderivativechange(cls, x):
        return (f"Permanent frequency-derivative change for glitch {x}",)

    @classmethod
    def _description_glitch_frequencysecondderivativechange(cls, x):
        return (f"Permanent second frequency-derivative change for glitch {x}",)

    @classmethod
    def _description_decaying_frequencychange(cls, x):
        return f"Decaying frequency change for glitch {x}"

    @classmethod
    def _description_decaytimeconstant(cls, x):
        return f"Decay time constant for glitch {x}"

    register = True
    category = "glitch"

    def __init__(self):
        super().__init__()

        self.add_param(
            prefixParameter(
                name="GLPH_1",
                units="pulse phase",
                value=0.0,
                description_template=self._description_glitch_phase,
                type_match="float",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            prefixParameter(
                name="GLEP_1",
                units="MJD",
                description_template=self._description_glitch_epoch,
                parameter_type="MJD",
                time_scale="tdb",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            prefixParameter(
                name="GLF0_1",
                units="Hz",
                value=0.0,
                description_template=self._description_glitch_frequencychange,
                type_match="float",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            prefixParameter(
                name="GLF1_1",
                units="Hz/s",
                value=0.0,
                description_template=self._description_glitch_frequencyderivativechange,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            prefixParameter(
                name="GLF2_1",
                units="Hz/s^2",
                value=0.0,
                description_template=self._description_glitch_frequencysecondderivativechange,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            prefixParameter(
                name="GLF0D_1",
                units="Hz",
                value=0.0,
                description_template=self._description_decaying_frequencychange,
                type_match="float",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )

        self.add_param(
            prefixParameter(
                name="GLTD_1",
                units="day",
                value=0.0,
                description_template=self._description_decaytimeconstant,
                type_match="float",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.phase_funcs_component += [self.glitch_phase]

    def setup(self):
        super().setup()
        # Check for required glitch epochs, set not specified parameters to 0
        self.glitch_prop = [
            "GLEP_",
            "GLPH_",
            "GLF0_",
            "GLF1_",
            "GLF2_",
            "GLF0D_",
            "GLTD_",
        ]
        self.glitch_indices = [
            getattr(self, y).index
            for x in self.glitch_prop
            for y in self.params
            if x in y
        ]
        for idx in set(self.glitch_indices):
            for param in self.glitch_prop:
                check = f"{param}{idx}"
                if not hasattr(self, check):
                    param0 = getattr(self, f"{param}1")
                    self.add_param(param0.new_param(idx))
                    getattr(self, check).value = 0.0
                self.register_deriv_funcs(
                    getattr(self, f"d_phase_d_{param[:-1]}"), check
                )

    def validate(self):
        """Validate parameters input."""
        super().validate()
        for idx in set(self.glitch_indices):
            glep = f"GLEP_{idx}"
            glph = f"GLPH_{idx}"
            if (not hasattr(self, glep)) or (getattr(self, glep).quantity is None):
                msg = f"Glitch Epoch is needed for Glitch {idx}"
                raise MissingParameter("Glitch", glep, msg)
            # Check to see if both the epoch and phase are to be fit
            if (
                hasattr(self, glph)
                and (not getattr(self, glep).frozen)
                and (not getattr(self, glph).frozen)
            ):
                raise ValueError(
                    f"Both the glitch epoch and phase cannot be fit for Glitch {idx}."
                )

        # Check the Decay Term.
        glf0dparams = [x for x in self.params if x.startswith("GLF0D_")]
        for glf0dnm in glf0dparams:
            glf0d = getattr(self, glf0dnm)
            idx = glf0d.index
            if glf0d.value != 0.0 and getattr(self, f"GLTD_{idx}").value == 0.0:
                msg = f"Non-zero GLF0D_{idx} parameter needs a non-zero GLTD_{idx} parameter"
                raise MissingParameter("Glitch", f"GLTD_{idx}", msg)

    def print_par(self, format="pint"):
        result = ""
        for idx in set(self.glitch_indices):
            for param in self.glitch_prop:
                par = getattr(self, f"{param}{idx}")
                result += par.as_parfile_line(format=format)
        return result

    def glitch_phase(self, toas, delay):
        """Glitch phase function.
        delay is the time delay from the TOA to time of pulse emission
        at the pulsar, in seconds.
        returns an array of phases in long double
        """
        tbl = toas.table
        phs = u.Quantity(np.zeros(toas.ntoas, dtype=np.longdouble))
        glepnames = [x for x in self.params if x.startswith("GLEP_")]
        for glepnm in glepnames:
            glep = getattr(self, glepnm)
            idx = glep.index
            eph = glep.value
            dphs = getattr(self, f"GLPH_{idx}").quantity
            dF0 = getattr(self, f"GLF0_{idx}").quantity
            dF1 = getattr(self, f"GLF1_{idx}").quantity
            dF2 = getattr(self, f"GLF2_{idx}").quantity
            dt = (tbl["tdbld"] - eph) * u.day - delay
            dt = dt.to(u.second)
            affected = dt > 0.0  # TOAs affected by glitch
            # decay term
            dF0D = getattr(self, f"GLF0D_{idx}").quantity
            if dF0D != 0.0:
                tau = getattr(self, f"GLTD_{idx}").quantity
                decayterm = dF0D * tau * (1.0 - np.exp(-(dt[affected] / tau)))
            else:
                decayterm = u.Quantity(0)

            phs[affected] += (
                dphs
                + dt[affected]
                * (
                    dF0
                    + 0.5 * dt[affected] * dF1
                    + 1.0 / 6.0 * dt[affected] * dt[affected] * dF2
                )
                + decayterm
            )
        return phs

    def deriv_prep(self, toas, param, delay, check_param):
        """Get the things we need for any of the derivative calcs"""
        p, ids, idv = split_prefixed_name(param)
        if p != f"{check_param}_":
            raise ValueError(
                f"Can not calculate d_phase_d_{check_param} with respect to {param}."
            )
        tbl = toas.table
        eph = getattr(self, f"GLEP_{ids}").value
        dt = (tbl["tdbld"] - eph) * u.day - delay
        dt = dt.to(u.second)
        affected = np.where(dt > 0.0)[0]
        par = getattr(self, param)
        zeros = np.zeros(len(tbl), dtype=np.longdouble) << 1 / par.units
        return tbl, p, ids, idv, dt, affected, par, zeros

    def d_phase_d_GLPH(self, toas, param, delay):
        """Calculate the derivative wrt GLPH"""
        tbl, p, ids, idv, dt, affected, par_GLPH, dpdGLPH = self.deriv_prep(
            toas, param, delay, "GLPH"
        )
        dpdGLPH[affected] = 1.0 / par_GLPH.units
        return dpdGLPH

    def d_phase_d_GLF0(self, toas, param, delay):
        """Calculate the derivative wrt GLF0"""
        tbl, p, ids, idv, dt, affected, par_GLF0, dpdGLF0 = self.deriv_prep(
            toas, param, delay, "GLF0"
        )
        dpdGLF0[affected] = dt[affected]
        return dpdGLF0

    def d_phase_d_GLF1(self, toas, param, delay):
        """Calculate the derivative wrt GLF1"""
        tbl, p, ids, idv, dt, affected, par_GLF1, dpdGLF1 = self.deriv_prep(
            toas, param, delay, "GLF1"
        )
        dpdGLF1[affected] = 0.5 * dt[affected] ** 2
        return dpdGLF1

    def d_phase_d_GLF2(self, toas, param, delay):
        """Calculate the derivative wrt GLF1"""
        tbl, p, ids, idv, dt, affected, par_GLF2, dpdGLF2 = self.deriv_prep(
            toas, param, delay, "GLF2"
        )
        dpdGLF2[affected] = (1.0 / 6.0) * dt[affected] ** 3
        return dpdGLF2

    def d_phase_d_GLF0D(self, toas, param, delay):
        """Calculate the derivative wrt GLF0D"""
        tbl, p, ids, idv, dt, affected, par_GLF0D, dpdGLF0D = self.deriv_prep(
            toas, param, delay, "GLF0D"
        )
        tau = getattr(self, f"GLTD_{ids}").quantity
        dpdGLF0D[affected] = tau * (1.0 - np.exp(-dt[affected] / tau))
        return dpdGLF0D

    def d_phase_d_GLTD(self, toas, param, delay):
        """Calculate the derivative wrt GLTD"""
        tbl, p, ids, idv, dt, affected, par_GLTD, dpdGLTD = self.deriv_prep(
            toas, param, delay, "GLTD"
        )
        if par_GLTD.value == 0.0:
            return dpdGLTD
        glf0d = getattr(self, f"GLF0D_{ids}").quantity
        tau = par_GLTD.quantity
        et = np.exp(-dt[affected] / tau)
        dpdGLTD[affected] = glf0d * (
            1.0 - np.exp(-dt[affected] / tau) * (1.0 + dt[affected] / tau)
        )
        return dpdGLTD

    def d_phase_d_GLEP(self, toas, param, delay):
        """Calculate the derivative wrt GLEP"""
        tbl, p, ids, idv, dt, affected, par_GLEP, dpdGLEP = self.deriv_prep(
            toas, param, delay, "GLEP"
        )
        glf0 = getattr(self, f"GLF0_{ids}").quantity
        glf1 = getattr(self, f"GLF1_{ids}").quantity
        glf2 = getattr(self, f"GLF2_{ids}").quantity
        glf0d = getattr(self, f"GLF0D_{ids}").quantity
        tau = getattr(self, f"GLTD_{ids}").quantity
        dpdGLEP[affected] += (
            -glf0 + -glf1 * dt[affected] + -0.5 * glf2 * dt[affected] ** 2
        )
        if tau.value != 0.0:
            dpdGLEP[affected] -= glf0d * np.exp(-dt[affected] / tau)
        return dpdGLEP
