"""Pulsar timing glitches."""
import astropy.units as u
import numpy as np

from loguru import logger as log

from pint.models.parameter import prefixParameter
from pint.models.timing_model import MissingParameter, PhaseComponent
from pint.utils import split_prefixed_name


class Glitch(PhaseComponent):
    """Pulsar spin-down glitches.

    Parameters supported:

    .. paramtable::
        :class: pint.models.glitch.Glitch
    """

    @classmethod
    def _description_glitch_phase(cls, x):
        return "Phase change for glitch %d" % x

    @classmethod
    def _description_glitch_epoch(cls, x):
        return "Epoch of glitch %d" % x

    @classmethod
    def _description_glitch_frequencychange(cls, x):
        return ("Permanent frequency change for glitch %d" % x,)

    @classmethod
    def _description_glitch_frequencyderivativechange(cls, x):
        return ("Permanent frequency-derivative change for glitch %d" % x,)

    @classmethod
    def _description_glitch_frequencysecondderivativechange(cls, x):
        return ("Permanent second frequency-derivative change for glitch %d" % x,)

    @classmethod
    def _description_decaying_frequencychange(cls, x):
        return "Decaying frequency change for glitch %d " % x

    @classmethod
    def _description_decaytimeconstant(cls, x):
        return "Decay time constant for glitch %d" % x

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
            )
        )
        self.add_param(
            prefixParameter(
                name="GLEP_1",
                units="MJD",
                description_template=self._description_glitch_epoch,
                parameter_type="MJD",
                time_scale="tdb",
            )
        )
        self.add_param(
            prefixParameter(
                name="GLF0_1",
                units="Hz",
                value=0.0,
                description_template=self._description_glitch_frequencychange,
                type_match="float",
            )
        )
        self.add_param(
            prefixParameter(
                name="GLF1_1",
                units="Hz/s",
                value=0.0,
                description_template=self._description_glitch_frequencyderivativechange,
            )
        )
        self.add_param(
            prefixParameter(
                name="GLF2_1",
                units="Hz/s^2",
                value=0.0,
                description_template=self._description_glitch_frequencysecondderivativechange,
            )
        )
        self.add_param(
            prefixParameter(
                name="GLF0D_1",
                units="Hz",
                value=0.0,
                description_template=self._description_decaying_frequencychange,
                type_match="float",
            )
        )

        self.add_param(
            prefixParameter(
                name="GLTD_1",
                units="day",
                value=0.0,
                description_template=self._description_decaytimeconstant,
                type_match="float",
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
                if not hasattr(self, param + "%d" % idx):
                    param0 = getattr(self, f"{param}1")
                    self.add_param(param0.new_param(idx))
                    getattr(self, param + "%d" % idx).value = 0.0
                self.register_deriv_funcs(
                    getattr(self, f"d_phase_d_{param[:-1]}"), param + "%d" % idx
                )

    def validate(self):
        """Validate parameters input."""
        super().validate()
        for idx in set(self.glitch_indices):
            if not hasattr(self, "GLEP_%d" % idx):
                msg = "Glitch Epoch is needed for Glitch %d." % idx
                raise MissingParameter("Glitch", "GLEP_%d" % idx, msg)
            else:  # Check to see if both the epoch and phase are to be fit
                if (
                    hasattr(self, "GLPH_%d" % idx)
                    and (not getattr(self, "GLEP_%d" % idx).frozen)
                    and (not getattr(self, "GLPH_%d" % idx).frozen)
                ):
                    raise ValueError(
                        "Both the glitch epoch and phase cannot be fit for Glitch %d."
                        % idx
                    )

        # Check the Decay Term.
        glf0dparams = [x for x in self.params if x.startswith("GLF0D_")]
        for glf0dnm in glf0dparams:
            glf0d = getattr(self, glf0dnm)
            idx = glf0d.index
            if glf0d.value != 0.0 and getattr(self, "GLTD_%d" % idx).value == 0.0:
                msg = (
                    "None zero GLF0D_%d parameter needs a none"
                    " zero GLTD_%d parameter" % (idx, idx)
                )
                raise MissingParameter("Glitch", "GLTD_%d" % idx, msg)

    def print_par(self, format="pint"):
        result = ""
        for idx in set(self.glitch_indices):
            for param in self.glitch_prop:
                par = getattr(self, param + "%d" % idx)
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
            dphs = getattr(self, "GLPH_%d" % idx).quantity
            dF0 = getattr(self, "GLF0_%d" % idx).quantity
            dF1 = getattr(self, "GLF1_%d" % idx).quantity
            dF2 = getattr(self, "GLF2_%d" % idx).quantity
            dt = (tbl["tdbld"] - eph) * u.day - delay
            dt = dt.to(u.second)
            affected = dt > 0.0  # TOAs affected by glitch
            # decay term
            dF0D = getattr(self, "GLF0D_%d" % idx).quantity
            if dF0D != 0.0:
                tau = getattr(self, "GLTD_%d" % idx).quantity
                decayterm = dF0D * tau * (1.0 - np.exp(-(dt[affected] / tau)))
            else:
                decayterm = 0.0 * u.Unit("")

            log.debug(f"Glitch phase for glitch {idx}: {dphs} {dphs.unit}")
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

    def deriv_prep(self, toas, param, delay):
        """Get the things we need for any of the derivative calcs"""
        tbl = toas.table
        p, ids, idv = split_prefixed_name(param)
        eph = getattr(self, f"GLEP_{ids}").value
        dt = (tbl["tdbld"] - eph) * u.day - delay
        dt = dt.to(u.second)
        affected = np.where(dt > 0.0)[0]
        return tbl, p, ids, idv, dt, affected

    def d_phase_d_GLPH(self, toas, param, delay):
        """Calculate the derivative wrt GLPH"""
        tbl, p, ids, idv, dt, affected = self.deriv_prep(toas, param, delay)
        if p != "GLPH_":
            raise ValueError(
                f"Can not calculate d_phase_d_GLPH with respect to {param}."
            )
        par_GLPH = getattr(self, param)
        dpdGLPH = np.zeros(len(tbl), dtype=np.longdouble) / par_GLPH.units
        dpdGLPH[affected] += 1.0 / par_GLPH.units
        return dpdGLPH

    def d_phase_d_GLF0(self, toas, param, delay):
        """Calculate the derivative wrt GLF0"""
        tbl, p, ids, idv, dt, affected = self.deriv_prep(toas, param, delay)
        if p != "GLF0_":
            raise ValueError(
                f"Can not calculate d_phase_d_GLF0 with respect to {param}."
            )
        par_GLF0 = getattr(self, param)
        dpdGLF0 = np.zeros(len(tbl), dtype=np.longdouble) / par_GLF0.units
        dpdGLF0[affected] = dt[affected]
        return dpdGLF0

    def d_phase_d_GLF1(self, toas, param, delay):
        """Calculate the derivative wrt GLF1"""
        tbl, p, ids, idv, dt, affected = self.deriv_prep(toas, param, delay)
        if p != "GLF1_":
            raise ValueError(
                f"Can not calculate d_phase_d_GLF1 with respect to {param}."
            )
        par_GLF1 = getattr(self, param)
        dpdGLF1 = np.zeros(len(tbl), dtype=np.longdouble) / par_GLF1.units
        dpdGLF1[affected] += np.longdouble(0.5) * dt[affected] * dt[affected]
        return dpdGLF1

    def d_phase_d_GLF2(self, toas, param, delay):
        """Calculate the derivative wrt GLF1"""
        tbl, p, ids, idv, dt, affected = self.deriv_prep(toas, param, delay)
        if p != "GLF2_":
            raise ValueError(
                f"Can not calculate d_phase_d_GLF2 with respect to {param}."
            )
        par_GLF2 = getattr(self, param)
        dpdGLF2 = np.zeros(len(tbl), dtype=np.longdouble) / par_GLF2.units
        dpdGLF2[affected] += (
            np.longdouble(1.0) / 6.0 * dt[affected] * dt[affected] * dt[affected]
        )
        return dpdGLF2

    def d_phase_d_GLF0D(self, toas, param, delay):
        """Calculate the derivative wrt GLF0D"""
        tbl, p, ids, idv, dt, affected = self.deriv_prep(toas, param, delay)
        if p != "GLF0D_":
            raise ValueError(
                f"Can not calculate d_phase_d_GLF0D with respect to {param}."
            )
        par_GLF0D = getattr(self, param)
        tau = getattr(self, "GLTD_%d" % idv).quantity
        dpdGLF0D = np.zeros(len(tbl), dtype=np.longdouble) / par_GLF0D.units
        dpdGLF0D[affected] += tau * (np.longdouble(1.0) - np.exp(-dt[affected] / tau))
        return dpdGLF0D

    def d_phase_d_GLTD(self, toas, param, delay):
        """Calculate the derivative wrt GLTD"""
        tbl, p, ids, idv, dt, affected = self.deriv_prep(toas, param, delay)
        if p != "GLTD_":
            raise ValueError(
                f"Can not calculate d_phase_d_GLTD with respect to {param}."
            )
        par_GLTD = getattr(self, param)
        if par_GLTD.value == 0.0:
            return np.zeros(len(tbl), dtype=np.longdouble) / par_GLTD.units
        glf0d = getattr(self, f"GLF0D_{ids}").quantity
        tau = par_GLTD.quantity
        dpdGLTD = np.zeros(len(tbl), dtype=np.longdouble) / par_GLTD.units
        dpdGLTD[affected] += glf0d * (
            np.longdouble(1.0) - np.exp(-dt[affected] / tau)
        ) + glf0d * tau * (-np.exp(-dt[affected] / tau)) * dt[affected] / (tau * tau)
        return dpdGLTD

    def d_phase_d_GLEP(self, toas, param, delay):
        """Calculate the derivative wrt GLEP"""
        tbl, p, ids, idv, dt, affected = self.deriv_prep(toas, param, delay)
        if p != "GLEP_":
            raise ValueError(
                f"Can not calculate d_phase_d_GLEP with respect to {param}."
            )
        par_GLEP = getattr(self, param)
        glf0 = getattr(self, f"GLF0_{ids}").quantity
        glf1 = getattr(self, f"GLF1_{ids}").quantity
        glf2 = getattr(self, f"GLF2_{ids}").quantity
        glf0d = getattr(self, f"GLF0D_{ids}").quantity
        tau = getattr(self, f"GLTD_{ids}").quantity
        dpdGLEP = np.zeros(len(tbl), dtype=np.longdouble) / par_GLEP.units
        dpdGLEP[affected] += (
            -glf0 + -glf1 * dt[affected] + -0.5 * glf2 * dt[affected] ** 2
        )
        if tau.value != 0.0:
            dpdGLEP[affected] += -glf0d / np.exp(dt[affected] / tau)
        return dpdGLEP
