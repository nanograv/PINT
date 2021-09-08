"""Support for independent binary models.

This module if for wrapping standalone binary models so that they work
as PINT timing models.
"""
import logging

import astropy.units as u
import numpy as np
from astropy.time import Time

from pint import ls
from pint.models.parameter import MJDParameter, floatParameter, prefixParameter
from pint.models.stand_alone_psr_binaries import binary_orbits as bo
from pint.models.timing_model import DelayComponent, MissingParameter
from pint.utils import taylor_horner_deriv

log = logging.getLogger(__name__)


class PulsarBinary(DelayComponent):
    """Base class for binary models in PINT.

    This class provides a wrapper for internal classes that do the actual calculations.
    The calculations are done by the classes located in
    :mod:`pint.models.stand_alone_psr_binary`.

    Binary models generally support the below parameters, although some may support
    additional parameters and/or remove/ignore some of these.

    Model parameters:

        - T0 - time of (any) periastron (MJD)
        - PB - binary period (days, non-negative)
        - PBDOT - time derivative of binary period (s/s)
        - A1 - projected orbital amplitude, $a \sin i$ (ls, non-negative)
        - A1DOT - time derivative of projected orbital amplitude (ls/s)
        - ECC (or E) - eccentricity (no units, 0<=ECC<1)
        - EDOT - time derivative of eccentricity (1/s)
        - OM - longitude of periastron (deg)
        - OMDOT - time derivative of longitude of periastron (deg/s)
        - M2 - companion mass for Shapiro delay (solMass, non-negative)
        - SINI - system inclination (0<=SINI<=1)
        - FB0 - orbital frequency (1/s, alternative to PB, non-negative)
        - FBn - time derivatives of orbital frequency (1/s**(n+1))

    The internal calculation code uses different names for some parameters:

        - Eccentric Anomaly:               E (not parameter ECC)
        - Mean Anomaly:                    M
        - True Anomaly:                    nu
        - Eccentric:                       ecc
        - Longitude of periastron:         omega
        - Projected semi-major axis of orbit:   a1

    Parameters supported:

    .. paramtable::
        :class: pint.models.pulsar_binary.PulsarBinary
    """

    category = "pulsar_system"

    def __init__(self):
        super().__init__()
        self.binary_model_name = None
        self.barycentric_time = None
        self.binary_model_class = None
        self.add_param(
            floatParameter(
                name="PB", units=u.day, description="Orbital period", long_double=True
            )
        )
        self.add_param(
            floatParameter(
                name="PBDOT",
                units=u.day / u.day,
                description="Orbital period derivitve respect to time",
                unit_scale=True,
                scale_factor=1e-12,
                scale_threshold=1e-7,
            )
        )
        self.add_param(
            floatParameter(
                name="A1", units=ls, description="Projected semi-major axis, a*sin(i)"
            )
        )
        # NOTE: the DOT here takes the value and times 1e-12, tempo/tempo2 can
        # take both.
        self.add_param(
            floatParameter(
                name="A1DOT",
                aliases=["XDOT"],
                units=ls / u.s,
                description="Derivative of projected semi-major axis, da*sin(i)/dt",
                unit_scale=True,
                scale_factor=1e-12,
                scale_threshold=1e-7,
            )
        )
        self.add_param(
            floatParameter(
                name="ECC", units="", aliases=["E"], description="Eccentricity"
            )
        )
        self.add_param(
            floatParameter(
                name="EDOT",
                units="1/s",
                description="Eccentricity derivitve respect to time",
                unit_scale=True,
                scale_factor=1e-12,
                scale_threshold=1e-7,
            )
        )
        self.add_param(
            MJDParameter(
                name="T0", description="Epoch of periastron passage", time_scale="tdb"
            )
        )
        self.add_param(
            floatParameter(
                name="OM",
                units=u.deg,
                description="Longitude of periastron",
                long_double=True,
            )
        )
        self.add_param(
            floatParameter(
                name="OMDOT",
                units="deg/year",
                description="Longitude of periastron",
                long_double=True,
            )
        )
        self.add_param(
            floatParameter(
                name="M2",
                units=u.M_sun,
                description="Mass of companian in the unit Sun mass",
            )
        )
        self.add_param(
            floatParameter(
                name="SINI", units="", description="Sine of inclination angle"
            )
        )
        self.add_param(
            prefixParameter(
                name="FB0",
                value=None,
                units="1/s^1",
                description="0th time derivative of frequency of orbit",
                unit_template=self.FBX_unit,
                aliases=["FB"],
                description_template=self.FBX_description,
                type_match="float",
                long_double=True,
            )
        )
        self.internal_params = []
        self.warn_default_params = ["ECC", "OM"]
        # Set up delay function
        self.delay_funcs_component += [self.binarymodel_delay]

    def setup(self):
        super().setup()
        for bpar in self.params:
            self.register_deriv_funcs(self.d_binary_delay_d_xxxx, bpar)
        # Setup the model isinstance
        self.binary_instance = self.binary_model_class()
        # Setup the FBX orbits if FB is set.
        # TODO this should use a smarter way to set up orbit.
        FBX_mapping = self.get_prefix_mapping_component("FB")
        FBXs = {}
        for fbn in FBX_mapping.values():
            FBXs[fbn] = getattr(self, fbn).quantity
        if any([v is not None for v in FBXs.values()]):
            if self.FB0.value is None:
                raise ValueError("Some FBn parameters are set but FB0 is not.")
            for fb_name, fb_value in FBXs.items():
                self.binary_instance.add_binary_params(fb_name, fb_value)
            self.binary_instance.orbits_cls = bo.OrbitFBX(
                self.binary_instance, list(FBXs.keys())
            )
        # Update the parameters in the stand alone binary
        self.update_binary_object(None)

    def validate(self):
        super().validate()
        if (
            hasattr(self, "SINI")
            and self.SINI.value is not None
            and not 0 <= self.SINI.value <= 1
        ):
            raise ValueError(
                f"Sine of inclination angle must be between zero and one ({self.SINI.quantity})"
            )
        if hasattr(self, "M2") and self.M2.value is not None and self.M2.value < 0:
            raise ValueError(
                f"Companion mass M2 cannot be negative ({self.M2.quantity})"
            )
        if (
            hasattr(self, "ECC")
            and self.ECC.value is not None
            and not 0 <= self.ECC.value <= 1
        ):
            raise ValueError(
                f"Eccentricity ECC must be between zero and one ({self.ECC.quantity})"
            )
        if self.A1.value is not None and self.A1.value < 0:
            raise ValueError(
                f"Projected semi-major axis A1 cannot be negative ({self.A1.quantity})"
            )
        if self.PB.value is not None:
            if self.PB.value <= 0:
                raise ValueError(
                    f"Binary period PB must be non-negative ({self.PB.quantity})"
                )
            if self.FB0.value is not None:
                raise ValueError("Model cannot have values for both FB0 and PB")
        if self.FB0.value is not None and self.FB0.value <= 0:
            raise ValueError(
                f"Binary frequency FB0 must be non-negative ({self.FB0.quantity})"
            )

    def check_required_params(self, required_params):
        # seach for all the possible to get the parameters.
        for p in required_params:
            par = getattr(self, p)
            if par.value is None:
                # try to search if there is any class method that computes it
                method_name = p.lower() + "_func"
                try:
                    par_method = getattr(self.binary_instance, method_name)
                except AttributeError:
                    raise MissingParameter(
                        self.binary_model_name,
                        p + " is required for '%s'." % self.binary_model_name,
                    )
                par_method()

    # With new parameter class set up, do we need this?
    def apply_units(self):
        """Apply units to parameter value."""
        for bpar in self.binary_params:
            bparObj = getattr(self, bpar)
            if bparObj.value is None or bparObj.units is None:
                continue
            bparObj.value = bparObj.value * u.Unit(bparObj.units)

    def update_binary_object(self, toas, acc_delay=None):
        """Update stand alone binary's parameters and toas from PINT-facing object.

        This function passes the PINT-facing object's parameter values and TOAs
        to the stand-alone binary object. If the TOAs are not provided, it only
        updates the parameters not the TOAs.

        Parameters
        ----------
        toas: pint.toa.TOAs
            The TOAs that need to pass to the stand alone model.Default value is
            None. If toas is None, this function only updates the parameter value.
            If 'acc_delay' is not provided, the stand alone binary receives the
            standard barycentered TOAs.

        acc_delay: numpy.ndarray
            If provided, TOAs will be corrected by provided acc_delay instead of
            the standard barycentering. The stand alone binary receives the
            input TOAs - acc_delay.

        Warns
        -----
        If passing 'None' to 'toa' argument, the stand alone binary model will use
        the TOAs were passed to it from last interation (i.e. last barycnetered
        TOAs) or no TOAs for stand alone binary model at all. This behavior will
        cause incorrect answers. Allowing the passing None to 'toa' argument is
        for some lower level functions and tests. We do not recommend PINT
        user to use it.
        """
        # Don't need to fill P0 and P1. Translate all the others to the format
        # that is used in bmodel.py
        # Get barycnetric toa first
        updates = {}
        if toas is not None:
            tbl = toas.table
            if acc_delay is None:
                # If the accumulated delay is not provided, calculate and
                # use the barycentered TOAS
                self.barycentric_time = self._parent.get_barycentric_toas(toas)
            else:
                self.barycentric_time = tbl["tdbld"] * u.day - acc_delay
            updates["barycentric_toa"] = self.barycentric_time
            updates["obs_pos"] = tbl["ssb_obs_pos"].quantity
            updates["psr_pos"] = self._parent.ssb_to_psb_xyz_ICRS(
                epoch=tbl["tdbld"].astype(np.float64)
            )
        for par in self.binary_instance.binary_params:
            if par in self.binary_instance.param_aliases.keys():
                alias = self.binary_instance.param_aliases[par]
            else:
                alias = []

            # the _parent attribute should give access to all the components
            if hasattr(self._parent, par) or set(alias).intersection(self.params):
                try:
                    pint_bin_name = self._parent.match_param_aliases(par)
                except ValueError:
                    if par in self.internal_params:
                        pint_bin_name = par
                binObjpar = getattr(self._parent, pint_bin_name)
                instance_par = getattr(self.binary_instance, par)
                if hasattr(instance_par, "value"):
                    instance_par_val = instance_par.value
                else:
                    instance_par_val = instance_par
                if binObjpar.value is None:
                    if binObjpar.name in self.warn_default_params:
                        log.warning(
                            "'%s' is not set, using the default value %f "
                            "instead." % (binObjpar.name, instance_par_val)
                        )
                    continue
                if binObjpar.units is not None:
                    updates[par] = binObjpar.value * binObjpar.units
                else:
                    updates[par] = binObjpar.value
        self.binary_instance.update_input(**updates)

    def binarymodel_delay(self, toas, acc_delay=None):
        """Return the binary model independent delay call."""
        self.update_binary_object(toas, acc_delay)
        return self.binary_instance.binary_delay()

    def d_binary_delay_d_xxxx(self, toas, param, acc_delay):
        """Return the binary model delay derivtives."""
        self.update_binary_object(toas, acc_delay)
        return self.binary_instance.d_binarydelay_d_par(param)

    def print_par(self):
        result = "BINARY {0}\n".format(self.binary_model_name)
        for p in self.params:
            par = getattr(self, p)
            if par.quantity is not None:
                result += par.as_parfile_line()
        return result

    def FBX_unit(self, n):
        return "1/s^%d" % (n + 1) if n else "1/s"

    def FBX_description(self, n):
        return "%dth time derivative of frequency of orbit" % n

    def change_binary_epoch(self, new_epoch):
        """Change the epoch for this binary model.

        T0 will be changed to the periapsis time closest to the supplied epoch,
        and the argument of periapsis (OM), eccentricity (ECC), and projected
        semimajor axis (A1 or X) will be updated according to the specified
        OMDOT, EDOT, and A1DOT or XDOT, if present.

        Note that derivatives of binary orbital frequency higher than the first
        (FB2, FB3, etc.) are ignored in computing the new T0, even if present in
        the model. If high-precision results are necessary, especially for models
        containing higher derivatives of orbital frequency, consider re-fitting
        the model to a set of TOAs. The use of :func:`pint.simulation.make_fake_toas`
        and the :class:`pint.fitter.Fitter` option ``track_mode="use_pulse_number"``
        can make this extremely simple.

        Parameters
        ----------
        new_epoch: float MJD (in TDB) or `astropy.Time` object
            The new epoch value.
        """
        if isinstance(new_epoch, Time):
            new_epoch = Time(new_epoch, scale="tdb", precision=9)
        else:
            new_epoch = Time(new_epoch, scale="tdb", format="mjd", precision=9)

        # Get PB and PBDOT from model
        if self.PB.quantity is not None:
            PB = self.PB.quantity
            if self.PBDOT.quantity is not None:
                PBDOT = self.PBDOT.quantity
            else:
                PBDOT = 0.0 * u.Unit("")
        else:
            PB = 1.0 / self.FB0.quantity
            try:
                PBDOT = -self.FB1.quantity / self.FB0.quantity ** 2
            except AttributeError:
                PBDOT = 0.0 * u.Unit("")

        # Find closest periapsis time and reassign T0
        t0_ld = self.T0.quantity.tdb.mjd_long
        dt = (new_epoch.tdb.mjd_long - t0_ld) * u.day
        d_orbits = dt / PB - PBDOT * dt ** 2 / (2.0 * PB ** 2)
        n_orbits = np.round(d_orbits.to(u.Unit("")))
        if n_orbits == 0:
            return
        dt_integer_orbits = PB * n_orbits + PB * PBDOT * n_orbits ** 2 / 2.0
        self.T0.quantity = self.T0.quantity + dt_integer_orbits

        try:
            if self.FB2.quantity is not None:
                log.warning(
                    "Ignoring orbital frequency derivatives higher than FB1"
                    "in computing new T0; a model fit should resolve this"
                )
        except AttributeError:
            pass

        # Update PB or FB0, FB1, etc.
        if isinstance(self.binary_instance.orbits_cls, bo.OrbitPB):
            dPB = PBDOT * dt_integer_orbits
            self.PB.quantity = self.PB.quantity + dPB
        else:
            fbterms = [0.0 * u.Unit("")] + self._parent.get_prefix_list("FB")

            for n in range(len(fbterms) - 1):
                cur_deriv = getattr(self, "FB{}".format(n))
                cur_deriv.value = taylor_horner_deriv(
                    dt_integer_orbits.to(u.s), fbterms, deriv_order=n + 1
                )

        # Update ECC, OM, and A1
        dECC = self.EDOT.quantity * dt_integer_orbits
        self.ECC.quantity = self.ECC.quantity + dECC
        dOM = self.OMDOT.quantity * dt_integer_orbits
        self.OM.quantity = self.OM.quantity + dOM
        dA1 = self.A1DOT.quantity * dt_integer_orbits
        self.A1.quantity = self.A1.quantity + dA1
