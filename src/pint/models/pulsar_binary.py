"""Support for independent binary models.

This module if for wrapping standalone binary models so that they work
as PINT timing models.
"""

import contextlib

import astropy.constants as consts
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from loguru import logger as log

from pint.exceptions import MissingParameter, TimingModelError, UnknownParameter
from pint.models.parameter import (
    MJDParameter,
    floatParameter,
    funcParameter,
    prefixParameter,
)
from pint.models.stand_alone_psr_binaries import binary_orbits as bo
from pint.models.timing_model import DelayComponent
from pint.pulsar_ecliptic import PulsarEcliptic
from pint.utils import parse_time, taylor_horner_deriv

# def _p_to_f(p):
#     return 1 / p


# def _pdot_to_fdot(p, pdot):
#     return -pdot / p**2


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

    The following ORBWAVEs parameters define a Fourier series model for orbital phase
    variations, as an alternative to the FBn Taylor series expansion:

        - ORBWAVE_OM - base angular frequency for ORBWAVEs expansion (rad / s)
        - ORBWAVE_EPOCH - reference epoch for ORBWAVEs model (MJD)
        - ORBWAVECn/ORBWAVESn - coefficients for cosine/sine components (dimensionless)

    The orbital phase is then given by:
        orbits(t) = (t - T0) / PB
                    + \sum_{n=0} (ORBWAVECn cos(ORBWAVE_OM * (n + 1) * (t - ORBWAVE_EPOCH))
                                + ORBWAVESn sin(ORBWAVE_OM * (n + 1) * (t - ORBWAVE_EPOCH))

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

    # Suffix appended to the PINT-facing parameter names of this component
    # (e.g. ``"_2"`` for the outer orbit of a hierarchical triple). The
    # standalone binary instance always uses the canonical (unsuffixed) names.
    # An empty string means the normal single-binary behaviour.
    param_suffix = ""

    # The top-level parameter that selects this binary model in the parfile.
    # Outer-orbit components override this with ``"BINARY2"``.
    binary_param_tag = "BINARY"

    def __init__(self):
        super().__init__()
        self.binary_model_name = None
        self.barycentric_time = None
        self.binary_model_class = None
        self.add_param(
            floatParameter(
                name="PB",
                units=u.day,
                description="Orbital period",
                long_double=True,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="PBDOT",
                units=u.day / u.day,
                description="Orbital period derivative respect to time",
                unit_scale=True,
                scale_factor=1e-12,
                scale_threshold=1e-7,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="A1",
                units=u.lsec,
                description="Projected semi-major axis of pulsar orbit, ap*sin(i)",
                tcb2tdb_scale_factor=(1 / consts.c),
            )
        )
        # NOTE: the DOT here takes the value and times 1e-12, tempo/tempo2 can
        # take both.
        self.add_param(
            floatParameter(
                name="A1DOT",
                aliases=["XDOT"],
                units=u.lsec / u.s,
                description="Derivative of projected semi-major axis, d[ap*sin(i)]/dt",
                unit_scale=True,
                scale_factor=1e-12,
                scale_threshold=1e-7,
                tcb2tdb_scale_factor=(1 / consts.c),
            )
        )
        self.add_param(
            floatParameter(
                name="ECC",
                units="",
                aliases=["E"],
                description="Eccentricity",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="EDOT",
                units="1/s",
                description="Eccentricity derivative respect to time",
                unit_scale=True,
                scale_factor=1e-12,
                scale_threshold=1e-7,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            MJDParameter(
                name="T0",
                description="Epoch of periastron passage",
                time_scale="tdb",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="OM",
                units=u.deg,
                description="Longitude of periastron",
                long_double=True,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="OMDOT",
                units="deg/year",
                description="Rate of advance of periastron",
                long_double=True,
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            floatParameter(
                name="M2",
                units=u.M_sun,
                description="Companion mass",
                tcb2tdb_scale_factor=(consts.G / consts.c**3),
            )
        )
        self.add_param(
            floatParameter(
                name="SINI",
                units="",
                description="Sine of inclination angle",
                tcb2tdb_scale_factor=u.Quantity(1),
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
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )

        self.add_param(
            prefixParameter(
                name="ORBWAVEC0",
                value=None,
                units="",
                description="Amplitude of cosine wave in Tasc-shift function",
                aliases=["ORBWAVEC"],
                description_template=self.ORBWAVEC_description,
                type_match="float",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )

        self.add_param(
            prefixParameter(
                name="ORBWAVES0",
                value=None,
                units="",
                description="Amplitude of sine wave in Tasc-shift function",
                aliases=["ORBWAVES"],
                description_template=self.ORBWAVES_description,
                type_match="float",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )

        self.add_param(
            floatParameter(
                name="ORBWAVE_OM",
                units="rad/s",
                description="Base frequency for ORBWAVEs model",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            MJDParameter(
                name="ORBWAVE_EPOCH",
                description="Reference epoch for ORBWAVEs model",
                time_scale="tdb",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )

        self.internal_params = []
        self.warn_default_params = ["ECC", "OM"]
        # Set up delay function
        self.delay_funcs_component += [self.binarymodel_delay]
        self.delay_deriv_wrt_prev_delay_funcs += [self.d_binary_delay_d_prev_delay]

    def _apply_param_suffix(self):
        """Rename this component's PINT-facing parameters with ``param_suffix``.

        This is used by outer-orbit components (e.g. the outer binary of a
        hierarchical triple) so that their parameters appear in the parfile as
        ``PB_2``, ``A1_2``, ... while the underlying standalone binary instance
        keeps the canonical names (``PB``, ``A1``, ...).

        The canonical attribute (e.g. ``self.PB``) is *removed* from this
        component so that it does not leak into the parent
        :class:`~pint.models.timing_model.TimingModel` namespace (where it would
        collide with the inner binary's identically named parameter). Base-class
        methods therefore access parameters through :meth:`_bp` / :meth:`_hasbp`,
        which apply the suffix.

        Prefix parameters (``FBn``, ``ORBWAVECn``, ...) are removed entirely:
        orbital-frequency and ORBWAVE parameterizations are not supported for a
        suffixed (outer) orbit.
        """
        suffix = self.param_suffix
        if not suffix:
            return
        for canonical in list(self.params):
            par = getattr(self, canonical)
            if getattr(par, "is_prefix", False):
                # Orbital-frequency / ORBWAVE parameterizations are not
                # supported for a suffixed (outer) orbit.
                self.remove_param(canonical)
                continue
            new_name = canonical + suffix
            par.aliases = [alias + suffix for alias in par.aliases]
            par.name = new_name
            # Keep any funcParameter cross-references consistent with the
            # renamed parameters (no-op for DD/BT, which have none).
            if hasattr(par, "_params"):
                par._params = [p + suffix for p in par._params]
            # Expose the renamed parameter under its suffixed name only and drop
            # the canonical attribute so it cannot leak to the parent model.
            setattr(self, new_name, par)
            delattr(self, canonical)
            self.params[self.params.index(canonical)] = new_name

    def _bp(self, name):
        """Return this component's parameter object for canonical ``name``.

        Applies :attr:`param_suffix` so that base-class code written in terms of
        canonical names (e.g. ``PB``) resolves to the suffixed parameter (e.g.
        ``PB_2``) for an outer-orbit component. For a normal binary
        (``param_suffix == ""``) this is just ``getattr(self, name)``.
        """
        suffixed = name + self.param_suffix
        if self.param_suffix and hasattr(self, suffixed):
            return getattr(self, suffixed)
        return getattr(self, name)

    def _hasbp(self, name):
        """Whether this component has the (possibly suffixed) parameter ``name``."""
        suffixed = name + self.param_suffix
        return bool(self.param_suffix and hasattr(self, suffixed)) or hasattr(
            self, name
        )

    def setup(self):
        super().setup()
        for bpar in self.params:
            self.register_deriv_funcs(self.d_binary_delay_d_xxxx, bpar)
        # Setup the model isinstance
        self.binary_instance = self.binary_model_class()
        # Setup the FBX orbits if FB is set.
        # TODO this should use a smarter way to set up orbit.
        FBX_mapping = self.get_prefix_mapping_component("FB")
        FBXs = {fbn: getattr(self, fbn).quantity for fbn in FBX_mapping.values()}
        if any(v is not None for v in FBXs.values()):
            if self.FB0.value is None:
                raise ValueError("Some FBn parameters are set but FB0 is not.")
            for fb_name, fb_value in FBXs.items():
                self.binary_instance.add_binary_params(fb_name, fb_value)
            self.binary_instance.orbits_cls = bo.OrbitFBX(
                self.binary_instance, list(FBXs.keys())
            )
            # Note: if we are happy to use these to show alternate parameterizations then this can be uncommented

            # # remove the PB parameterization, replace with functions
            # self.remove_param("PB")
            # self.remove_param("PBDOT")
            # self.add_param(
            #     funcParameter(
            #         name="PB",
            #         units=u.day,
            #         description="Orbital period",
            #         long_double=True,
            #         params=("FB0",),
            #         func=_p_to_f,
            #     )
            # )
            # self.add_param(
            #     funcParameter(
            #         name="PBDOT",
            #         units=u.day / u.day,
            #         description="Orbital period derivative respect to time",
            #         unit_scale=True,
            #         scale_factor=1e-12,
            #         scale_threshold=1e-7,
            #         params=("FB0", "FB1"),
            #         func=_pdot_to_fdot,
            #     )
            # )

        ORBWAVES_mapping = self.get_prefix_mapping_component("ORBWAVES")
        ORBWAVES = {
            ows: getattr(self, ows).quantity for ows in ORBWAVES_mapping.values()
        }
        ORBWAVEC_mapping = self.get_prefix_mapping_component("ORBWAVEC")
        ORBWAVEC = {
            owc: getattr(self, owc).quantity for owc in ORBWAVEC_mapping.values()
        }

        if any(v is not None for v in ORBWAVES.values()):
            for k in ["ORBWAVE_OM", "ORBWAVE_EPOCH"]:
                self.binary_instance.add_binary_params(k, getattr(self, k).value)

            for k in ORBWAVES.keys():
                self.binary_instance.add_binary_params(k, ORBWAVES[k])

            for k in ORBWAVEC.keys():
                self.binary_instance.add_binary_params(k, ORBWAVEC[k])

            using_FBX = any(v is not None for v in FBXs.values())
            if using_FBX:
                fbx = sorted(list(FBXs.keys()))
                if len(fbx) > 2:
                    raise ValueError("Only FB0/FB1 are supported.")
                if (len(fbx) == 2) and (fbx[1] != "FB1"):
                    raise ValueError("Only FB0/FB1 are supported.")
                self.binary_instance.orbits_cls = bo.OrbitWavesFBX(
                    self.binary_instance,
                    fbx
                    + ["TASC", "ORBWAVE_OM", "ORBWAVE_EPOCH"]
                    + list(ORBWAVES.keys())
                    + list(ORBWAVEC.keys()),
                )
            else:
                self.binary_instance.orbits_cls = bo.OrbitWaves(
                    self.binary_instance,
                    ["PB", "TASC", "ORBWAVE_OM", "ORBWAVE_EPOCH"]
                    + list(ORBWAVES.keys())
                    + list(ORBWAVEC.keys()),
                )

        # Note: if we are happy to use these to show alternate parameterizations then this can be uncommented
        # else:
        #     # remove the FB parameterization, replace with functions
        #     self.remove_param("FB0")
        #     self.add_param(
        #         funcParameter(
        #             name="FB0",
        #             units="1/s^1",
        #             description="0th time derivative of frequency of orbit",
        #             aliases=["FB"],
        #             long_double=True,
        #             params=("PB",),
        #             func=_p_to_f,
        #         )
        #     )
        #     self.add_param(
        #         funcParameter(
        #             name="FB1",
        #             units="1/s^2",
        #             description="1st time derivative of frequency of orbit",
        #             long_double=True,
        #             params=("PB", "PBDOT"),
        #             func=_pdot_to_fdot,
        #         )
        #     )

        # Update the parameters in the stand alone binary
        self.update_binary_object(None)

    def validate(self):
        super().validate()
        if (
            self._hasbp("SINI")
            and self._bp("SINI").value is not None
            and not 0 <= self._bp("SINI").value <= 1
        ):
            raise ValueError(
                f"Sine of inclination angle must be between zero and one ({self._bp('SINI').quantity})"
            )
        if (
            self._hasbp("M2")
            and self._bp("M2").value is not None
            and self._bp("M2").value < 0
        ):
            raise ValueError(
                f"Companion mass M2 cannot be negative ({self._bp('M2').quantity})"
            )
        if (
            self._hasbp("ECC")
            and self._bp("ECC").value is not None
            and not 0 <= self._bp("ECC").value <= 1
        ):
            raise ValueError(
                f"Eccentricity ECC must be between zero and one ({self._bp('ECC').quantity})"
            )
        if (
            self._hasbp("A1")
            and self._bp("A1").value is not None
            and self._bp("A1").value < 0
        ):
            raise ValueError(
                f"Projected semi-major axis A1 cannot be negative ({self._bp('A1').quantity})"
            )
        has_fb0 = self._hasbp("FB0")
        if self._hasbp("PB") and self._bp("PB").value is not None:
            if self._bp("PB").value <= 0:
                raise ValueError(
                    f"Binary period PB must be non-negative ({self._bp('PB').quantity})"
                )
            if (
                has_fb0
                and self._bp("FB0").value is not None
                and not (
                    isinstance(self._bp("FB0"), funcParameter)
                    or isinstance(self._bp("PB"), funcParameter)
                )
            ):
                raise ValueError("Model cannot have values for both FB0 and PB")
        if has_fb0 and self._bp("FB0").value is not None and self._bp("FB0").value <= 0:
            raise ValueError(
                f"Binary frequency FB0 must be non-negative ({self._bp('FB0').quantity})"
            )

    def check_required_params(self, required_params):
        # search for all the possible to get the parameters.
        for p in required_params:
            par = self._bp(p)
            if par.value is None:
                # try to search if there is any class method that computes it
                method_name = f"{p.lower()}_func"
                try:
                    par_method = getattr(self.binary_instance, method_name)
                except AttributeError as e:
                    raise MissingParameter(
                        self.binary_model_name,
                        f"{p} is required for '{self.binary_model_name}'.",
                    ) from e
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

        Notes
        -----
        The values for ``obs_pos`` (the observatory position wrt the Solar System Barycenter) and ``psr_pos``
        (the pulsar position wrt the Solar System Barycenter) are both computed in the same reference frame, ICRS or ECL depending on the model.

        Warns
        -----
        If passing 'None' to 'toa' argument, the stand alone binary model will use
        the TOAs were passed to it from last iteration (i.e. last barycentered
        TOAs) or no TOAs for stand alone binary model at all. This behavior will
        cause incorrect answers. Allowing the passing None to 'toa' argument is
        for some lower level functions and tests. We do not recommend PINT
        user to use it.
        """
        # Don't need to fill P0 and P1. Translate all the others to the format
        # that is used in bmodel.py
        # Get barycentric toa first
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
            if "AstrometryEquatorial" in self._parent.components:
                # it's already in ICRS
                updates["obs_pos"] = tbl["ssb_obs_pos"].quantity
                updates["psr_pos"] = self._parent.ssb_to_psb_xyz_ICRS(
                    epoch=tbl["tdbld"].astype(np.float64)
                )
            elif "AstrometryEcliptic" in self._parent.components:
                # convert from ICRS to ECL
                obs_pos = SkyCoord(
                    tbl["ssb_obs_pos"].quantity,
                    representation_type="cartesian",
                    frame="icrs",
                )
                updates["obs_pos"] = obs_pos.transform_to(
                    PulsarEcliptic(ecl=self._parent.ECL.value)
                ).cartesian.xyz.transpose()
                updates["psr_pos"] = self._parent.ssb_to_psb_xyz_ECL(
                    epoch=tbl["tdbld"].astype(np.float64), ecl=self._parent.ECL.value
                )
        for par in self.binary_instance.binary_params:
            # The standalone binary instance uses canonical names (``par``),
            # while this component's PINT-facing parameter may carry a suffix
            # (e.g. ``PB_2`` for an outer orbit). ``param_suffix`` is empty for
            # a normal single binary, in which case this is a no-op.
            pint_par_name = par + self.param_suffix
            if par in self.binary_instance.param_aliases.keys():
                alias = [
                    a + self.param_suffix
                    for a in self.binary_instance.param_aliases[par]
                ]
            else:
                alias = []

            # the _parent attribute should give access to all the components
            if hasattr(self._parent, pint_par_name) or set(alias).intersection(
                self.params
            ):
                try:
                    pint_bin_name = self._parent.match_param_aliases(pint_par_name)
                except UnknownParameter as e:
                    if par in self.internal_params:
                        pint_bin_name = pint_par_name
                    else:
                        raise UnknownParameter(
                            f"Unable to find {pint_par_name} in the parent model"
                        ) from e
                binObjpar = getattr(self._parent, pint_bin_name)

                # make sure we aren't passing along derived parameters to the binary instance
                if isinstance(binObjpar, funcParameter):
                    continue
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
        """Return the binary model delay derivatives."""
        self.update_binary_object(toas, acc_delay)
        # The standalone binary instance only knows the canonical (unsuffixed)
        # parameter names, so strip the suffix before requesting a derivative.
        if self.param_suffix and param.endswith(self.param_suffix):
            param = param[: -len(self.param_suffix)]
        return self.binary_instance.d_binarydelay_d_par(param)

    def d_binary_delay_d_prev_delay(self, toas, acc_delay):
        """Return derivative of binary delay w.r.t. previous delays"""
        self.update_binary_object(toas, acc_delay)
        return self.binary_instance.d_binarydelay_d_prevdelay

    def print_par(self, format="pint"):
        tag = self.binary_param_tag
        tag_par = getattr(self._parent, tag) if self._parent is not None else None
        if self._parent is None:
            result = f"{tag} {self.binary_model_name}\n"
        elif tag_par.value != self.binary_model_name:
            raise TimingModelError(
                f"Parameter {tag} {tag_par.value}"
                f" does not match the binary"
                f" component {self.binary_model_name}"
            )
        else:
            result = tag_par.as_parfile_line(format=format)

        for p in self.params:
            par = getattr(self, p)
            if par.quantity is not None:
                result += par.as_parfile_line(format=format)

        return result

    def FBX_unit(self, n):
        return "1/s^%d" % (n + 1) if n else "1/s"

    def FBX_description(self, n):
        return "%dth time derivative of frequency of orbit" % n

    def ORBWAVES_description(self, n):
        return (
            "Coefficient of the %dth sine wave in Fourier series model of Tasc variations"
            % n
        )

    def ORBWAVEC_description(self, n):
        return (
            "Coefficient of the %dth cosine wave in Fourier series model of Tasc variations"
            % n
        )

    def change_binary_epoch(self, new_epoch):
        """Change the epoch for this binary model.

        T0 will be changed to the periapsis time closest to the supplied epoch,
        and the argument of periapsis (OM), eccentricity (ECC), and projected
        semi-major axis (A1 or X) will be updated according to the specified
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
        new_epoch = parse_time(new_epoch, scale="tdb", precision=9)

        # Parameter access goes through _bp() so that this works for both a
        # normal (inner) binary and a suffixed outer-orbit component.
        PB_par = self._bp("PB")
        PBDOT_par = self._bp("PBDOT")
        T0_par = self._bp("T0")

        # Get PB and PBDOT from model
        if PB_par.quantity is not None and not isinstance(PB_par, funcParameter):
            PB = PB_par.quantity
            if PBDOT_par.quantity is not None:
                PBDOT = PBDOT_par.quantity
            else:
                PBDOT = 0.0 * u.Unit("")
        else:
            PB = 1.0 / self._bp("FB0").quantity
            try:
                PBDOT = -self._bp("FB1").quantity / self._bp("FB0").quantity ** 2
            except AttributeError:
                PBDOT = 0.0 * u.Unit("")

        # Find closest periapsis time and reassign T0
        t0_ld = T0_par.quantity.tdb.mjd_long
        dt = (new_epoch.tdb.mjd_long - t0_ld) * u.day
        d_orbits = dt / PB - PBDOT * dt**2 / (2.0 * PB**2)
        n_orbits = np.round(d_orbits.to(u.Unit("")))
        if n_orbits == 0:
            return
        dt_integer_orbits = PB * n_orbits + PB * PBDOT * n_orbits**2 / 2.0
        T0_par.quantity = T0_par.quantity + dt_integer_orbits

        with contextlib.suppress(AttributeError):
            if self._bp("FB2").quantity is not None:
                log.warning(
                    "Ignoring orbital frequency derivatives higher than FB1"
                    "in computing new T0; a model fit should resolve this"
                )
        # Update PB or FB0, FB1, etc.
        if isinstance(self.binary_instance.orbits_cls, bo.OrbitPB):
            dPB = PBDOT * dt_integer_orbits
            PB_par.quantity = PB_par.quantity + dPB
        else:
            fbterms = [0.0 * u.Unit("")] + self._parent.get_prefix_list("FB")

            for n in range(len(fbterms) - 1):
                cur_deriv = self._bp(f"FB{n}")
                cur_deriv.value = taylor_horner_deriv(
                    dt_integer_orbits.to(u.s), fbterms, deriv_order=n + 1
                )

        # Update ECC, OM, and A1
        dECC = self._bp("EDOT").quantity * dt_integer_orbits
        self._bp("ECC").quantity = self._bp("ECC").quantity + dECC
        dOM = self._bp("OMDOT").quantity * dt_integer_orbits
        self._bp("OM").quantity = self._bp("OM").quantity + dOM
        dA1 = self._bp("A1DOT").quantity * dt_integer_orbits
        self._bp("A1").quantity = self._bp("A1").quantity + dA1

    def pb(self, t=None):
        """Return binary period and uncertainty (optionally evaluated at different times) regardless of binary model

        Parameters
        ----------
        t : astropy.time.Time, astropy.units.Quantity, numpy.ndarray, float, int, str, optional
            Time(s) to evaluate period

        Returns
        -------
        astropy.units.Quantity :
            Binary period
        astropy.units.Quantity :
            Binary period uncertainty

        """
        PB_par = self._bp("PB")
        PBDOT_par = self._bp("PBDOT")
        if self.binary_model_name.startswith("ELL1"):
            t0 = self._bp("TASC").quantity
        else:
            t0 = self._bp("T0").quantity
        t = t0 if t is None else parse_time(t)
        if PB_par.quantity is not None:
            if PBDOT_par.quantity is None and (
                not self._hasbp("XPBDOT") or self._bp("XPBDOT").quantity is not None
            ):
                return PB_par.quantity, PB_par.uncertainty
            pb = PB_par.as_ufloat(u.d)
            if PBDOT_par.quantity is not None:
                pbdot = PBDOT_par.as_ufloat(u.s / u.s)
            if self._hasbp("XPBDOT") and self._bp("XPBDOT").quantity is not None:
                pbdot += self._bp("XPBDOT").as_ufloat(u.s / u.s)
            pnew = pb + pbdot * (t - t0).jd
            if not isinstance(pnew, np.ndarray):
                return pnew.n * u.d, pnew.s * u.d if pnew.s > 0 else None
            import uncertainties.unumpy

            return (
                uncertainties.unumpy.nominal_values(pnew) * u.d,
                uncertainties.unumpy.std_devs(pnew) * u.d,
            )

        elif self._hasbp("FB0") and self._bp("FB0").quantity is not None:
            # assume FB terms
            dt = (t - t0).sec
            coeffs = []
            unit = u.Hz
            for p in self.get_prefix_mapping_component("FB").values():
                coeffs.append(getattr(self, p).as_ufloat(unit))
                unit /= u.s
            pnew = 1 / taylor_horner_deriv(dt, coeffs, deriv_order=0)
            if not isinstance(pnew, np.ndarray):
                return pnew.n * u.s, pnew.s * u.s if pnew.s > 0 else None
            import uncertainties.unumpy

            return (
                uncertainties.unumpy.nominal_values(pnew) * u.s,
                uncertainties.unumpy.std_devs(pnew) * u.s,
            )
        raise AttributeError("Neither PB nor FB0 is present in the timing model.")
