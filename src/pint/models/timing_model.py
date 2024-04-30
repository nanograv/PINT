"""Timing model objects.

Defines the basic timing model interface classes.

A PINT timing model will be an instance of
:class:`~pint.models.timing_model.TimingModel`. It will have a number of
"components", each an instance of a subclass of
:class:`~pint.models.timing_model.Component`. These components each
implement some part of the timing model, whether astrometry (for
example :class:`~pint.models.astrometry.AstrometryEcliptic`), noise
modelling (for example :class:`~pint.models.noise_model.ScaleToaError`),
interstellar dispersion (for example
:class:`~pint.models.dispersion_model.DispersionDM`), or pulsar binary orbits.
This last category is somewhat unusual in that the code for each model is
divided into a PINT-facing side (for example
:class:`~pint.models.binary_bt.BinaryBT`) and an internal model that does the
actual computation (for example
:class:`~pint.models.stand_alone_psr_binaries.BT_model.BTmodel`); the management of
data passing between these two parts is carried out by
:class:`~pint.models.pulsar_binary.PulsarBinary` and
:class:`~pint.models.stand_alone_psr_binaries.binary_generic.PSR_BINARY`.

To actually create a timing model, you almost certainly want to use
:func:`~pint.models.model_builder.get_model`.

See :ref:`Timing Models` for more details on how PINT's timing models work.

"""

import abc
import copy
import inspect
import contextlib
from collections import OrderedDict, defaultdict
from functools import wraps
from warnings import warn
from uncertainties import ufloat

import astropy.time as time
from astropy import units as u, constants as c
import numpy as np
from astropy.utils.decorators import lazyproperty
import astropy.coordinates as coords
from pint.pulsar_ecliptic import OBL, PulsarEcliptic
from scipy.optimize import brentq
from loguru import logger as log

import pint
from pint.models.parameter import (
    _parfile_formats,
    AngleParameter,
    MJDParameter,
    Parameter,
    boolParameter,
    floatParameter,
    funcParameter,
    intParameter,
    maskParameter,
    strParameter,
    prefixParameter,
)
from pint.phase import Phase
from pint.toa import TOAs
from pint.utils import (
    PrefixError,
    split_prefixed_name,
    open_or_use,
    colorize,
    xxxselections,
)
from pint.derived_quantities import dispersion_slope


__all__ = [
    "DEFAULT_ORDER",
    "TimingModel",
    "Component",
    "AllComponents",
    "TimingModelError",
    "MissingParameter",
    "MissingTOAs",
    "MissingBinaryError",
    "UnknownBinaryModel",
]
# Parameters or lines in par files we don't understand but shouldn't
# complain about. These are still passed to components so that they
# can use them if they want to.
#
# Other unrecognized parameters produce warnings and possibly indicate
# errors in the par file.
#
# Comparisons with keywords in par file lines is done in a case insensitive way.
ignore_params = {
    #    "TRES",
    "TZRMJD",
    "TZRFRQ",
    "TZRSITE",
    "NITS",
    "IBOOT",
    #    "CHI2R",
    "MODE",
    "PLANET_SHAPIRO2",
}

ignore_prefix = {"DMXF1_", "DMXF2_", "DMXEP_"}

DEFAULT_ORDER = [
    "astrometry",
    "jump_delay",
    "troposphere",
    "solar_system_shapiro",
    "solar_wind",
    "dispersion_constant",
    "dispersion_dmx",
    "dispersion_jump",
    "pulsar_system",
    "frequency_dependent",
    "absolute_phase",
    "spindown",
    "phase_jump",
    "wave",
    "wavex",
]


class MissingTOAs(ValueError):
    """Some parameter does not describe any TOAs."""

    def __init__(self, parameter_names):
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        if len(parameter_names) == 1:
            msg = f"Parameter {parameter_names[0]} does not correspond to any TOAs: you might need to run `model.find_empty_masks(toas, freeze=True)`"
        elif len(parameter_names) > 1:
            msg = f"Parameters {' '.join(parameter_names)} do not correspond to any TOAs: you might need to run `model.find_empty_masks(toas, freeze=True)`"
        else:
            raise ValueError("Incorrect attempt to construct MissingTOAs")
        super().__init__(msg)
        self.parameter_names = parameter_names


class PropertyAttributeError(ValueError):
    pass


def property_exists(f):
    """Mark a function as a property but handle AttributeErrors.

    Normal @property has the unfortunate feature that if the called function
    should accidentally emit an AttributeError, if __getattr__ is in use, this
    will be reported as if the attribute does not exist. With this decorator
    instead, the AttributeError will be caught and re-raised as a specific kind
    of ValueError, so it will be treated like an error and the backtrace printed.
    """

    @property
    @wraps(f)
    def wrapper(self):
        try:
            return f(self)
        except AttributeError as e:
            raise PropertyAttributeError(
                f"Property {f} raised AttributeError internally"
            ) from e

    return wrapper


class TimingModel:
    """Timing model object built from Components.

    This object is the primary object to represent a timing model in PINT.  It
    is normally constructed with :func:`~pint.models.model_builder.get_model`,
    and it contains a variety of :class:`~pint.models.timing_model.Component`
    objects, each representing a
    physical process that either introduces delays in the pulse arrival time or
    introduces shifts in the pulse arrival phase.  These components have
    parameters, described by :class:`~pint.models.parameter.Parameter` objects,
    and methods. Both the parameters and the methods are accessible through
    this object using attribute access, for example as ``model.F0`` or
    ``model.coords_as_GAL()``.

    Components in a TimingModel objects are accessible through the
    ``model.components`` property, using their class name to index the
    TimingModel, as ``model.components["Spindown"]``. They can be added and
    removed with methods on this object, and for many of them additional
    parameters in families (``DMXEP_1234``) can be added.

    Parameters in a TimingModel object are listed in the ``model.params`` object.
    Each Parameter can be set as free or frozen using its ``.frozen`` attribute,
    and a list of the free parameters is available through the ``model.free_params``
    property; this can also be used to set which parameters are free. Several methods
    are available to get and set some or all parameters in the forms of dictionaries.

    TimingModel objects also support a number of functions for computing
    various things like orbital phase, and barycentric versions of TOAs,
    as well as the various derivatives and matrices needed to support fitting.

    TimingModel objects forward attribute lookups to their components, so
    that you can access any method or attribute (in particular Parameters)
    of any Component directly on the TimingModel object, for example as
    ``model.F0``.

    TimingModel objects can be written out to ``.par`` files using
    :func:`pint.models.timing_model.TimingModel.write_parfile` or .
    :func:`pint.models.timing_model.TimingModel.as_parfile`::

        >>> model.write_parfile("output.par")

    PINT Parameters supported (here, rather than in any Component):

    .. paramtable::
        :class: pint.models.timing_model.TimingModel

    Parameters
    ----------
    name: str, optional
        The name of the timing model.
    components: list of Component, optional
        The model components for timing model.

    Notes
    -----
    PINT models pulsar pulse time of arrival at observer from its emission process and
    propagation to observer. Emission generally modeled as pulse 'Phase' and propagation.
    'time delay'. In pulsar timing different astrophysics phenomenons are separated to
    time model components for handling a specific emission or propagation effect.

    Each timing model component generally requires the following parts:

        - Timing Parameters
        - Delay/Phase functions which implements the time delay and phase.
        - Derivatives of delay and phase respect to parameter for fitting toas.

    Each timing parameters are stored as TimingModel attribute in the type of
    :class:`~pint.models.parameter.Parameter` delay or phase and its derivatives
    are implemented as TimingModel Methods.

    Attributes
    ----------
    name : str
        The name of the timing model
    component_types : list
        A list of the distinct categories of component. For example,
        delay components will be register as 'DelayComponent'.
    top_level_params : list
        Names of parameters belonging to the TimingModel as a whole
        rather than to any particular component.
    """

    def __init__(self, name="", components=[]):
        if not isinstance(name, str):
            raise ValueError(
                "First parameter should be the model name, was {!r}".format(name)
            )
        self.name = name
        self.component_types = []
        self.top_level_params = []
        self.add_param_from_top(
            strParameter(
                name="PSR", description="Source name", aliases=["PSRJ", "PSRB"]
            ),
            "",
        )
        self.add_param_from_top(
            strParameter(name="TRACK", description="Tracking Information"), ""
        )
        self.add_param_from_top(
            strParameter(name="EPHEM", description="Ephemeris to use"), ""
        )
        self.add_param_from_top(
            strParameter(name="CLOCK", description="Timescale to use", aliases=["CLK"]),
            "",
        )
        self.add_param_from_top(
            strParameter(name="UNITS", description="Units (TDB assumed)"), ""
        )
        self.add_param_from_top(
            MJDParameter(name="START", description="Start MJD for fitting"), ""
        )
        self.add_param_from_top(
            MJDParameter(name="FINISH", description="End MJD for fitting"), ""
        )
        self.add_param_from_top(
            floatParameter(
                name="RM", description="Rotation measure", units=u.radian / u.m**2
            ),
            "",
        )
        self.add_param_from_top(
            strParameter(
                name="INFO",
                description="Tells TEMPO to write some extra information about frontend/backend combinations; -f is recommended",
            ),
            "",
        )
        self.add_param_from_top(
            strParameter(
                name="TIMEEPH",
                description="Time ephemeris to use for TDB conversion; for PINT, always FB90",
            ),
            "",
        )
        self.add_param_from_top(
            strParameter(
                name="T2CMETHOD",
                description="Method for transforming from terrestrial to celestial frame (IAU2000B/TEMPO; PINT only supports ????)",
            ),
            "",
        )
        self.add_param_from_top(
            strParameter(
                name="BINARY",
                description="Pulsar System/Binary model",
                value=None,
            ),
            "",
        )
        self.add_param_from_top(
            boolParameter(
                name="DILATEFREQ",
                value=False,
                description="Whether or not TEMPO2 should apply gravitational redshift and time dilation to observing frequency (Y/N; PINT only supports N)",
            ),
            "",
        )
        self.add_param_from_top(
            boolParameter(
                name="DMDATA",
                value=False,
                description="Was the fit done using per-TOA DM information?",
            ),
            "",
        )
        self.add_param_from_top(
            intParameter(
                name="NTOA", value=0, description="Number of TOAs used in the fitting"
            ),
            "",
        )
        self.add_param_from_top(
            floatParameter(
                name="CHI2",
                units="",
                description="Chi-squared value obtained during fitting",
            ),
            "",
        )
        self.add_param_from_top(
            floatParameter(
                name="CHI2R",
                units="",
                description="Reduced chi-squared value obtained during fitting",
            ),
            "",
        )

        self.add_param_from_top(
            floatParameter(
                name="TRES",
                units=u.us,
                description="TOA residual after fitting",
            ),
            "",
        )
        self.add_param_from_top(
            floatParameter(
                name="DMRES",
                units=u.pc / u.cm**3,
                description="DM residual after fitting (wideband only)",
            ),
            "",
        )
        for cp in components:
            self.add_component(cp, setup=False, validate=False)

    def __repr__(self):
        return "{}(\n  {}\n)".format(
            self.__class__.__name__,
            ",\n  ".join(str(v) for k, v in sorted(self.components.items())),
        )

    def __str__(self):
        return self.as_parfile()

    def validate(self, allow_tcb=False):
        """Validate component setup.

        The checks include required parameters and parameter values, and component types.
        See also: :func:`pint.models.timing_model.TimingModel.validate_component_types`.
        """
        if self.DILATEFREQ.value:
            warn("PINT does not support 'DILATEFREQ Y'")
            self.DILATEFREQ.value = False
        if self.TIMEEPH.value not in [None, "FB90"]:
            warn("PINT only supports 'TIMEEPH FB90'")
            self.TIMEEPH.value = "FB90"
        if self.T2CMETHOD.value not in [None, "IAU2000B"]:  # FIXME: really?
            warn("PINT only supports 'T2CMETHOD IAU2000B'")
            self.T2CMETHOD.value = "IAU2000B"

        if self.UNITS.value not in [None, "TDB", "TCB"]:
            error_message = f"PINT only supports 'UNITS TDB'. The given timescale '{self.UNITS.value}' is invalid."
            raise ValueError(error_message)
        elif self.UNITS.value == "TCB":
            if not allow_tcb:
                error_message = """The TCB timescale is not fully supported by PINT. 
                PINT only supports 'UNITS TDB' internally. See https://nanograv-pint.readthedocs.io/en/latest/explanation.html#time-scales
                for an explanation on different timescales. A TCB par file can be 
                converted to TDB using the `tcb2tdb` command like so:
                
                    $ tcb2tdb J1234+6789_tcb.par J1234+6789_tdb.par
                
                However, this conversion is not exact and a fit must be performed to obtain 
                reliable results. Note that PINT only supports writing TDB par files. 
                """
                raise ValueError(error_message)
            else:
                log.warning(
                    "PINT does not support 'UNITS TCB' internally. Reading this par file nevertheless "
                    "because the `allow_tcb` option was given. This `TimingModel` object should not be "
                    "used for anything except converting to TDB."
                )
        if not self.START.frozen:
            warn("START cannot be unfrozen... Setting START.frozen to True")
            self.START.frozen = True
        if not self.FINISH.frozen:
            warn("FINISH cannot be unfrozen... Setting FINISH.frozen to True")
            self.FINISH.frozen = True

        for cp in self.components.values():
            cp.validate()

        self.validate_component_types()

    def validate_component_types(self):
        """Physically motivated validation of a timing model. This method checks the
        compatibility of different model components when used together.

        This function throws an error if multiple deterministic components that model
        the same effect are used together (e.g. :class:`pint.models.astrometry.AstrometryEquatorial`
        and :class:`pint.models.astrometry.AstrometryEcliptic`). It emits a warning if
        a deterministic component and a stochastic component that model the same effect
        are used together (e.g. :class:`pint.models.noise_model.PLDMNoise`
        and :class:`pint.models.dispersion_model.DispersionDMX`). It also requires that
        one and only one :class:`pint.models.spindown.SpindownBase` component is present
        in a timing model.
        """

        def num_components_of_type(type):
            return len(
                list(filter(lambda c: isinstance(c, type), self.components.values()))
            )

        from pint.models.spindown import SpindownBase

        assert (
            num_components_of_type(SpindownBase) == 1
        ), "Model must have one and only one spindown component (Spindown or another subclass of SpindownBase)."

        from pint.models.astrometry import Astrometry

        assert (
            num_components_of_type(Astrometry) <= 1
        ), "Model can have at most one Astrometry component."

        from pint.models.solar_system_shapiro import SolarSystemShapiro

        if num_components_of_type(SolarSystemShapiro) == 1:
            assert (
                num_components_of_type(Astrometry) == 1
            ), "Model cannot have SolarSystemShapiro component without an Astrometry component."

        from pint.models.pulsar_binary import PulsarBinary

        has_binary_attr = hasattr(self, "BINARY") and self.BINARY.value
        if has_binary_attr:
            assert (
                num_components_of_type(PulsarBinary) == 1
            ), "BINARY attribute is set but no PulsarBinary component found."
        assert (
            num_components_of_type(PulsarBinary) <= 1
        ), "Model can have at most one PulsarBinary component."

        from pint.models.solar_wind_dispersion import SolarWindDispersionBase

        assert (
            num_components_of_type(SolarWindDispersionBase) <= 1
        ), "Model can have at most one solar wind dispersion component."

        from pint.models.dispersion_model import DispersionDMX
        from pint.models.wave import Wave
        from pint.models.wavex import WaveX
        from pint.models.dmwavex import DMWaveX
        from pint.models.noise_model import PLRedNoise, PLDMNoise

        if num_components_of_type((DispersionDMX, PLDMNoise, DMWaveX)) > 1:
            log.warning(
                "DispersionDMX, PLDMNoise, and DMWaveX cannot be used together. "
                "They are ways of modelling the same effect."
            )
        if num_components_of_type((Wave, WaveX, PLRedNoise)) > 1:
            log.warning(
                "Wave, WaveX, and PLRedNoise cannot be used together. "
                "They are ways of modelling the same effect."
            )

    # def __str__(self):
    #    result = ""
    #    comps = self.components
    #    for k, cp in list(comps.items()):
    #        result += "In component '%s'" % k + "\n\n"
    #        for pp in cp.params:
    #            result += str(getattr(cp, pp)) + "\n"
    #    return result

    def __getattr__(self, name):
        if name in ["components", "component_types", "search_cmp_attr"]:
            raise AttributeError
        if not hasattr(self, "component_types"):
            raise AttributeError
        for cp in self.components.values():
            try:
                return getattr(cp, name)
            except AttributeError:
                continue
        raise AttributeError(
            f"Attribute {name} not found in TimingModel or any Component"
        )

    def __setattr__(self, name, value):
        """Mostly this just sets ``self.name = value``.  But in the case where they are both :class:`Parameter` instances
        with different names, this copies the ``quantity``, ``uncertainty``, ``frozen`` attributes only.
        """
        if isinstance(value, (Parameter, prefixParameter)) and name != value.name:
            for p in ["quantity", "uncertainty", "frozen"]:
                setattr(getattr(self, name), p, getattr(value, p))
        else:
            super().__setattr__(name, value)

    @property_exists
    def params_ordered(self):
        """List of all parameter names in this model and all its components.
        This is the same as `params`."""

        # Historically, this was different from `params` because Python
        # dictionaries were unordered until Python 3.7. Now there is no reason for
        # them to be different.

        warn(
            "`TimingModel.params_ordered` is now deprecated and may be removed in the future. "
            "Use `TimingModel.params` instead. It gives the same output as `TimingModel.params_ordered`.",
            DeprecationWarning,
        )

        return self.params

    @property_exists
    def params(self):
        """List of all parameter names in this model and all its components, in a sensible order."""

        # Define the order of components in the list
        # Any not included will be printed between the first and last set.
        # FIXME: make order completely canonical (sort components by name?)

        start_order = ["astrometry", "spindown", "dispersion"]
        last_order = ["jump_delay"]
        compdict = self.get_components_by_category()
        used_cats = set()
        pstart = copy.copy(self.top_level_params)
        for cat in start_order:
            if cat not in compdict:
                continue
            cp = compdict[cat]
            for cpp in cp:
                pstart += cpp.params
            used_cats.add(cat)
        pend = []
        for cat in last_order:
            if cat not in compdict:
                continue

            cp = compdict[cat]
            for cpp in cp:
                pend += cpp.parms
            used_cats.add(cat)
        # Now collect any components that haven't already been included in the list
        pmid = []
        for cat in compdict:
            if cat in used_cats:
                continue
            cp = compdict[cat]
            for cpp in cp:
                pmid += cpp.params
            used_cats.add(cat)

        return pstart + pmid + pend

    @property_exists
    def free_params(self):
        """List of all the free parameters in the timing model.
        Can be set to change which are free.

        These are ordered as ``self.params`` does.

        Upon setting, order does not matter, and aliases are accepted.
        ValueError is raised if a parameter is not recognized.

        On setting, parameter aliases are converted with
        :func:`pint.models.timing_model.TimingModel.match_param_aliases`.
        """
        return [p for p in self.params if not getattr(self, p).frozen]

    @free_params.setter
    def free_params(self, params):
        params_true = {self.match_param_aliases(p) for p in params}
        for p in self.params:
            getattr(self, p).frozen = p not in params_true
            params_true.discard(p)
        if params_true:
            raise ValueError(
                f"Parameter(s) are familiar but not in the model: {params}"
            )

    @property_exists
    def fittable_params(self):
        """List of parameters that are fittable, i.e., the parameters
        which have a derivative implemented. These derivatives are usually
        accessed via the `d_delay_d_param` and `d_phase_d_param` methods."""
        return [
            p
            for p in self.params
            if (
                p in self.phase_deriv_funcs
                or p in self.delay_deriv_funcs
                or (
                    (
                        hasattr(self, "toasigma_deriv_funcs")
                        and p in self.toasigma_deriv_funcs
                    )
                )
                or (hasattr(self[p], "prefix") and self[p].prefix == "ECORR")
            )
        ]

    def match_param_aliases(self, alias):
        """Return PINT parameter name corresponding to this alias.

        Parameters
        ----------
        alias: str
           Parameter's alias.

        Returns
        -------
        str
            PINT parameter name corresponding to the input alias.
        """
        # Search the top level first.
        for p in self.top_level_params:
            if p == alias:
                return p
            if alias in getattr(self, p).aliases:
                return p
        # if not in the top level, check parameters.
        pint_par = None
        for cp in self.components.values():
            try:
                pint_par = cp.match_param_aliases(alias)
            except UnknownParameter:
                continue
            return pint_par

        raise UnknownParameter(f"{alias} is not recognized as a parameter or alias")

    def get_params_dict(self, which="free", kind="quantity"):
        """Return a dict mapping parameter names to values.

        This can return only the free parameters or all; and it can return the
        parameter objects, the floating-point values, or the uncertainties.

        Parameters
        ----------
        which : "free", "all"
        kind : "quantity", "value", "uncertainty"

        Returns
        -------
        OrderedDict
        """
        if which == "free":
            ps = self.free_params
        elif which == "all":
            ps = self.params
        else:
            raise ValueError("get_params_dict expects which to be 'all' or 'free'")
        c = OrderedDict()
        for p in ps:
            q = getattr(self, p)
            if kind == "quantity":
                c[p] = q
            elif kind in ("value", "num"):
                c[p] = q.value
            elif kind == "uncertainty":
                c[p] = q.uncertainty_value
            else:
                raise ValueError(f"Unknown kind {kind!r}")
        return c

    def get_params_of_component_type(self, component_type):
        """Get a list of parameters belonging to a component type.

        Parameters
        ----------
        component_type : "PhaseComponent", "DelayComponent", "NoiseComponent"

        Returns
        -------
        list
        """
        component_type_list_str = f"{component_type}_list"
        if hasattr(self, component_type_list_str):
            component_type_list = getattr(self, component_type_list_str)
            return [
                param for component in component_type_list for param in component.params
            ]
        else:
            return []

    def set_param_values(self, fitp):
        """Set the model parameters to the value contained in the input dict.

        Ex. model.set_param_values({'F0':60.1,'F1':-1.3e-15})
        """
        # In Powell fitter this sometimes fails because after some iterations the values change from
        # plain float to Quantities. No idea why.
        for k, v in fitp.items():
            p = getattr(self, k)
            if isinstance(v, (Parameter, prefixParameter)):
                if v.value is None:
                    raise ValueError(f"Parameter {v} is unset")
                p.value = v.value
            elif isinstance(v, u.Quantity):
                p.value = v.to_value(p.units)
            else:
                p.value = v

    def set_param_uncertainties(self, fitp):
        """Set the model parameters to the value contained in the input dict."""
        for k, v in fitp.items():
            p = getattr(self, k)
            p.uncertainty = v if isinstance(v, u.Quantity) else v * p.units

    @property_exists
    def components(self):
        """All the components in a dictionary indexed by name."""
        comps = {}
        for ct in self.component_types:
            for cp in getattr(self, f"{ct}_list"):
                comps[cp.__class__.__name__] = cp
        return comps

    @property_exists
    def delay_funcs(self):
        """List of all delay functions."""
        dfs = []
        for d in self.DelayComponent_list:
            dfs += d.delay_funcs_component
        return dfs

    @property_exists
    def phase_funcs(self):
        """List of all phase functions."""
        pfs = []
        for p in self.PhaseComponent_list:
            pfs += p.phase_funcs_component
        return pfs

    @property_exists
    def is_binary(self):
        """Does the model describe a binary pulsar?"""
        return any(x.startswith("Binary") for x in self.components.keys())

    def orbital_phase(self, barytimes, anom="mean", radians=True):
        """Return orbital phase (in radians) at barycentric MJD times.

        Parameters
        ----------
        barytimes: Time, TOAs, array-like, or float
            MJD barycentric time(s). The times to compute the
            orbital phases.  Needs to be a barycentric time in TDB.
            If a TOAs instance is passed, the barycentering will happen
            automatically.  If an astropy Time object is passed, it must
            be in scale='tdb'.  If an array-like object is passed or
            a simple float, the time must be in MJD format.
        anom: str, optional
            Type of phase/anomaly. Defaults to "mean".
            Other options are "eccentric" or "true"
        radians: bool, optional
            Units to return.  Defaults to True.
            If False, will return unitless phases in cycles (i.e. 0-1).

        Raises
        ------
        ValueError
            If anom.lower() is not "mean", "ecc*", or "true",
            or if an astropy Time object is passed with scale!="tdb".

        Returns
        -------
        array
            The specified anomaly in radians (with unit), unless
            radians=False, which return unitless cycles (0-1).
        """
        if not self.is_binary:  # punt if not a binary
            return None
        # Find the binary model
        b = self.components[
            [x for x in self.components.keys() if x.startswith("Binary")][0]
        ]
        # Make sure that the binary instance has the binary params
        b.update_binary_object(None)
        # Handle input times and update them in stand-alone binary models
        if isinstance(barytimes, TOAs):
            # If we pass the TOA table, then barycenter the TOAs
            bts = self.get_barycentric_toas(barytimes)
        elif isinstance(barytimes, time.Time):
            if barytimes.scale == "tdb":
                bts = np.asarray(barytimes.mjd_long)
            else:
                raise ValueError("barytimes as Time instance needs scale=='tdb'")
        elif isinstance(barytimes, MJDParameter):
            bts = np.asarray(barytimes.value)  # .value is always a MJD long double
        else:
            bts = np.asarray(barytimes)
        bbi = b.binary_instance  # shorthand
        # Update the times in the stand-alone binary model
        updates = {"barycentric_toa": bts}
        bbi.update_input(**updates)
        if anom.lower() == "mean":
            anoms = bbi.M()
        elif anom.lower().startswith("ecc"):
            anoms = bbi.E()
        elif anom.lower() == "true":
            anoms = bbi.nu()  # can be negative
        else:
            raise ValueError(f"anom='{anom}' is not a recognized type of anomaly")
        # Make sure all angles are between 0-2*pi
        anoms = np.remainder(anoms.value, 2 * np.pi)
        # return with radian units or return as unitless cycles from 0-1
        return anoms * u.rad if radians else anoms / (2 * np.pi)

    def pulsar_radial_velocity(self, barytimes):
        """Return line-of-sight velocity of the pulsar relative to the system barycenter at barycentric MJD times.

        Parameters
        ----------
        barytimes: Time, TOAs, array-like, or float
            MJD barycentric time(s). The times to compute the
            orbital phases.  Needs to be a barycentric time in TDB.
            If a TOAs instance is passed, the barycentering will happen
            automatically.  If an astropy Time object is passed, it must
            be in scale='tdb'.  If an array-like object is passed or
            a simple float, the time must be in MJD format.

        Raises
        ------
        ValueError
            If an astropy Time object is passed with scale!="tdb".

        Returns
        -------
        array
            The line-of-sight velocity

        Notes
        -----
        This is the radial velocity of the pulsar.

        See [1]_

        .. [1] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.24
        """
        # this should also update the binary instance
        nu = self.orbital_phase(barytimes, anom="true")
        b = self.components[
            [x for x in self.components.keys() if x.startswith("Binary")][0]
        ]
        bbi = b.binary_instance  # shorthand
        psi = nu + bbi.omega()
        return (
            2
            * np.pi
            * bbi.a1()
            / (bbi.pb() * np.sqrt(1 - bbi.ecc() ** 2))
            * (np.cos(psi) + bbi.ecc() * np.cos(bbi.omega()))
        ).cgs

    def companion_radial_velocity(self, barytimes, massratio):
        """Return line-of-sight velocity of the companion relative to the system barycenter at barycentric MJD times.

        Parameters
        ----------
        barytimes: Time, TOAs, array-like, or float
            MJD barycentric time(s). The times to compute the
            orbital phases.  Needs to be a barycentric time in TDB.
            If a TOAs instance is passed, the barycentering will happen
            automatically.  If an astropy Time object is passed, it must
            be in scale='tdb'.  If an array-like object is passed or
            a simple float, the time must be in MJD format.
        massratio : float
            Ratio of pulsar mass to companion mass


        Raises
        ------
        ValueError
            If an astropy Time object is passed with scale!="tdb".

        Returns
        -------
        array
            The line-of-sight velocity

        Notes
        -----
        This is the radial velocity of the companion.

        See [1]_

        .. [1] Lorimer & Kramer, 2008, "The Handbook of Pulsar Astronomy", Eqn. 8.24
        """
        return -self.pulsar_radial_velocity(barytimes) * massratio

    def conjunction(self, baryMJD):
        """Return the time(s) of the first superior conjunction(s) after baryMJD.

        Args
        ----
        baryMJD: floats or Time
            barycentric (tdb) MJD(s) prior to the
            conjunction we are looking for.  Can be an array.

        Raises
        ------
        ValueError
            If baryMJD is an astropy Time object with scale!="tdb".

        Returns
        -------
        float or array
            The barycentric MJD(tdb) time(s) of the next superior conjunction(s) after baryMJD
        """
        if not self.is_binary:  # punt if not a binary
            return None
        # Find the binary model
        b = self.components[
            [x for x in self.components.keys() if x.startswith("Binary")][0]
        ]
        bbi = b.binary_instance  # shorthand
        # Superior conjunction occurs when true anomaly + omega == 90 deg
        # We will need to solve for this using a root finder (brentq)
        # This is the function to root-find:

        def funct(t):
            nu = self.orbital_phase(t, anom="true")
            return np.remainder((nu + bbi.omega()).value, 2 * np.pi) - np.pi / 2

        # Handle the input time(s)
        if isinstance(baryMJD, time.Time):
            if baryMJD.scale == "tdb":
                bts = np.atleast_1d(baryMJD.mjd)
            else:
                raise ValueError("baryMJD as Time instance needs scale=='tdb'")
        else:
            bts = np.atleast_1d(baryMJD)
        # Step over the maryMJDs
        scs = []
        for bt in bts:
            # Make 11 times over one orbit after bt
            pb = self.pb()[0].to_value("day")
            ts = np.linspace(bt, bt + pb, 11)
            # Compute the true anomalies and omegas for those times
            nus = self.orbital_phase(ts, anom="true")
            omegas = bbi.omega()
            x = np.remainder((nus + omegas).value, 2 * np.pi) - np.pi / 2
            # find the lowest index where x is just below 0
            for lb in range(len(x)):
                if x[lb] < 0 and x[lb + 1] > 0:
                    break
            # Now use scipy to find the root
            scs.append(brentq(funct, ts[lb], ts[lb + 1]))
        return scs[0] if len(scs) == 1 else np.asarray(scs)

    @property_exists
    def dm_funcs(self):
        """List of all dm value functions."""
        dmfs = []
        for cp in self.components.values():
            if hasattr(cp, "dm_value_funcs"):
                dmfs += cp.dm_value_funcs
            else:
                continue
        return dmfs

    @property_exists
    def has_correlated_errors(self):
        """Whether or not this model has correlated errors."""

        return (
            "NoiseComponent" in self.component_types
            and len(
                [
                    nc
                    for nc in self.NoiseComponent_list
                    if nc.introduces_correlated_errors
                ]
            )
            > 0
        )

    @property_exists
    def has_time_correlated_errors(self):
        """Whether or not this model has time-correlated errors."""

        return (
            "NoiseComponent" in self.component_types
            and len(
                [
                    nc
                    for nc in self.NoiseComponent_list
                    if (nc.introduces_correlated_errors and nc.is_time_correlated)
                ]
            )
            > 0
        )

    @property_exists
    def covariance_matrix_funcs(self):
        """List of covariance matrix functions."""
        cvfs = []
        if "NoiseComponent" in self.component_types:
            for nc in self.NoiseComponent_list:
                cvfs += nc.covariance_matrix_funcs
        return cvfs

    @property_exists
    def dm_covariance_matrix_funcs(self):
        """List of covariance matrix functions."""
        cvfs = []
        if "NoiseComponent" in self.component_types:
            for nc in self.NoiseComponent_list:
                cvfs += nc.dm_covariance_matrix_funcs_component
        return cvfs

    # Change sigma to uncertainty to avoid name conflict.
    @property_exists
    def scaled_toa_uncertainty_funcs(self):
        """List of scaled toa uncertainty functions."""
        ssfs = []
        if "NoiseComponent" in self.component_types:
            for nc in self.NoiseComponent_list:
                ssfs += nc.scaled_toa_sigma_funcs
        return ssfs

    # Change sigma to uncertainty to avoid name conflict.
    @property_exists
    def scaled_dm_uncertainty_funcs(self):
        """List of scaled dm uncertainty functions."""
        ssfs = []
        if "NoiseComponent" in self.component_types:
            for nc in self.NoiseComponent_list:
                if hasattr(nc, "scaled_dm_sigma_funcs"):
                    ssfs += nc.scaled_dm_sigma_funcs
        return ssfs

    @property_exists
    def basis_funcs(self):
        """List of scaled uncertainty functions."""
        bfs = []
        if "NoiseComponent" in self.component_types:
            for nc in self.NoiseComponent_list:
                bfs += nc.basis_funcs
        return bfs

    @property_exists
    def phase_deriv_funcs(self):
        """List of derivative functions for phase components."""
        return self.get_deriv_funcs("PhaseComponent")

    @property_exists
    def delay_deriv_funcs(self):
        """List of derivative functions for delay components."""
        return self.get_deriv_funcs("DelayComponent")

    @property_exists
    def dm_derivs(self):  #  TODO need to be careful about the name here.
        """List of DM derivative functions."""
        return self.get_deriv_funcs("DelayComponent", "dm")

    @property_exists
    def toasigma_derivs(self):
        """List of scaled TOA uncertainty derivative functions"""
        return self.get_deriv_funcs("NoiseComponent", "toasigma")

    @property_exists
    def d_phase_d_delay_funcs(self):
        """List of d_phase_d_delay functions."""
        Dphase_Ddelay = []
        for cp in self.PhaseComponent_list:
            Dphase_Ddelay += cp.phase_derivs_wrt_delay
        return Dphase_Ddelay

    def get_deriv_funcs(self, component_type, derivative_type=""):
        """Return a dictionary of derivative functions.

        Parameters
        ----------
        component_type: str
            Type of component to look for derivatives ("PhaseComponent",
            "DelayComponent", or "NoiseComponent")
        derivative_type: str
            Derivative type ("", "dm", or "toasigma". Empty string
            denotes delay and phase derivatives.)
        """
        # TODO, this function can be a more generic function collector.
        deriv_funcs = defaultdict(list)
        if derivative_type != "":
            derivative_type += "_"
        for cp in getattr(self, f"{component_type}_list"):
            try:
                df = getattr(cp, f"{derivative_type}deriv_funcs")
            except AttributeError:
                continue
            for k, v in df.items():
                deriv_funcs[k] += v
        return dict(deriv_funcs)

    def search_cmp_attr(self, name):
        """Search for an attribute in all components.

        Return the component, or None.

        If multiple components have same attribute, it will return the first
        component.
        """
        for cp in list(self.components.values()):
            if hasattr(cp, name):
                return cp
        raise AttributeError(f"{name} not found in any component")

    def get_component_type(self, component):
        """Identify the component object's type.

        Parameters
        ----------
        component: component instance
           The component object need to be inspected.

        Note
        ----
        Since a component can be an inheritance from other component We inspect
        all the component object bases. "inspect getmro" method returns the
        base classes (including 'object') in method resolution order. The
        third level of inheritance class name is what we want.
        Object --> component --> TypeComponent. (i.e. DelayComponent)
        This class type is in the third to the last of the getmro returned
        result.

        """
        # check component type
        comp_base = inspect.getmro(component.__class__)
        if comp_base[-2].__name__ != "Component":
            raise TypeError(
                f"Class '{component.__class__.__name__}' is not a Component type class."
            )
        elif len(comp_base) < 3:
            raise TypeError(
                f"'{component.__class__.__name__}' class is not a subclass of 'Component' class."
            )
        else:
            comp_type = comp_base[-3].__name__
        return comp_type

    def map_component(self, component):
        """Get the location of component.

        Parameters
        ----------
        component: str or `Component` object
            Component name or component object.

        Returns
        -------
        comp: `Component` object
            Component object.
        order: int
            The index/order of the component in the component list
        host_list: List
            The host list of the component.
        comp_type: str
            The component type (e.g., Delay or Phase)
        """
        comps = self.components
        if isinstance(component, str):
            if component not in list(comps.keys()):
                raise AttributeError(f"No '{component}' in the timing model.")
            comp = comps[component]
        elif component in list(comps.values()):
            comp = component
        else:
            raise AttributeError(
                f"No '{component.__class__.__name__}' in the timing model."
            )
        comp_type = self.get_component_type(comp)
        host_list = getattr(self, f"{comp_type}_list")
        order = host_list.index(comp)
        return comp, order, host_list, comp_type

    def add_component(
        self, component, order=DEFAULT_ORDER, force=False, setup=True, validate=True
    ):
        """Add a component into TimingModel.

        Parameters
        ----------
        component : Component
            The component to be added to the timing model.
        order : list, optional
            The component category order list. Default is the DEFAULT_ORDER.
        force : bool, optional
            If true, add a duplicate component. Default is False.
        """
        comp_type = self.get_component_type(component)
        cur_cps = []
        if comp_type in self.component_types:
            comp_list = getattr(self, f"{comp_type}_list")
            for cp in comp_list:
                # If component order is not defined.
                cp_order = (
                    order.index(cp.category) if cp.category in order else len(order) + 1
                )
                cur_cps.append((cp_order, cp))
            # Check if the component has been added already.
            if component.__class__ in (x.__class__ for x in comp_list):
                log.warning(
                    f"Component '{component.__class__.__name__}' is already present but was added again."
                )
                if not force:
                    raise ValueError(
                        f"Component '{component.__class__.__name__}' is already present and will not be "
                        f"added again. To force add it, use force=True option."
                    )
        else:
            self.component_types.append(comp_type)
        # link new component to TimingModel
        component._parent = self

        # If the category is not in the order list, it will be added to the end.
        if component.category not in order:
            new_cp = len(order) + 1, component
        else:
            new_cp = order.index(component.category), component
        # add new component
        cur_cps.append(new_cp)
        cur_cps.sort(key=lambda x: x[0])
        new_comp_list = [c[1] for c in cur_cps]
        setattr(self, f"{comp_type}_list", new_comp_list)
        # Set up components
        if setup:
            self.setup()
        # Validate inputs
        if validate:
            self.validate()

    def remove_component(self, component):
        """Remove one component from the timing model.

        Parameters
        ----------
        component: str or `Component` object
            Component name or component object.
        """
        cp, co_order, host, cp_type = self.map_component(component)
        host.remove(cp)

    def _locate_param_host(self, param):
        """Search for the parameter host component in the timing model.

        Parameters
        ----------
        param: str
            Target parameter.

        Return
        ------
        list of tuples
           All possible components that host the target parameter.  The first
           element is the component object that have the target parameter, the
           second one is the parameter object. If it is a prefix-style parameter,
           it will return one example of such parameter.
        """
        result_comp = []
        for cp_name, cp in self.components.items():
            if param in cp.params:
                result_comp.append((cp_name, cp, getattr(cp, param)))
            else:
                # search for prefixed parameter
                prefixs = cp.param_prefixs
                try:
                    prefix, index_str, index = split_prefixed_name(param)
                except PrefixError:
                    prefix = param

                if prefix in prefixs.keys():
                    result_comp.append(cp_name, cp, getattr(cp, prefixs[param][0]))

        return result_comp

    def get_components_by_category(self):
        """Return a dict of this model's component objects keyed by the category name."""
        categorydict = defaultdict(list)
        for cp in self.components.values():
            categorydict[cp.category].append(cp)
        # Convert from defaultdict to dict
        return dict(categorydict)

    def add_param_from_top(self, param, target_component, setup=False):
        """Add a parameter to a timing model component.

        Parameters
        ----------
        param: pint.models.parameter.Parameter
            Parameter instance
        target_component: str
            Parameter host component name. If given as "" it would add
            parameter to the top level `TimingModel` class
        setup: bool, optional
            Flag to run setup() function.
        """
        if target_component == "":
            setattr(self, param.name, param)
            self.top_level_params += [param.name]
        elif target_component in list(self.components.keys()):
            self.components[target_component].add_param(param, setup=setup)
        else:
            raise AttributeError(
                f"Can not find component '{target_component}' in " "timing model."
            )

    def remove_param(self, param):
        """Remove a parameter from timing model.

        Parameters
        ----------
        param: str
            The name of parameter to be removed.
        """
        param_map = self.get_params_mapping()
        if param not in param_map:
            raise AttributeError(f"Can not find '{param}' in timing model.")
        if param_map[param] == "timing_model":
            delattr(self, param)
            self.top_level_params.remove(param)
        else:
            target_component = param_map[param]
            self.components[target_component].remove_param(param)
        self.setup()

    def get_params_mapping(self):
        """Report which component each parameter name comes from."""
        param_mapping = {p: "TimingModel" for p in self.top_level_params}
        for cp in list(self.components.values()):
            for pp in cp.params:
                param_mapping[pp] = cp.__class__.__name__
        return param_mapping

    def get_params_of_type_top(self, param_type):
        result = []
        for cp in self.components.values():
            result += cp.get_params_of_type(param_type)
        return result

    def get_prefix_mapping(self, prefix):
        """Get the index mapping for the prefix parameters.

        Parameters
        ----------
        prefix : str
           Name of prefix.

        Returns
        -------
        dict
           A dictionary with prefix parameter real index as key and parameter
           name as value.
        """
        for cp in self.components.values():
            mapping = cp.get_prefix_mapping_component(prefix)
            if len(mapping) != 0:
                return mapping
        raise ValueError(f"Can not find prefix {prefix!r}")

    def get_prefix_list(self, prefix, start_index=0):
        """Return the Quantities associated with a sequence of prefix parameters.

        Parameters
        ----------
        prefix : str
            Name of prefix.
        start_index : int
            The index to start the sequence at (DM1, DM2, ... vs F0, F1, ...)

        Returns
        -------
        list of astropy.units.Quantity
            The ``.quantity`` associated with parameter prefix + start_index,
            prefix + (start_index+1), ... up to the last that exists and is set.

        Raises
        ------
        ValueError
            If any prefix parameters exist outside the sequence that would be returned
            (for example if there are DM1 and DM3 but not DM2, or F0 exists but start_index
            was given as 1).
        """
        matches = {}
        for p in self.params:
            if not p.startswith(prefix):
                continue
            pm = getattr(self, p)
            if not pm.is_prefix:
                continue
            if pm.quantity is None:
                continue
            if pm.prefix != prefix:
                continue
            matches[pm.index] = pm
        r = []
        i = start_index
        while True:
            try:
                r.append(matches.pop(i).quantity)
            except KeyError:
                break
            i += 1
        if matches:
            raise ValueError(
                f"Unused prefix parameters for start_index {start_index}: {matches}"
            )
        return r

    def param_help(self):
        """Print help lines for all available parameters in model."""
        return "".join(
            "{:<40}{}\n".format(cp, getattr(self, par).help_line())
            for par, cp in self.get_params_mapping().items()
        )

    def delay(self, toas, cutoff_component="", include_last=True):
        """Total delay for the TOAs.

        Return the total delay which will be subtracted from the given
        TOA to get time of emission at the pulsar.

        Parameters
        ----------
        toas: pint.toa.TOAs
            The toas for analysis delays.
        cutoff_component: str
            The delay component name that a user wants the calculation to stop
            at.
        include_last: bool
            If the cutoff delay component is included.
        """
        delay = np.zeros(toas.ntoas) * u.second
        if cutoff_component == "":
            idx = len(self.DelayComponent_list)
        else:
            delay_names = [x.__class__.__name__ for x in self.DelayComponent_list]
            if cutoff_component not in delay_names:
                raise KeyError(f"No delay component named '{cutoff_component}'.")

            idx = delay_names.index(cutoff_component)
            if include_last:
                idx += 1
        # Do NOT cycle through delay_funcs - cycle through components until cutoff
        for dc in self.DelayComponent_list[:idx]:
            for df in dc.delay_funcs_component:
                delay += df(toas, delay)
        return delay

    def phase(self, toas, abs_phase=None):
        """Return the model-predicted pulse phase for the given TOAs.

        This is the phase as observed at the observatory at the exact moment
        specified in each TOA. The result is a :class:`pint.phase.Phase` object.
        """
        # First compute the delays to "pulsar time"
        delay = self.delay(toas)
        phase = Phase(np.zeros(toas.ntoas), np.zeros(toas.ntoas))
        # Then compute the relevant pulse phases
        for pf in self.phase_funcs:
            phase += Phase(pf(toas, delay))

        # abs_phase defaults to True if AbsPhase is in the model, otherwise to
        # False.  Of course, if you manually set it, it will use that setting.
        if abs_phase is None:
            abs_phase = "AbsPhase" in list(self.components.keys())

        # This function gets called in `Residuals.calc_phase_resids()` with `abs_phase=True`
        # by default. Hence, this branch is not run by default.
        if not abs_phase:
            return phase

        if "AbsPhase" not in list(self.components.keys()):
            log.info("Creating a TZR TOA (AbsPhase) using the given TOAs object.")

            # if no absolute phase (TZRMJD), add the component to the model and calculate it
            self.add_tzr_toa(toas)

        tz_toa = self.get_TZR_toa(toas)
        tz_delay = self.delay(tz_toa)
        tz_phase = Phase(np.zeros(len(toas.table)), np.zeros(len(toas.table)))
        for pf in self.phase_funcs:
            tz_phase += Phase(pf(tz_toa, tz_delay))
        return phase - tz_phase

    def add_tzr_toa(self, toas):
        """Create a TZR TOA for the given TOAs object and add it to
        the timing model. This corresponds to TOA closest to the PEPOCH."""
        from pint.models.absolute_phase import AbsPhase

        self.add_component(AbsPhase(), validate=False)
        self.make_TZR_toa(toas)
        self.validate()

    def total_dm(self, toas):
        """Calculate dispersion measure from all the dispersion type of components."""
        # Here we assume the unit would be the same for all the dm value function.
        # By doing so, we do not have to hard code an unit here.
        dm = self.dm_funcs[0](toas)

        for dm_f in self.dm_funcs[1::]:
            dm += dm_f(toas)
        return dm

    def total_dispersion_slope(self, toas):
        """Calculate the dispersion slope from all the dispersion-type components."""
        dm_tot = self.total_dm(toas)
        return dispersion_slope(dm_tot)

    def toa_covariance_matrix(self, toas):
        """Get the TOA covariance matrix for noise models.

        If there is no noise model component provided, a diagonal matrix with
        TOAs error as diagonal element will be returned.
        """
        result = np.zeros((len(toas), len(toas)))
        if "ScaleToaError" not in self.components:
            result += np.diag(toas.table["error"].quantity.to(u.s).value ** 2)

        for nf in self.covariance_matrix_funcs:
            result += nf(toas)
        return result

    def dm_covariance_matrix(self, toas):
        """Get the DM covariance matrix for noise models.

        If there is no noise model component provided, a diagonal matrix with
        TOAs error as diagonal element will be returned.
        """
        dms, valid_dm = toas.get_flag_value("pp_dm", as_type=float)
        dmes, valid_dme = toas.get_flag_value("pp_dme", as_type=float)
        dms = np.array(dms)[valid_dm]
        n_dms = len(dms)
        dmes = np.array(dmes)[valid_dme]
        result = np.zeros((n_dms, n_dms))
        # When there is no noise model.
        # FIXME: specifically when there is no DMEFAC
        if len(self.dm_covariance_matrix_funcs) == 0:
            result += np.diag(dmes**2)
            return result

        for nf in self.dm_covariance_matrix_funcs:
            result += nf(toas)
        return result

    def scaled_toa_uncertainty(self, toas):
        """Get the scaled TOA data uncertainties noise models.

        If there is no noise model component provided, a vector with
        TOAs error as values will be returned.

        Parameters
        ----------
        toas: pint.toa.TOAs
            The input data object for TOAs uncertainty.
        """
        ntoa = toas.ntoas
        tbl = toas.table
        result = np.zeros(ntoa) * u.us
        # When there is no noise model.
        if len(self.scaled_toa_uncertainty_funcs) == 0:
            result += tbl["error"].quantity
            return result

        for nf in self.scaled_toa_uncertainty_funcs:
            result += nf(toas)
        return result

    def scaled_dm_uncertainty(self, toas):
        """Get the scaled DM data uncertainties noise models.

        If there is no noise model component provided, a vector with
        DM error as values will be returned.

        Parameters
        ----------
        toas: pint.toa.TOAs
            The input data object for DM uncertainty.
        """
        dm_error, valid = toas.get_flag_value("pp_dme", as_type=float)
        dm_error = np.array(dm_error)[valid] * u.pc / u.cm**3
        result = np.zeros(len(dm_error)) * u.pc / u.cm**3
        # When there is no noise model.
        if len(self.scaled_dm_uncertainty_funcs) == 0:
            result += dm_error
            return result

        for nf in self.scaled_dm_uncertainty_funcs:
            result += nf(toas)
        return result

    def noise_model_designmatrix(self, toas):
        if len(self.basis_funcs) == 0:
            return None
        result = [nf(toas)[0] for nf in self.basis_funcs]
        return np.hstack(list(result))

    def noise_model_basis_weight(self, toas):
        if len(self.basis_funcs) == 0:
            return None
        result = [nf(toas)[1] for nf in self.basis_funcs]
        return np.hstack(list(result))

    def noise_model_dimensions(self, toas):
        """Number of basis functions for each noise model component.

        Returns a dictionary of correlated-noise components in the noise
        model. Each entry contains a tuple (offset, size) where size is the
        number of basis functions for the component, and offset is their
        starting location in the design matrix and weights vector.
        """
        result = {}

        # Correct results rely on this ordering being the
        # same as what is done in the self.basis_funcs
        # property.
        if len(self.basis_funcs) > 0:
            ntot = 0
            for nc in self.NoiseComponent_list:
                bfs = nc.basis_funcs
                if len(bfs) == 0:
                    continue
                nbf = sum(len(bf(toas)[1]) for bf in bfs)
                result[nc.category] = (ntot, nbf)
                ntot += nbf

        return result

    def jump_flags_to_params(self, toas):
        """Add JUMP parameters corresponding to tim_jump flags.

        When a ``.tim`` file contains pairs of JUMP lines, the user's expectation
        is that the TOAs between each pair of flags will be affected by a JUMP, even
        if that JUMP does not appear in the ``.par`` file. (This is how TEMPO works.)
        In PINT, those TOAs have a flag attached, `-tim_jump N`, where N is a
        number that is different for each JUMPed set of TOAs. The goal of this function
        is to add JUMP parameters to the model corresponding to these.

        Some complexities arise: TOAs may also have `-tim_jump` flags associated
        with them, just as flags, for example if such a ``.tim`` file were exported
        in PINT-native format and then reloaded. And models may already have JUMPs
        associated with some or all ``tim_jump`` values.

        This function looks at all the ``tim_jump`` values and adds JUMP parameters
        for any that do not have any. It does not change the TOAs object it is passed.
        """
        from . import jump

        tjvals, idxs = toas.get_flag_value("tim_jump")
        tim_jump_values = set(tjvals)
        tim_jump_values.remove(None)
        if not tim_jump_values:
            log.info("No jump flags to process from .tim file")
            return None
        log.info(f"There are {len(tim_jump_values)} JUMPs from the timfile.")

        if "PhaseJump" not in self.components:
            log.info("PhaseJump component added")
            a = jump.PhaseJump()
            a.setup()
            self.add_component(a)
            self.remove_param("JUMP1")
            a.setup()

        used_indices = set()
        for p in self.get_jump_param_objects():
            if p.key == "-tim_jump":
                used_indices.add(p.index)
                (tjv,) = p.key_value
                if tjv in tim_jump_values:
                    log.info(f"JUMP -tim_jump {tjv} already exists")
                    tim_jump_values.remove(tjv)
        num = max(used_indices) + 1 if used_indices else 1
        if not tim_jump_values:
            log.info("All tim_jump values have corresponding JUMPs")
            return

        # FIXME: arrange for these to be in a sensible order (might not be integers
        # but if they are then lexicographical order is not wanted)
        t_j_v = set()
        for v in tim_jump_values:
            try:
                vi = int(v)
            except ValueError:
                vi = v
            t_j_v.add(vi)
        for v in sorted(t_j_v):
            # Now we need to add a JUMP for each of these
            log.info(f"Adding JUMP -tim_jump {v}")
            param = maskParameter(
                name="JUMP",
                index=num,
                key="-tim_jump",
                key_value=v,
                value=0.0,
                units="second",
                uncertainty=0.0,
            )
            self.add_param_from_top(param, "PhaseJump")
            getattr(self, param.name).frozen = False
            num += 1

        self.components["PhaseJump"].setup()

    def delete_jump_and_flags(self, toa_table, jump_num):
        """Delete jump object from PhaseJump and remove its flags from TOA table.

        This is a helper function for pintk.

        Parameters
        ----------
        toa_table: list or None
            The TOA table which must be modified. In pintk (pulsar.py), for the
            prefit model, this will be all_toas.table["flags"].
            For the postfit model, it will be None (one set of TOA tables for both
            models).
        jump_num: int
            Specifies the index of the jump to be deleted.
        """
        # remove jump of specified index
        self.remove_param(f"JUMP{jump_num}")

        # remove jump flags from selected TOA tables
        if toa_table is not None:
            for d in toa_table:
                if "jump" in d:
                    index_list = d["jump"].split(",")
                    if str(jump_num) in index_list:
                        del index_list[index_list.index(str(jump_num))]
                        if not index_list:
                            del d["jump"]
                        else:
                            d["jump"] = ",".join(index_list)

        # if last jump deleted, remove PhaseJump object from model
        if (
            self.components["PhaseJump"].get_number_of_jumps() == 0
        ):  # means last jump just deleted
            comp_list = getattr(self, "PhaseComponent_list")
            for item in comp_list:
                if isinstance(item, pint.models.jump.PhaseJump):
                    self.remove_component(item)
            return
        self.components["PhaseJump"].setup()

    def get_barycentric_toas(self, toas, cutoff_component=""):
        """Conveniently calculate the barycentric TOAs.

        Parameters
        ----------
        toas: TOAs object
            The TOAs the barycentric corrections are applied on
        cutoff_delay: str, optional
            The cutoff delay component name. If it is not provided, it will
            search for binary delay and apply all the delay before binary.

        Return
        ------
        astropy.units.Quantity
            Barycentered TOAs.
        """
        tbl = toas.table
        if cutoff_component == "":
            delay_list = self.DelayComponent_list
            for cp in delay_list:
                if cp.category == "pulsar_system":
                    cutoff_component = cp.__class__.__name__
        corr = self.delay(toas, cutoff_component, False)
        return tbl["tdbld"] * u.day - corr

    def d_phase_d_toa(self, toas, sample_step=None):
        """Return the finite-difference derivative of phase wrt TOA.

        Parameters
        ----------
        toas : pint.toa.TOAs
            The toas when the derivative of phase will be evaluated at.
        sample_step : float, optional
            Finite difference steps. If not specified, it will take 1000 times the
            spin period.
        """
        copy_toas = copy.deepcopy(toas)
        if sample_step is None:
            pulse_period = 1.0 / (self.F0.quantity)
            sample_step = pulse_period * 2
        # Note that sample_dt is applied cumulatively, so this evaluates phase at TOA-dt and TOA+dt
        sample_dt = [-sample_step, 2 * sample_step]

        sample_phase = []
        for dt in sample_dt:
            dt_array = [dt.value] * copy_toas.ntoas * dt._unit
            deltaT = time.TimeDelta(dt_array)
            copy_toas.adjust_TOAs(deltaT)
            phase = self.phase(copy_toas, abs_phase=False)
            sample_phase.append(phase)
        # Use finite difference method.
        # phase'(t) = (phase(t+h)-phase(t-h))/2+ 1/6*F2*h^2 + ..
        # The error should be near 1/6*F2*h^2
        dp = sample_phase[1] - sample_phase[0]
        d_phase_d_toa = dp.int / (2 * sample_step) + dp.frac / (2 * sample_step)
        del copy_toas
        return d_phase_d_toa.to(u.Hz)

    def d_phase_d_tpulsar(self, toas):
        """Return the derivative of phase wrt time at the pulsar.

        NOT implemented yet.
        """
        raise NotImplementedError

    def d_phase_d_param(self, toas, delay, param):
        """Return the derivative of phase with respect to the parameter.

        This is the derivative of the phase observed at each TOA with
        respect to each parameter. This is closely related to the derivative
        of residuals with respect to each parameter, differing only by a
        factor of the spin frequency and possibly a minus sign. See
        :meth:`pint.models.timing_model.TimingModel.designmatrix` for a way
        of evaluating many derivatives at once.

        The calculation is done by combining the analytical derivatives
        reported by all the components in the model.

        Parameters
        ----------
        toas : pint.toa.TOAs
            The TOAs at which the derivative should be evaluated.
        delay : astropy.units.Quantity or None
            The delay at the TOAs where the derivatives should be evaluated.
            This permits certain optimizations in the derivative calculations;
            the value should be ``self.delay(toas)``.
        param : str
            The name of the parameter to differentiate with respect to.

        Returns
        -------
        astropy.units.Quantity
            The derivative of observed phase with respect to the model parameter.
        """
        # TODO need to do correct chain rule stuff wrt delay derivs, etc
        # Is it safe to assume that any param affecting delay only affects
        # phase indirectly (and vice-versa)??
        if delay is None:
            delay = self.delay(toas)
        par = getattr(self, param)
        result = np.longdouble(np.zeros(toas.ntoas)) / par.units
        phase_derivs = self.phase_deriv_funcs
        if param in list(phase_derivs.keys()):
            for df in phase_derivs[param]:
                result += df(toas, param, delay).to(
                    result.unit, equivalencies=u.dimensionless_angles()
                )
        else:
            # Apply chain rule for the parameters in the delay.
            # total_phase = Phase1(delay(param)) + Phase2(delay(param))
            # d_total_phase_d_param = d_Phase1/d_delay*d_delay/d_param +
            #                         d_Phase2/d_delay*d_delay/d_param
            #                       = (d_Phase1/d_delay + d_Phase2/d_delay) *
            #                         d_delay_d_param
            d_delay_d_p = self.d_delay_d_param(toas, param)
            dpdd_result = np.longdouble(np.zeros(toas.ntoas)) / u.second
            for dpddf in self.d_phase_d_delay_funcs:
                dpdd_result += dpddf(toas, delay)
            result = dpdd_result * d_delay_d_p
        return result.to(result.unit, equivalencies=u.dimensionless_angles())

    def d_delay_d_param(self, toas, param, acc_delay=None):
        """Return the derivative of delay with respect to the parameter."""
        par = getattr(self, param)
        result = np.longdouble(np.zeros(toas.ntoas) << (u.s / par.units))
        delay_derivs = self.delay_deriv_funcs
        if param not in list(delay_derivs.keys()):
            raise AttributeError(
                f"Derivative function for '{param}' is not provided"
                f" or not registered; parameter '{param}' may not be fittable. "
            )
        for df in delay_derivs[param]:
            result += df(toas, param, acc_delay).to(
                result.unit, equivalencies=u.dimensionless_angles()
            )
        return result

    def d_phase_d_param_num(self, toas, param, step=1e-2):
        """Return the derivative of phase with respect to the parameter.

        Compute the value numerically, using a symmetric finite difference.
        """
        # TODO : We need to know the range of parameter.
        par = getattr(self, param)
        ori_value = par.value
        unit = par.units
        h = 1.0 * step if ori_value == 0 else ori_value * step
        parv = [par.value - h, par.value + h]

        phase_i = (
            np.zeros((toas.ntoas, 2), dtype=np.longdouble) * u.dimensionless_unscaled
        )
        phase_f = (
            np.zeros((toas.ntoas, 2), dtype=np.longdouble) * u.dimensionless_unscaled
        )
        for ii, val in enumerate(parv):
            par.value = val
            ph = self.phase(toas, abs_phase=False)
            phase_i[:, ii] = ph.int
            phase_f[:, ii] = ph.frac
        res_i = -phase_i[:, 0] + phase_i[:, 1]
        res_f = -phase_f[:, 0] + phase_f[:, 1]
        result = (res_i + res_f) / (2.0 * h * unit)
        # shift value back to the original value
        par.value = ori_value
        return result

    def d_delay_d_param_num(self, toas, param, step=1e-2):
        """Return the derivative of delay with respect to the parameter.

        Compute the value numerically, using a symmetric finite difference.
        """
        # TODO : We need to know the range of parameter.
        par = getattr(self, param)
        ori_value = par.value
        if ori_value is None:
            # A parameter did not get to use in the model
            log.warning(f"Parameter '{param}' is not used by timing model.")
            return np.zeros(toas.ntoas) * (u.second / par.units)
        unit = par.units
        h = 1.0 * step if ori_value == 0 else ori_value * step
        parv = [par.value - h, par.value + h]
        delay = np.zeros((toas.ntoas, 2))
        for ii, val in enumerate(parv):
            par.value = val
            try:
                delay[:, ii] = self.delay(toas)
            except:
                par.value = ori_value
                raise
        d_delay = (-delay[:, 0] + delay[:, 1]) / 2.0 / h
        par.value = ori_value
        return d_delay * (u.second / unit)

    def d_dm_d_param(self, data, param):
        """Return the derivative of DM with respect to the parameter."""
        par = getattr(self, param)
        result = np.zeros(len(data)) << (u.pc / u.cm**3 / par.units)
        dm_df = self.dm_derivs.get(param, None)
        if dm_df is None:
            if param not in self.params:  # Maybe add differentiable params
                raise AttributeError(f"Parameter {param} does not exist")
            else:
                return result

        for df in dm_df:
            result += df(data, param).to(
                result.unit, equivalencies=u.dimensionless_angles()
            )
        return result

    def d_toasigma_d_param(self, data, param):
        """Return the derivative of the scaled TOA uncertainty with respect to the parameter."""
        par = getattr(self, param)
        result = np.zeros(len(data)) << (u.s / par.units)
        sigma_df = self.toasigma_derivs.get(param, None)
        if sigma_df is None:
            if param not in self.params:  # Maybe add differentiable params
                raise AttributeError(f"Parameter {param} does not exist")
            else:
                return result

        for df in sigma_df:
            result += df(data, param).to(
                result.unit, equivalencies=u.dimensionless_angles()
            )
        return result

    def designmatrix(self, toas, acc_delay=None, incfrozen=False, incoffset=True):
        """Return the design matrix.

        The design matrix is the matrix with columns of ``d_phase_d_param/F0``
        or ``d_toa_d_param``; it is used in fitting and calculating parameter
        covariances.

        The value of ``F0`` used here is the parameter value in the model.

        The order of parameters that are included is that returned by
        ``self.params``.

        Parameters
        ----------
        toas : pint.toa.TOAs
            The TOAs at which to compute the design matrix.
        acc_delay
            ???
        incfrozen : bool
            Whether to include frozen parameters in the design matrix
        incoffset : bool
            Whether to include the constant offset in the design matrix
            This option is ignored if a `PhaseOffset` component is present.

        Returns
        -------
        M : array
            The design matrix, with shape (len(toas), len(self.free_params)+1)
        names : list of str
            The names of parameters in the corresponding parts of the design matrix
        units : astropy.units.Unit
            The units of the corresponding parts of the design matrix

        Notes
        -----
        1. We have negative sign here. Since the residuals are calculated as
        (Phase - int(Phase)) in pulsar timing, which is different from the conventional
        definition of least square definition (Data - model), we have decided to add
        a minus sign here in the design matrix so that the fitter keeps the conventional
        sign.

        2. Design matrix entries can be computed only for parameters for which the
        derivatives are implemented. If a parameter without a derivative is unfrozen
        while calling this method, it will raise an informative error, except in the
        case of unfrozen noise parameters, which are simply ignored.
        """

        noise_params = self.get_params_of_component_type("NoiseComponent")

        if (
            not set(self.free_params)
            .difference(noise_params)
            .issubset(self.fittable_params)
        ):
            free_unfittable_params = (
                set(self.free_params)
                .difference(noise_params)
                .difference(self.fittable_params)
            )
            raise ValueError(
                f"Cannot compute the design matrix because the following unfittable parameters "
                f"were found unfrozen in the model: {free_unfittable_params}. "
                f"Freeze these parameters before computing the design matrix."
            )

        # unfrozen_noise_params = [
        #     param for param in noise_params if not getattr(self, param).frozen
        # ]

        # The entries for any unfrozen noise parameters will not be
        # included in the design matrix as they are not well-defined.

        incoffset = incoffset and "PhaseOffset" not in self.components

        params = ["Offset"] if incoffset else []
        params += [
            par
            for par in self.params
            if (incfrozen or not getattr(self, par).frozen) and par not in noise_params
        ]

        F0 = self.F0.quantity  # 1/sec
        ntoas = len(toas)
        nparams = len(params)
        delay = self.delay(toas)
        units = []
        # Apply all delays ?
        # tt = toas['tdbld']
        # for df in self.delay_funcs:
        #    tt -= df(toas)

        M = np.zeros((ntoas, nparams))
        for ii, param in enumerate(params):
            if param == "Offset":
                M[:, ii] = 1.0 / F0.value
                units.append(u.s / u.s)
            else:
                q = -self.d_phase_d_param(toas, delay, param)
                the_unit = u.Unit("") / getattr(self, param).units
                M[:, ii] = q.to_value(the_unit) / F0.value
                units.append(the_unit / F0.unit)

        return M, params, units

    def compare(
        self,
        othermodel,
        nodmx=True,
        convertcoordinates=True,
        threshold_sigma=3.0,
        unc_rat_threshold=1.05,
        verbosity="max",
        usecolor=True,
        format="text",
    ):
        """Print comparison with another model

        Parameters
        ----------
        othermodel
            TimingModel object to compare to
        nodmx : bool, optional
            If True, don't print the DMX parameters in
            the comparison
        convertcoordinates : bool, optional
            Convert coordinates from ICRS<->ECL to make models consistent
        threshold_sigma : float, optional
            Pulsar parameters for which diff_sigma > threshold will be printed
            with an exclamation point at the end of the line
        unc_rat_threshold : float, optional
            Pulsar parameters for which the uncertainty has increased by a
            factor of unc_rat_threshold will be printed with an asterisk at
            the end of the line
        verbosity : string, optional
            Dictates amount of information returned. Options include "max",
            "med", and "min", which have the following results:

                "max"     - print all lines from both models whether they are fit or not (note that nodmx will override this); DEFAULT
                "med"     - only print lines for parameters that are fit
                "min"     - only print lines for fit parameters for which diff_sigma > threshold
                "check"   - only print significant changes with logging.warning, not as string (note that all other modes will still print this)
        usecolor : bool, optional
            Use colors on the output to complement use of "!" and "*"
        format : string, optional
            One of "text" or "markdown"

        Returns
        -------
        str
            Human readable comparison, for printing.
            Formatted as a five column table with titles of
            ``PARAMETER NAME | Model1 | Model2 | Diff_Sigma1 | Diff_Sigma2``
            where ``ModelX`` refer to self and othermodel Timing Model objects,
            and ``Diff_SigmaX`` is the difference in a given parameter as reported by the two models,
            normalized by the uncertainty in model X. If model X has no reported uncertainty,
            nothing will be printed.

            If ``format="text"``, when either ``Diff_SigmaX`` value is greater than ``threshold_sigma``,
            an exclamation point (``!``) will be appended to the line and color will be added if ``usecolor=True``. If the uncertainty in the first model
            if smaller than the second, an asterisk (``*``) will be appended to the line and color will be added if ``usecolor=True``.

            If ``format="markdown"`` then will be formatted as a markdown table with bold, colored, and highlighted text as appropriate.

            For both output formats, warnings and info statements will be printed.

        Note
        ----
            Prints logging warnings for parameters that have changed significantly
            and/or have increased in uncertainty.

        Examples
        --------
        To use this in a Jupyter notebook with and without markdown::

            >>> from pint.models import get_model
            >>> import pint.logging
            >>> from IPython.display import display_markdown
            >>> pint.logging.setup(level="WARNING")
            >>> m1 = get_model(<file1>)
            >>> m2 = get_model(<file2>)
            >>> print(m1.compare(m2))
            >>> display_markdown(m1.compare(m2, format="markdown"), raw=True)

        Make sure to use ``raw=True`` to get the markdown output in a notebook.

        """
        assert verbosity.lower() in ["max", "med", "min", "check"]
        verbosity = verbosity.lower()
        assert format.lower() in ["text", "markdown"]
        format = format.lower()

        if self.name != "":
            model_name = self.name.split("/")[-1]
        else:
            model_name = "Model 1"
        if othermodel.name != "":
            other_model_name = othermodel.name.split("/")[-1]
        else:
            other_model_name = "Model 2"

        # 5 columns of the output, + a way to keep track of values/uncertainties that have changed a lot
        parameter = {}
        value1 = {}
        value2 = {}
        diff1 = {}
        diff2 = {}
        modifier = {}
        parameter["TITLE"] = "PARAMETER"
        value1["TITLE"] = model_name
        value2["TITLE"] = other_model_name
        diff1["TITLE"] = "Diff_Sigma1"
        diff2["TITLE"] = "Diff_Sigma2"
        modifier["TITLE"] = []
        log.info("Comparing ephemerides for PSR %s" % self.PSR.value)
        log.debug("Threshold sigma = %2.2f" % threshold_sigma)
        log.debug("Threshold uncertainty ratio = %2.2f" % unc_rat_threshold)
        log.debug("Creating a copy of model from %s" % other_model_name)
        if verbosity == "max":
            log.debug("Maximum verbosity - printing all parameters")
        elif verbosity == "med":
            log.debug("Medium verbosity - printing parameters that are fit")
        elif verbosity == "min":
            log.debug(
                "Minimum verbosity - printing parameters that are fit and significantly changed"
            )
        elif verbosity == "check":
            log.debug("Check verbosity - only warnings/info will be displayed")
        othermodel = copy.deepcopy(othermodel)

        if (
            "POSEPOCH" in self.params
            and "POSEPOCH" in othermodel.params
            and self.POSEPOCH.value is not None
            and othermodel.POSEPOCH.value is not None
            and self.POSEPOCH.value != othermodel.POSEPOCH.value
        ):
            log.info(
                "Updating POSEPOCH in %s to match %s" % (other_model_name, model_name)
            )
            othermodel.change_posepoch(self.POSEPOCH.value)

        if (
            "PEPOCH" in self.params
            and "PEPOCH" in othermodel.params
            and self.PEPOCH.value is not None
            and self.PEPOCH.value != othermodel.PEPOCH.value
        ):
            log.info(
                "Updating PEPOCH in %s to match %s" % (other_model_name, model_name)
            )
            othermodel.change_pepoch(self.PEPOCH.value)

        if (
            "DMEPOCH" in self.params
            and "DMEPOCH" in othermodel.params
            and self.DMEPOCH.value is not None
            and self.DMEPOCH.value != othermodel.DMEPOCH.value
        ):
            log.info(
                "Updating DMEPOCH in %s to match %s" % (other_model_name, model_name)
            )
            othermodel.change_dmepoch(self.DMEPOCH.value)

        if (
            self.BINARY.value is not None
            and othermodel.BINARY.value is not None
            and self.BINARY.value == othermodel.BINARY.value
        ):
            log.info(
                "Updating binary epoch (T0 or TASC) in %s to match %s"
                % (other_model_name, model_name)
            )
            if (
                "T0" in self
                and "T0" in othermodel
                and self.T0.value is not None
                and othermodel.T0.value is not None
                and self.T0.value != othermodel.T0.value
            ):
                othermodel.change_binary_epoch(self.T0.quantity)
            elif (
                "TASC" in self
                and "TASC" in othermodel
                and self.TASC.value is not None
                and othermodel.TASC.value is not None
                and self.TASC.value != othermodel.TASC.value
            ):
                othermodel.change_binary_epoch(self.TASC.quantity)

        if (
            "AstrometryEquatorial" in self.components
            and "AstrometryEcliptic" in othermodel.components
        ):
            if convertcoordinates:
                log.warning(f"Converting {other_model_name} from ECL to ICRS")
                othermodel = othermodel.as_ICRS()
            else:
                log.warning(
                    f"{model_name} is in ICRS coordinates but {other_model_name} is in ECL coordinates and convertcoordinates=False"
                )
        elif (
            "AstrometryEcliptic" in self.components
            and "AstrometryEquatorial" in othermodel.components
        ):
            if convertcoordinates:
                log.warning(
                    f"Converting {other_model_name} from ICRS to ECL({self.ECL.value})"
                )
                othermodel = othermodel.as_ECL(ecl=self.ECL.value)
            else:
                log.warning(
                    f"{model_name} is in ECL({self.ECL.value}) coordinates but {other_model_name} is in ICRS coordinates and convertcoordinates=False"
                )

        for pn in self.params:
            par = getattr(self, pn)
            if par.value is None:
                continue
            try:
                otherpar = getattr(othermodel, pn)
            except AttributeError:
                otherpar = None
            if isinstance(par, strParameter):
                parameter[pn] = str(pn)
                value1[pn] = str(par.value)
                if otherpar is not None and otherpar.value is not None:
                    value2[pn] = str(otherpar.value)
                else:
                    value2[pn] = "Missing"
                diff1[pn] = ""
                diff2[pn] = ""
                modifier[pn] = []
            elif isinstance(par, AngleParameter):
                if par.frozen:
                    # If not fitted, just print both values
                    parameter[pn] = str(pn)
                    value1[pn] = str(par.quantity)
                    if otherpar is not None and otherpar.quantity is not None:
                        value2[pn] = str(otherpar.quantity)
                        if otherpar.quantity != par.quantity:
                            log.info(
                                "Parameter %s not fit, but has changed between these models"
                                % par.name
                            )
                    else:
                        value2[pn] = "Missing"
                    diff1[pn] = ""
                    diff2[pn] = ""
                    modifier[pn] = []
                else:
                    # If fitted, print both values with uncertainties
                    if par.units == u.hourangle:
                        uncertainty_unit = pint.hourangle_second
                    else:
                        uncertainty_unit = u.arcsec
                    parameter[pn] = pn
                    modifier[pn] = []
                    value1[pn] = "{:>16s} +/- {:7.2g}".format(
                        str(par.quantity),
                        par.uncertainty.to_value(uncertainty_unit),
                    )

                    if otherpar is not None:
                        if otherpar.uncertainty is not None:
                            value2[pn] = "{:>16s} +/- {:7.2g}".format(
                                str(otherpar.quantity),
                                otherpar.uncertainty.to_value(uncertainty_unit),
                            )
                        else:
                            # otherpar must have no uncertainty
                            if otherpar.quantity is not None:
                                value2[pn] = "{:>s}".format(str(otherpar.quantity))
                            else:
                                value2[pn] = "Missing"
                    else:
                        value2[pn] = "Missing"
                        diff1[pn] = ""
                        diff2[pn] = ""
                    if otherpar is not None and otherpar.quantity is not None:
                        diff = otherpar.quantity - par.quantity
                        if par.uncertainty is not None:
                            diff_sigma = (diff / par.uncertainty).decompose()
                        else:
                            diff_sigma = np.inf
                        if abs(diff_sigma) != np.inf:
                            diff1[pn] = "{:>10.2f}".format(diff_sigma)
                            if abs(diff_sigma) > threshold_sigma:
                                modifier[pn].append("diff1")
                        else:
                            diff1[pn] = ""
                        if otherpar.uncertainty is not None:
                            diff_sigma2 = (diff / otherpar.uncertainty).decompose()
                        else:
                            diff_sigma2 = np.inf
                        if abs(diff_sigma2) != np.inf:
                            diff2[pn] = "{:>10.2f}".format(diff_sigma2)
                            if abs(diff_sigma2) > threshold_sigma:
                                modifier[pn].append("diff2")
                        else:
                            diff2[pn] = ""
                    if (
                        otherpar is not None
                        and par.uncertainty is not None
                        and otherpar.uncertainty is not None
                        and (unc_rat_threshold * par.uncertainty < otherpar.uncertainty)
                    ):
                        modifier[pn].append("unc_rat")
            else:
                # Assume numerical parameter
                if nodmx and pn.startswith("DMX"):
                    continue
                parameter[pn] = str(pn)
                modifier[pn] = []
                if par.frozen:
                    # If not fitted, just print both values
                    value1[pn] = str(par.value)
                    diff1[pn] = ""
                    diff2[pn] = ""
                    if otherpar is not None and otherpar.value is not None:
                        if otherpar.uncertainty is not None:
                            value2[pn] = "{:SP}".format(
                                ufloat(otherpar.value, otherpar.uncertainty.value)
                            )
                        else:
                            value2[pn] = str(otherpar.value)
                        if otherpar.value != par.value:
                            if par.name in ["START", "FINISH", "CHI2", "CHI2R", "NTOA"]:
                                if verbosity == "max":
                                    log.info(
                                        "Parameter %s has changed between these models"
                                        % par.name
                                    )
                            elif isinstance(par, boolParameter):
                                if otherpar.value is True:
                                    status = "ON"
                                else:
                                    status = "OFF"
                                log.info(
                                    "Parameter %s has changed between these models (turned %s in %s)"
                                    % (par.name, status, other_model_name)
                                )
                            else:
                                log.warning(
                                    "Parameter %s not fit, but has changed between these models"
                                    % par.name
                                )
                                modifier[pn].append("change")
                        if (
                            par.uncertainty is not None
                            and otherpar.uncertainty is not None
                            and (
                                par.uncertainty * unc_rat_threshold
                                < otherpar.uncertainty
                            )
                        ):
                            modifier[pn].append("unc_rat")
                    else:
                        value2[pn] = "Missing"
                else:
                    # If fitted, print both values with uncertainties
                    if par.uncertainty is not None:
                        value1[pn] = "{:SP}".format(
                            ufloat(par.value, par.uncertainty.value)
                        )
                    else:
                        value1[pn] = str(par.value)
                    if otherpar is not None and otherpar.value is not None:
                        if otherpar.uncertainty is not None:
                            value2[pn] = "{:SP}".format(
                                ufloat(otherpar.value, otherpar.uncertainty.value)
                            )
                        else:
                            # otherpar must have no uncertainty
                            if otherpar.value is not None:
                                value2[pn] = str(otherpar.value)
                            else:
                                value2[pn] = "Missing"
                    else:
                        value2[pn] = "Missing"
                    if value2[pn] == "Missing":
                        log.info(
                            "Parameter %s missing from %s"
                            % (par.name, other_model_name)
                        )
                    if value1[pn] == "Missing":
                        log.info(
                            "Parameter %s missing from %s" % (par.name, model_name)
                        )

                    if otherpar is not None and otherpar.value is not None:
                        diff = otherpar.value - par.value
                        diff_sigma = diff / par.uncertainty.value
                        if abs(diff_sigma) != np.inf:
                            diff1[pn] = "{:>10.2f}".format(diff_sigma)
                            if abs(diff_sigma) > threshold_sigma:
                                modifier[pn].append("diff1")
                        else:
                            diff1[pn] = ""
                        if otherpar.uncertainty is not None:
                            diff_sigma2 = diff / otherpar.uncertainty.value
                            if abs(diff_sigma2) != np.inf:
                                diff2[pn] = "{:>10.2f}".format(diff_sigma2)
                                if abs(diff_sigma2) > threshold_sigma:
                                    modifier[pn].append("diff2")
                            else:
                                diff2[pn] = ""
                        else:
                            diff2[pn] = ""
                        if (
                            par.uncertainty is not None
                            and otherpar.uncertainty is not None
                            and (
                                par.uncertainty * unc_rat_threshold
                                < otherpar.uncertainty
                            )
                        ):
                            modifier[pn].append("unc_rat")
                    else:
                        diff1[pn] = ""
                        diff2[pn] = ""
            if "diff1" in modifier[pn] and not par.frozen:
                log.warning(
                    "Parameter %s has changed significantly (%s sigma1)"
                    % (parameter[pn], diff1[pn])
                )
            if "diff2" in modifier[pn] and not par.frozen:
                log.warning(
                    "Parameter %s has changed significantly (%s sigma2)"
                    % (parameter[pn], diff2[pn])
                )

            if "unc_rat" in modifier[pn]:
                log.warning(
                    "Uncertainty on parameter %s has increased (unc2/unc1 = %2.2f)"
                    % (parameter[pn], float(otherpar.uncertainty / par.uncertainty))
                )

        # Now print any parameters in othermodel that were missing in self.
        mypn = self.params
        for opn in othermodel.params:
            if opn in mypn and getattr(self, opn).value is not None:
                continue
            if nodmx and opn.startswith("DMX"):
                continue
            try:
                otherpar = getattr(othermodel, opn)
            except AttributeError:
                otherpar = None
            if otherpar.value is None:
                continue
            log.info("Parameter %s missing from %s" % (opn, model_name))
            if verbosity == "max":
                parameter[opn] = str(opn)
                value1[opn] = "Missing"
                value2[opn] = str(otherpar.quantity)
                diff1[opn] = ""
                diff2[opn] = ""
                modifier[opn] = []
        separation = self.get_psr_coords().separation(othermodel.get_psr_coords())
        pn = "SEPARATION"
        parameter[pn] = "SEPARATION"
        if separation < 60 * u.arcsec:
            value1[pn] = "{:>f} arcsec".format(separation.arcsec)
        elif separation < 60 * u.arcmin:
            value1[pn] = "{:>f} arcmin".format(separation.arcmin)
        else:
            value1[pn] = "{:>f} deg".format(separation.deg)
        value2[pn] = ""
        diff1[pn] = ""
        diff2[pn] = ""
        modifier[pn] = []
        s = []
        pad = 2
        longest_parameter = len(max(parameter.values(), key=len))
        longest_value1 = len(max(value1.values(), key=len))
        longest_value2 = len(max(value2.values(), key=len))
        longest_diff1 = len(max(diff1.values(), key=len))
        longest_diff2 = len(max(diff2.values(), key=len))
        param = "TITLE"
        if format == "text":
            p = parameter[param]
            v1 = value1[param]
            v2 = value2[param]
            d1 = diff1[param]
            d2 = diff2[param]
            s.append(
                f"{p:<{longest_parameter+pad}} {v1:>{longest_value1+pad}} {v2:>{longest_value2+pad}} {d1:>{longest_diff1+pad}} {d2:>{longest_diff2+pad}}"
            )
            p = "-" * longest_parameter
            v1 = "-" * longest_value1
            v2 = "-" * longest_value2
            d1 = "-" * longest_diff1
            d2 = "-" * longest_diff2
            s.append(
                f"{p:<{longest_parameter+pad}} {v1:>{longest_value1+pad}} {v2:>{longest_value2+pad}} {d1:>{longest_diff1+pad}} {d2:>{longest_diff2+pad}}"
            )
        elif format == "markdown":
            p = parameter[param]
            v1 = value1[param]
            v2 = value2[param]
            d1 = diff1[param]
            d2 = diff2[param]
            s.append(f"| {p} | {v1} | {v2} | {d1} | {d2} |")
            s.append(f" :--- | ---: | ---: | ---: | ---: |")
        for param in parameter:
            if param == "TITLE":
                continue
            p = parameter[param]
            v1 = value1[param]
            v2 = value2[param]
            d1 = diff1[param]
            d2 = diff2[param]
            m = modifier[param]
            if format == "text":
                sout = f"{p:<{longest_parameter+pad}} {v1:>{longest_value1+pad}} {v2:>{longest_value2+pad}} {d1:>{longest_diff1+pad}} {d2:>{longest_diff2+pad}}"
                if "change" in m or "diff1" in m or "diff2" in m:
                    sout += " !"
                if "unc_rat" in m:
                    sout += " *"
                if usecolor:
                    if (
                        "change" in m
                        or "diff1" in m
                        or "diff2" in m
                        and not "unc_rat" in m
                    ):
                        sout = colorize(sout, "red")
                    elif (
                        "change" in m or "diff1" in m or "diff2" in m and "unc_rat" in m
                    ):
                        sout = colorize(sout, "red", bg_color="green")
                    elif "unc_rat" in m:
                        sout = colorize(sout, bg_color="green")
            elif format == "markdown":
                sout = [p.strip(), v1.strip(), v2.strip(), d1.strip(), d2.strip()]
                if "change" in m or "diff1" in m or "diff2" in m:
                    sout = [
                        f"<span style='color:red'>**{x}**</span>" if len(x) > 0 else x
                        for x in sout
                    ]
                if "unc_rat" in m:
                    sout = [f"<mark>{x}</mark>" if len(x) > 0 else x for x in sout]
                sout = " | ".join(sout).strip()
            if verbosity == "max":
                s.append(sout)
            elif verbosity == "med" and len(d1) > 0:
                # not frozen so has uncertainty
                s.append(sout)
            elif verbosity == "min" and len(m) > 0:
                # has a modifier
                s.append(sout)

        if verbosity != "check":
            return "\n".join(s)

    def use_aliases(self, reset_to_default=True, alias_translation=None):
        """Set the parameters to use aliases as specified upon writing.

        Parameters
        ----------
        reset_to_default : bool
            If True, forget what name was used for each parameter in the input par file.
        alias_translation : dict or None
            If not None, use this to map PINT parameter names to output names. This overrides
            input names even if they are not otherwise being reset to default.
            This is to allow compatibility with TEMPO/TEMPO2. The dictionary
            ``pint.toa.tempo_aliases`` should provide a reasonable selection.
        """
        for p in self.params:
            po = getattr(self, p)
            if reset_to_default:
                po.use_alias = None
            if alias_translation is not None:
                if hasattr(po, "origin_name"):
                    try:
                        po.use_alias = alias_translation[po.origin_name]
                    except KeyError:
                        pass
                else:
                    try:
                        po.use_alias = alias_translation[p]
                    except KeyError:
                        pass

    def as_parfile(
        self,
        start_order=["astrometry", "spindown", "dispersion"],
        last_order=["jump_delay"],
        *,
        include_info=True,
        comment=None,
        format="pint",
    ):
        """Represent the entire model as a parfile string.

        See also :func:`pint.models.TimingModel.write_parfile`.

        Parameters
        ----------
        start_order : list
            Categories to include at the beginning
        last_order : list
            Categories to include at the end
        include_info : bool, optional
            Include information string if True
        comment : str, optional
            Additional comment string to include in parfile
        format : str, optional
             Parfile output format. PINT outputs in 'tempo', 'tempo2' and 'pint'
             formats. The defaul format is `pint`.
        """
        if not format.lower() in _parfile_formats:
            raise ValueError(f"parfile format must be one of {_parfile_formats}")

        self.validate()
        if include_info:
            info_string = pint.utils.info_string(prefix_string="# ", comment=comment)
            info_string += f"\n# Format: {format.lower()}"
            result_begin = info_string + "\n"
        else:
            result_begin = ""
        result_end = ""
        result_middle = ""
        cates_comp = self.get_components_by_category()
        printed_cate = []
        # make sure TEMPO2 format start with "MODE 1"
        if format.lower() == "tempo2":
            result_begin += "MODE 1\n"
        for p in self.top_level_params:
            if p == "BINARY":  # Will print the Binary model name in the binary section
                continue
            result_begin += getattr(self, p).as_parfile_line(format=format)
        for cat in start_order:
            if cat in list(cates_comp.keys()):
                # print("Starting: %s" % cat)
                cp = cates_comp[cat]
                for cpp in cp:
                    result_begin += cpp.print_par(format=format)
                printed_cate.append(cat)
            else:
                continue

        for cat in last_order:
            if cat in list(cates_comp.keys()):
                # print("Ending: %s" % cat)
                cp = cates_comp[cat]
                for cpp in cp:
                    result_end += cpp.print_par(format=format)
                printed_cate.append(cat)
            else:
                continue

        for cat in list(cates_comp.keys()):
            if cat in printed_cate:
                continue
            else:
                cp = cates_comp[cat]
                for cpp in cp:
                    result_middle += cpp.print_par(format=format)
                printed_cate.append(cat)

        return result_begin + result_middle + result_end

    def write_parfile(
        self,
        filename,
        start_order=["astrometry", "spindown", "dispersion"],
        last_order=["jump_delay"],
        *,
        include_info=True,
        comment=None,
        format="pint",
    ):
        """Write the entire model as a parfile.

        See also :func:`pint.models.TimingModel.as_parfile`.

        Parameters
        ----------
        filename : str or Path or file-like
            The destination to write the parfile to
        start_order : list
            Categories to include at the beginning
        last_order : list
            Categories to include at the end
        include_info : bool, optional
            Include information string if True
        comment : str, optional
            Additional comment string to include in parfile
        format : str, optional
             Parfile output format. PINT outputs in 'tempo', 'tempo2' and 'pint'
             formats. The defaul format is `pint`.
        """
        with open_or_use(filename, "wt") as f:
            f.write(
                self.as_parfile(
                    start_order=start_order,
                    last_order=last_order,
                    include_info=include_info,
                    comment=comment,
                    format=format,
                )
            )

    def validate_toas(self, toas):
        """Sanity check to verify that this model is compatible with these toas.

        This checks that where this model needs TOAs to constrain parameters,
        that there is at least one TOA. This includes checking that every DMX
        range for with the DMX is free has at least one TOA, and it verifies
        that each "mask parameter" (for example JUMP) corresponds to at least one
        TOA.

        Individual components can implement a ``validate_toas`` method; this
        method will automatically call such a method on each component that has
        one.

        If some TOAs are missing, this method will raise a MissingTOAError that
        lists some (at least one) of the problem parameters.
        """
        bad_parameters = []
        for maskpar in self.get_params_of_type_top("maskParameter"):
            par = getattr(self, maskpar)
            if par.frozen:
                continue
            if len(par.select_toa_mask(toas)) == 0:
                bad_parameters.append(f"'{maskpar}, {par.key}, {par.key_value}'")
        for c in self.components.values():
            try:
                c.validate_toas(toas)
            except MissingTOAs as e:
                bad_parameters += e.parameter_names
        if bad_parameters:
            raise MissingTOAs(bad_parameters)

    def find_empty_masks(self, toas, freeze=False):
        """Find unfrozen mask parameters with no TOAs before trying to fit

        Parameters
        ----------
        toas : pint.toa.TOAs
        freeze : bool, optional
            Should the parameters with on TOAs be frozen

        Returns
        -------
        list
            Parameters with no TOAs
        """
        bad_parameters = []
        for maskpar in self.get_params_of_type_top("maskParameter"):
            par = getattr(self, maskpar)
            if par.frozen:
                continue
            if len(par.select_toa_mask(toas)) == 0:
                bad_parameters.append(maskpar)
                if freeze:
                    log.info(f"'{maskpar}' has no TOAs so freezing")
                    getattr(self, maskpar).frozen = True
        for prefix in ["DM", "SW"]:
            mapping = pint.utils.xxxselections(self, toas, prefix=prefix)
            for k in mapping:
                if len(mapping[k]) == 0:
                    if freeze:
                        log.info(f"'{k}' has no TOAs so freezing")
                        getattr(self, k).frozen = True
                    bad_parameters.append(k)
        return bad_parameters

    def setup(self):
        """Run setup methods on all components."""
        for cp in self.components.values():
            cp.setup()

    def __contains__(self, name):
        return name in self.params

    def __getitem__(self, name):
        if name in self.top_level_params:
            return getattr(self, name)
        for cp in self.components.values():
            if name in cp.params:
                return getattr(cp, name)
        raise KeyError(f"TimingModel does not have parameter {name}")

    def __setitem__(self, name, value):
        # FIXME: This could be the right way to add Parameters?
        raise NotImplementedError

    def keys(self):
        return self.params

    def items(self):
        return [(p, self[p]) for p in self.params]

    def __len__(self):
        return len(self.params)

    def __iter__(self):
        for p in self.params:
            yield p

    def as_ECL(self, epoch=None, ecl="IERS2010"):
        """Return TimingModel in PulsarEcliptic frame.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed.
            New epoch for position.
        ecl : str, optional
            Obliquity for PulsarEcliptic frame

        Returns
        -------
        pint.models.timing_model.TimingModel
            In PulsarEcliptic frame

        Notes
        -----
        For the ``DDK`` model, the ``KOM`` vector is also transformed

        """
        if "AstrometryEquatorial" in self.components:
            astrometry_model_type = "AstrometryEquatorial"
        elif "AstrometryEcliptic" in self.components:
            astrometry_model_type = "AstrometryEcliptic"
        astrometry_model_component = self.components[astrometry_model_type]
        new_astrometry_model_component = astrometry_model_component.as_ECL(
            epoch=epoch, ecl=ecl
        )
        new_model = copy.deepcopy(self)
        new_model.remove_component(astrometry_model_type)
        new_model.add_component(new_astrometry_model_component)

        if "BinaryDDK" in self.components and "AstrometryEquatorial" in self.components:
            c = coords.SkyCoord(
                lon=new_model.ELONG.quantity,
                lat=new_model.ELAT.quantity,
                obstime=self.POSEPOCH.quantity,
                pm_lon_coslat=np.sin(self.KOM.quantity) * u.mas / u.yr,
                pm_lat=np.cos(self.KOM.quantity) * u.mas / u.yr,
                obliquity=OBL[new_model.ECL.value],
                frame=PulsarEcliptic,
            )
            c_ICRS = c.transform_to(coords.ICRS)
            new_model.KOM.quantity = (
                np.arctan2(c_ICRS.pm_ra_cosdec.value, c_ICRS.pm_dec.value) * u.rad
            ).to(self.KOM.units)

        return new_model

    def as_ICRS(self, epoch=None):
        """Return TimingModel in ICRS frame.

        Parameters
        ----------
        epoch : float or astropy.time.Time or astropy.units.Quantity, optional
            If float or Quantity, MJD(TDB) is assumed.
            New epoch for position.

        Returns
        -------
        pint.models.timing_model.TimingModel
            In ICRS frame

        Notes
        -----
        For the ``DDK`` model, the ``KOM`` vector is also transformed
        """
        if "AstrometryEquatorial" in self.components:
            astrometry_model_type = "AstrometryEquatorial"
        elif "AstrometryEcliptic" in self.components:
            astrometry_model_type = "AstrometryEcliptic"
        astrometry_model_component = self.components[astrometry_model_type]
        new_astrometry_model_component = astrometry_model_component.as_ICRS(epoch=epoch)
        new_model = copy.deepcopy(self)
        new_model.remove_component(astrometry_model_type)
        new_model.add_component(new_astrometry_model_component)

        if "BinaryDDK" in self.components and "AstrometryEcliptic" in self.components:
            c = coords.SkyCoord(
                ra=new_model.RAJ.quantity,
                dec=new_model.DECJ.quantity,
                obstime=self.POSEPOCH.quantity,
                pm_ra_cosdec=np.sin(self.KOM.quantity) * u.mas / u.yr,
                pm_dec=np.cos(self.KOM.quantity) * u.mas / u.yr,
                frame=coords.ICRS,
            )
            c_ECL = c.transform_to(PulsarEcliptic(ecl=self.ECL.value))
            new_model.KOM.quantity = (
                np.arctan2(c_ECL.pm_lon_coslat.value, c_ECL.pm_lat.value) * u.rad
            ).to(self.KOM.units)

        return new_model

    def get_derived_params(self, rms=None, ntoas=None, returndict=False):
        """Return a string with various derived parameters from the fitted model

        Parameters
        ----------
        rms : astropy.units.Quantity, optional
            RMS of fit for checking ELL1 validity
        ntoas : int, optional
            Number of TOAs for checking ELL1 validity
        returndict : bool, optional
            Whether to only return the string of results or also a dictionary

        Returns
        -------
        results : str
        parameters : dict, optional
        """

        import uncertainties.umath as um
        from uncertainties import ufloat

        outdict = {}

        # Now print some useful derived parameters
        s = "Derived Parameters:\n"
        if hasattr(self, "F0"):
            F0 = self.F0.as_ufloat()
            p = 1 / F0
            s += f"Period = {p:P} s\n"
            outdict["P (s)"] = p
        if hasattr(self, "F1"):
            F1 = self.F1.as_ufloat()
            pdot = -F1 / F0**2
            outdict["Pdot (s/s)"] = pdot
            s += f"Pdot = {pdot:P}\n"
            if self.F1.value < 0.0:  # spinning-down
                brakingindex = 3
                s += f"Characteristic age = {pint.derived_quantities.pulsar_age(self.F0.quantity, self.F1.quantity, n=brakingindex):.4g} (braking index = {brakingindex})\n"
                s += f"Surface magnetic field = {pint.derived_quantities.pulsar_B(self.F0.quantity, self.F1.quantity):.3g}\n"
                s += f"Magnetic field at light cylinder = {pint.derived_quantities.pulsar_B_lightcyl(self.F0.quantity, self.F1.quantity):.4g}\n"
                I_NS = I = 1.0e45 * u.g * u.cm**2
                s += f"Spindown Edot = {pint.derived_quantities.pulsar_edot(self.F0.quantity, self.F1.quantity, I=I_NS):.4g} (I={I_NS})\n"
                outdict["age"] = pint.derived_quantities.pulsar_age(
                    self.F0.quantity, self.F1.quantity, n=brakingindex
                )
                outdict["B"] = pint.derived_quantities.pulsar_B(
                    self.F0.quantity, self.F1.quantity
                )
                outdict["Blc"] = pint.derived_quantities.pulsar_B_lightcyl(
                    self.F0.quantity, self.F1.quantity
                )
                outdict["Edot"] = pint.derived_quantities.pulsar_B_lightcyl(
                    self.F0.quantity, self.F1.quantity
                )
            else:
                s += "Not computing Age, B, or Edot since F1 > 0.0\n"

        if hasattr(self, "PX") and not self.PX.frozen:
            s += "\n"
            px = self.PX.as_ufloat(u.arcsec)
            s += f"Parallax distance = {1.0/px:.3uP} pc\n"
            outdict["Dist (pc)"] = 1.0 / px
        # Now binary system derived parameters
        if self.is_binary:
            for x in self.components:
                if x.startswith("Binary"):
                    binary = x

            s += f"\nBinary model {binary}\n"
            outdict["Binary"] = binary

            btx = False
            if (
                hasattr(self, "FB0")
                and self.FB0.quantity is not None
                and self.FB0.value != 0.0
            ):
                btx = True
                pb = 1 / self.FB0.as_ufloat(1 / u.d)
                s += f"Orbital Period  (PB) = {pb:P} (d)\n"
            else:
                pb = self.PB.as_ufloat(u.d)
            outdict["PB (d)"] = pb

            pbdot = None
            if (
                hasattr(self, "FB1")
                and self.FB1.quantity is not None
                and self.FB1.value != 0.0
            ):
                pbdot = -self.FB1.as_ufloat(u.Hz / u.s) / self.FB0.as_ufloat(u.Hz) ** 2
            elif (
                hasattr(self, "PBDOT")
                and self.PBDOT.quantity is not None
                and self.PBDOT.value != 0
            ):
                pbdot = self.PBDOT.as_ufloat(u.s / u.s)

            if pbdot is not None:
                s += f"Orbital Pdot (PBDOT) = {pbdot:P} (s/s)\n"
                outdict["PBDOT (s/s)"] = pbdot

            ell1 = False
            if binary.startswith("BinaryELL1"):
                ell1 = True
                eps1 = self.EPS1.as_ufloat()
                eps2 = self.EPS2.as_ufloat()
                tasc = ufloat(
                    # This is a time in MJD
                    self.TASC.quantity.mjd,
                    (
                        self.TASC.uncertainty.to(u.d).value
                        if self.TASC.uncertainty is not None
                        else 0
                    ),
                )
                s += "Conversion from ELL1 parameters:\n"
                ecc = um.sqrt(eps1**2 + eps2**2)
                s += "ECC = {:P}\n".format(ecc)
                outdict["ECC"] = ecc
                om = um.atan2(eps1, eps2) * 180.0 / np.pi
                if om < 0.0:
                    om += 360.0
                s += f"OM  = {om:P} deg\n"
                outdict["OM (deg)"] = om
                t0 = tasc + pb * om / 360.0
                s += f"T0  = {t0:SP}\n"
                outdict["T0"] = t0

                a1 = self.A1.quantity if self.A1.quantity is not None else 0 * pint.ls
                if rms is not None and ntoas is not None:
                    s += pint.utils.ELL1_check(
                        a1,
                        ecc.nominal_value * u.s / u.s,
                        rms,
                        ntoas,
                        outstring=True,
                    )
                s += "\n"
            # Masses and inclination
            if not self.A1.frozen:
                a1 = self.A1.as_ufloat(pint.ls)
                # This is the mass function, done explicitly so that we get
                # uncertainty propagation automatically.
                # TODO: derived quantities funcs should take uncertainties
                fm = 4.0 * np.pi**2 * a1**3 / (4.925490947e-6 * (pb * 86400) ** 2)
                s += f"Mass function = {fm:SP} Msun\n"
                outdict["Mass Function (Msun)"] = fm
                mcmed = pint.derived_quantities.companion_mass(
                    pb.n * u.d,
                    self.A1.quantity,
                    i=60.0 * u.deg,
                    mp=1.4 * u.solMass,
                )
                mcmin = pint.derived_quantities.companion_mass(
                    pb.n * u.d,
                    self.A1.quantity,
                    i=90.0 * u.deg,
                    mp=1.4 * u.solMass,
                )
                s += f"Min / Median Companion mass (assuming Mpsr = 1.4 Msun) = {mcmin.value:.4f} / {mcmed.value:.4f} Msun\n"
                outdict["Mc,med (Msun)"] = mcmed.value
                outdict["Mc,min (Msun)"] = mcmin.value

            if (
                hasattr(self, "OMDOT")
                and self.OMDOT.quantity is not None
                and self.OMDOT.value != 0.0
            ):
                omdot = self.OMDOT.as_ufloat(u.rad / u.s)
                e = ecc if ell1 else self.ECC.as_ufloat()
                mt = (
                    (
                        omdot
                        / (
                            3
                            * (c.G * u.Msun / c.c**3).to_value(u.s) ** (2.0 / 3)
                            * ((pb * 86400 / 2 / np.pi)) ** (-5.0 / 3)
                            * (1 - e**2) ** -1
                        )
                    )
                ) ** (3.0 / 2)
                s += f"Total mass, assuming GR, from OMDOT is {mt:SP} Msun\n"
                outdict["Mtot (Msun)"] = mt

            if (
                hasattr(self, "SINI")
                and self.SINI.quantity is not None
                and (self.SINI.value >= 0.0 and self.SINI.value < 1.0)
            ):
                with contextlib.suppress(TypeError, ValueError):
                    # Put this in a try in case SINI is UNSET or an illegal value
                    if not self.SINI.frozen:
                        si = self.SINI.as_ufloat()
                        s += f"From SINI in model:\n"
                        s += f"    cos(i) = {um.sqrt(1 - si**2):SP}\n"
                        s += f"    i = {um.asin(si) * 180.0 / np.pi:SP} deg\n"

                    psrmass = pint.derived_quantities.pulsar_mass(
                        pb.n * u.d,
                        self.A1.quantity,
                        self.M2.quantity,
                        np.arcsin(self.SINI.quantity),
                    )
                    s += f"Pulsar mass (Shapiro Delay) = {psrmass}"
                    outdict["Mp (Msun)"] = psrmass
        if not returndict:
            return s
        return s, outdict


class ModelMeta(abc.ABCMeta):
    """Ensure timing model registration.

    When a new subclass of Component is created, record its identity in
    a class attribute ``component_types``, provided that the class has
    an attribute ``register``. This makes sure all timing model components
    are listed in ``Component.component_types``.

    """

    def __init__(cls, name, bases, dct):
        regname = "component_types"
        if "register" in dct and cls.register:
            getattr(cls, regname)[name] = cls
        super().__init__(name, bases, dct)


class Component(metaclass=ModelMeta):
    """Timing model components.

    When such a class is defined, it registers itself in
    ``Component.component_types`` so that it can be found and used
    when parsing par files.
    Note that classes are registered when their modules are imported,
    so ensure all classes of interest are imported before this list
    is checked.

    These objects can be constructed with no particular values, but
    their `.setup()` and `.validate()` methods should be called
    before using them to compute anything. These should check
    parameter values for validity, raising an exception if
    invalid parameter values are chosen.
    """

    component_types = {}

    def __init__(self):
        self.params = []
        self._parent = None
        self.deriv_funcs = {}
        self.component_special_params = []

    def __repr__(self):
        return "{}(\n    {})".format(
            self.__class__.__name__,
            ",\n    ".join(
                str(getattr(self, p))
                for p in self.params
                if not isinstance(p, funcParameter)
            ),
        )

    def setup(self):
        """Finalize construction loaded values."""
        pass

    def validate(self):
        """Validate loaded values."""
        pass

    def validate_toas(self, toas):
        """Check that this model component has TOAs where needed."""
        pass

    @property_exists
    def category(self):
        """Category is a feature the class, so delegate."""
        return self.__class__.category

    @property_exists
    def free_params_component(self):
        """Return the free parameters in the component.

        This function collects the non-frozen parameters.

        Returns
        -------
        A list of free parameters.
        """
        free_param = []
        for p in self.params:
            par = getattr(self, p)
            if not par.frozen:
                free_param.append(p)
        return free_param

    @property_exists
    def param_prefixs(self):
        prefixs = {}
        for p in self.params:
            par = getattr(self, p)
            if par.is_prefix:
                if par.prefix not in prefixs.keys():
                    prefixs[par.prefix] = [p]
                else:
                    prefixs[par.prefix].append(p)
        return prefixs

    @property_exists
    def aliases_map(self):
        """Return all the aliases and map to the PINT parameter name.

        This property returns a dictionary from the current in timing model
        parameters' aliase to the pint defined parameter names. For the aliases
        of a prefixed parameter, the aliase with an existing prefix index maps
        to the PINT defined parameter name with the same index. Behind the scenes,
        the indexed parameter adds the indexed aliase to its aliase list.
        """
        ali_map = {}
        for p in self.params:
            par = getattr(self, p)
            ali_map[p] = p
            for ali in par.aliases:
                ali_map[ali] = p
        return ali_map

    def add_param(self, param, deriv_func=None, setup=False):
        """Add a parameter to the Component.

        The parameter is stored in an attribute on the Component object.
        Its name is also recorded in a list, ``self.params``.

        Parameters
        ----------
        param : pint.models.Parameter
            The parameter to be added.
        deriv_func: function
            Derivative function for parameter.
        """
        # This is the case for add "JUMP" like parameters, It will add an
        # index to the parameter name for avoding the conflicts
        # TODO: this is a work around in the current system, but it will be
        # optimized in the future release.
        if isinstance(param, maskParameter):
            # TODO, right now maskParameter add index to parameter name by
            # default. But This is should be optimized. In the future versions,
            # it will change.

            # First get prefix and index from input parameter name
            try:
                prefix, idx_str, idx = split_prefixed_name(param.name)
            except PrefixError:
                prefix = param.name
                idx = 1

            # Check existing prefix
            prefix_map = self.get_prefix_mapping_component(prefix)
            exist_par_name = prefix_map.get(idx, None)
            # Check if parameter value has been set.
            if exist_par_name and getattr(self, exist_par_name).value is not None:
                idx = max(list(prefix_map.keys())) + 1

            # TODO here we have an assumption that maskParameter follow the
            # convention of name + no_leading_zero_index
            param.name = prefix + str(idx)
            param.index = idx

            if hasattr(self, f"{prefix}1"):
                param.description = getattr(self, f"{prefix}1").description

        # A more general check
        if param.name in self.params:
            exist_par = getattr(self, param.name)
            if exist_par.value is not None:
                raise ValueError(
                    "Tried to add a second parameter called {}. "
                    "Old value: {} New value: {}".format(
                        param.name, getattr(self, param.name), param
                    )
                )
            else:
                setattr(self, param.name, param)
        else:  # When parameter not in the params list, we also need to add it.
            setattr(self, param.name, param)
            self.params.append(param.name)
        # Adding parameters to an existing model sometimes need to run setup()
        # function again.
        if setup:
            self.setup()
        if deriv_func is not None:
            self.register_deriv_funcs(deriv_func, param.name)
        param._parent = self

    def remove_param(self, param):
        """Remove a parameter from the Component.

        Parameters
        ----------
        param : str or pint.models.Parameter
            The parameter to remove.
        """
        if isinstance(param, str):
            param_name = param
        else:
            param_name = param.name
        if param_name not in self.params:
            raise ValueError(
                f"Tried to remove parameter {param_name} but it is not listed: {self.params}"
            )

        self.params.remove(param_name)
        par = getattr(self, param_name)
        all_names = [param] + par.aliases
        if param in self.component_special_params:
            for pn in all_names:
                self.component_special_params.remove(pn)
        delattr(self, param)

    def set_special_params(self, spcl_params):
        als = []
        for p in spcl_params:
            als += getattr(self, p).aliases
        spcl_params += als
        for sp in spcl_params:
            if sp not in self.component_special_params:
                self.component_special_params.append(sp)

    def param_help(self):
        """Print help lines for all available parameters in model."""
        s = "Available parameters for %s\n" % self.__class__
        for par in self.params:
            s += "%s\n" % getattr(self, par).help_line()
        return s

    def get_params_of_type(self, param_type):
        """Get all the parameters in timing model for one specific type."""
        result = []
        for p in self.params:
            par = getattr(self, p)
            par_type = type(par).__name__
            par_prefix = par_type[:-9]
            if (
                param_type.upper() == par_type.upper()
                or param_type.upper() == par_prefix.upper()
            ):
                result.append(par.name)
        return result

    def get_prefix_mapping_component(self, prefix):
        """Get the index mapping for the prefix parameters.

        Parameters
        ----------
        prefix : str
           Name of prefix.

        Returns
        -------
        dict
           A dictionary with prefix parameter real index as key and parameter
           name as value.

        """
        parnames = [x for x in self.params if x.startswith(prefix)]
        mapping = {}
        for parname in parnames:
            par = getattr(self, parname)
            if par.is_prefix and par.prefix == prefix:
                mapping[par.index] = parname
        return OrderedDict(sorted(mapping.items()))

    def match_param_aliases(self, alias):
        """Return the parameter corresponding to this alias.

        Parameters
        ----------
        alias: str
            Alias name.

        Note
        ----
        This function only searches the parameter aliases within the current
        component. If one wants to search the aliases in the scope of TimingModel,
        please use :py:meth:`TimingModel.match_param_aliase`.
        """
        pname = self.aliases_map.get(alias, None)
        # Split the alias prefix, see if it is a perfix alias
        try:
            prefix, idx_str, idx = split_prefixed_name(alias)
        except PrefixError:  # Not a prefixed name
            if pname is not None:
                par = getattr(self, pname)
                if par.is_prefix:
                    raise UnknownParameter(
                        f"Prefix {alias} maps to mulitple parameters"
                        ". Please specify the index as well."
                    )
            else:
                # Not a prefix, not an alias
                raise UnknownParameter(f"Unknown parameter name or alias {alias}")
        # When the alias is a prefixed name but not in the parameter list yet
        if pname is None:
            prefix_pname = self.aliases_map.get(prefix, None)
            if prefix_pname:
                par = getattr(self, prefix_pname)
                if par.is_prefix:
                    raise UnknownParameter(
                        f"Found a similar prefixed parameter '{prefix_pname}'"
                        f" But parameter {par.prefix}{idx} need to be added"
                        f" to the model."
                    )
                else:
                    raise UnknownParameter(
                        f"{par} is not a prefixed parameter, howere the input"
                        f" {alias} has index with it."
                    )
            else:
                raise UnknownParameter(f"Unknown parameter name or alias {alias}")
        else:
            return pname

    def register_deriv_funcs(self, func, param):
        """Register the derivative function in to the deriv_func dictionaries.

        Parameters
        ----------
        func : callable
            Calculates the derivative
        param : str
            Name of parameter the derivative is with respect to

        """
        pn = self.match_param_aliases(param)

        if pn not in list(self.deriv_funcs.keys()):
            self.deriv_funcs[pn] = [func]
        else:
            # TODO:
            # Runing setup() mulitple times can lead to adding derivative
            # function multiple times. This prevent it from happening now. But
            # in the future, we should think a better way to do so.
            if func in self.deriv_funcs[pn]:
                return
            else:
                self.deriv_funcs[pn] += [func]

    def is_in_parfile(self, para_dict):
        """Check if this subclass included in parfile.

        Parameters
        ----------
        para_dict : dictionary
            A dictionary contain all the parameters with values in string
            from one parfile

        Returns
        -------
        bool
            Whether the subclass is included in the parfile.

        """
        if self.component_special_params:
            for p in self.component_special_params:
                if p in para_dict:
                    return True
            return False

        pNames_inpar = list(para_dict.keys())
        pNames_inModel = self.params

        # FIXME: we have derived classes, this is the sort of thing that
        # should go in them.
        # For solar system Shapiro delay component
        if hasattr(self, "PLANET_SHAPIRO"):
            if "NO_SS_SHAPIRO" in pNames_inpar:
                return False
            else:
                return True

        try:
            bmn = getattr(self, "binary_model_name")
        except AttributeError:
            # This isn't a binary model, keep looking
            pass
        else:
            if "BINARY" in para_dict:
                return bmn == para_dict["BINARY"][0]
            else:
                return False

        # Compare the componets parameter names with par file parameters
        compr = list(set(pNames_inpar).intersection(pNames_inModel))

        if compr == []:
            # Check aliases
            for p in pNames_inModel:
                al = getattr(self, p).aliases
                # No aliases in parameters
                if al == []:
                    continue
                # Find alias check if match any of parameter name in parfile
                if list(set(pNames_inpar).intersection(al)):
                    return True
                else:
                    continue
            # TODO Check prefix parameter
            return False

        return True

    def print_par(self, format="pint"):
        """
        Parameters
        ----------
        format : str, optional
             Parfile output format. PINT outputs the 'tempo', 'tempo2' and 'pint'
             format. The defaul format is `pint`.  Actual formatting done elsewhere.

        Returns
        -------
        str : formatted line for par file
        """
        result = ""
        for p in self.params:
            result += getattr(self, p).as_parfile_line(format=format)
        return result


class DelayComponent(Component):
    def __init__(self):
        super().__init__()
        self.delay_funcs_component = []


class PhaseComponent(Component):
    def __init__(self):
        super().__init__()
        self.phase_funcs_component = []
        self.phase_derivs_wrt_delay = []


class AllComponents:
    """A class for the components pool.

    This object stores and manages the instances of component classes with class
    attribute .register = True. This includes the PINT built-in components and
    user defined components and there is no need to import the component class.
    This class constructs the available component instances, but without any
    valid parameter values (parameters are initialized when a component instance
    gets constructed, however, the parameter values are unknown to the components
    at the moment). Thus, runing `.validate()` function in the component instance
    will fail. This class is designed for helping model building and parameter
    seraching, not for direct data analysis.

    Note
    ----
    This is a low level class for managing all the components. To build a timing
    model, we recommend to use the subclass `models.model_builder.ModelBuilder`,
    where higher level interface are provided. If one wants to use this class
    directly, one has to construct the instance separately.
    """

    def __init__(self):
        self.components = {}
        for k, v in Component.component_types.items():
            self.components[k] = v()

    @lazyproperty
    def param_component_map(self):
        """Return the parameter to component map.

        This property returns the all PINT defined parameters to their host
        components. The parameter aliases are not included in this map. If
        searching the host component for a parameter alias, pleaase use
        `alias_to_pint_param` method to translate the alias to PINT parameter
        name first.
        """
        p2c_map = defaultdict(list)
        for k, cp in self.components.items():
            for p in cp.params:
                p2c_map[p].append(k)
                # Add alias
                par = getattr(cp, p)
                for ap in par.aliases:
                    p2c_map[ap].append(k)
        tm = TimingModel()
        for tp in tm.params:
            p2c_map[tp].append("timing_model")
            par = getattr(tm, tp)
            for ap in par.aliases:
                p2c_map[ap].append("timing_model")
        return p2c_map

    def _check_alias_conflict(self, alias, param_name, alias_map):
        """Check if a aliase has conflict in the alias map.

        This function checks if an alias already have record in the alias_map.
        If there is a record, it will check if the record matches the given
        paramter name, `param_name`. If not match, it will raise a AliasConflict
        error.

        Parameter
        ---------
        alias: str
            The alias name that needs to check if it has entry in the alias_map.
        param_name: str
            The parameter name that a alias is going to be mapped to.

        Raise
        -----
        AliasConflict
            When the input alias has a record in the aliases map, but the record
            does not match the input parameter name that is going to be mapped
            to the input alias.
        """
        if alias in alias_map.keys():
            if param_name == alias_map[alias]:
                return
            else:
                raise AliasConflict(
                    f"Alias {alias} has been used by" f" parameter {param_name}."
                )
        else:
            return

    @lazyproperty
    def _param_alias_map(self):
        """Return the aliases map of all parameters

        The returned map includes: 1. alias to PINT parameter name. 2. PINT
        parameter name to pint parameter name. 3.prefix to PINT parameter name.

        Notes
        -----
        Please use `alias_to_pint_param` method to map an alias to a PINT parameter.
        """
        alias = {}
        for k, cp in self.components.items():
            for p in cp.params:
                par = getattr(cp, p)
                # Check if an existing record
                self._check_alias_conflict(p, p, alias)
                alias[p] = p
                for als in par.aliases:
                    self._check_alias_conflict(als, p, alias)
                    alias[als] = p
        tm = TimingModel()
        for tp in tm.params:
            par = getattr(tm, tp)
            self._check_alias_conflict(tp, tp, alias)
            alias[tp] = tp
            for als in par.aliases:
                self._check_alias_conflict(als, tp, alias)
                alias[als] = tp
        return alias

    @lazyproperty
    def _param_unit_map(self):
        """A dictionary to map parameter names to their units

        This excludes prefix parameters and aliases.  Use :func:`param_to_unit` to handle those.
        """
        units = {}
        for k, cp in self.components.items():
            for p in cp.params:
                if p in units.keys() and units[p] != getattr(cp, p).units:
                    raise TimingModelError(
                        f"Units of parameter '{p}' in component '{cp}' ({getattr(cp, p).units}) do not match those of existing parameter ({units[p]})"
                    )
                units[p] = getattr(cp, p).units
        tm = TimingModel()
        for tp in tm.params:
            units[p] = getattr(tm, tp).units
        return units

    @lazyproperty
    def repeatable_param(self):
        """Return the repeatable parameter map."""
        repeatable = []
        for k, cp in self.components.items():
            for p in cp.params:
                par = getattr(cp, p)
                if par.repeatable:
                    repeatable.append(p)
                    repeatable.append(par._parfile_name)
                    # also add the aliases to the repeatable param
                    for als in par.aliases:
                        repeatable.append(als)
        return set(repeatable)

    @lazyproperty
    def category_component_map(self):
        """A dictionary mapping category to a list of component names.

        Return
        ------
        dict
            The mapping from categories to the componens belongs to the categore.
            The key is the categore name, and the value is a list of all the
            components in the categore.
        """
        category = defaultdict(list)
        for k, cp in self.components.items():
            cat = cp.category
            category[cat].append(k)
        return category

    @lazyproperty
    def component_category_map(self):
        """A dictionary mapping component name to its category name.

        Return
        ------
        dict
            The mapping from components to its categore. The key is the component
            name and the value is the component's category name.
        """
        cp_ca = {}
        for k, cp in self.components.items():
            cp_ca[k] = cp.category
        return cp_ca

    @lazyproperty
    def component_unique_params(self):
        """Return the parameters that are only present in one component.

        Return
        ------
        dict
            A mapping from a component name to a list of parameters are only
            in this component.

        Note
        ----
        This function only returns the PINT defined parameter name, not
        including the aliases.
        """
        component_special_params = defaultdict(list)
        for param, cps in self.param_component_map.items():
            if len(cps) == 1:
                component_special_params[cps[0]].append(param)
        return component_special_params

    def search_binary_components(self, system_name):
        """Search the pulsar binary component based on given name.

        Parameters
        ----------
        system_name : str
            Searching name for the pulsar binary/system

        Return
        ------
        The matching binary model component instance.

        Raises
        ------
        UnknownBinaryModel
            If the input binary model name does not match any PINT defined binary
            model.
        """
        all_systems = self.category_component_map["pulsar_system"]
        # Search the system name first
        if system_name in all_systems:
            return self.components[system_name]
        else:  # search for the pulsar system aliases
            for cp_name in all_systems:
                if system_name == self.components[cp_name].binary_model_name:
                    return self.components[cp_name]

            if system_name == "BTX":
                raise UnknownBinaryModel(
                    "`BINARY  BTX` is not supported bt PINT. Use "
                    "`BINARY  BT` instead. It supports both orbital "
                    "period (PB, PBDOT) and orbital frequency (FB0, ...) "
                    "parametrizations."
                )
            elif system_name == "DDFWHE":
                raise UnknownBinaryModel(
                    "`BINARY  DDFWHE` is not supported, but the same model "
                    "is available as `BINARY  DDH`."
                )
            elif system_name in ["MSS", "EH", "H88", "DDT", "BT1P", "BT2P"]:
                # Binary model list taken from
                # https://tempo.sourceforge.net/ref_man_sections/binary.txt
                raise UnknownBinaryModel(
                    f"`The binary model {system_name} is not yet implemented."
                )

            raise UnknownBinaryModel(
                f"Pulsar system/Binary model component"
                f" {system_name} is not provided."
            )

    def alias_to_pint_param(self, alias):
        """Translate a alias to a PINT parameter name.

        This is a wrapper function over the property ``_param_alias_map``. It
        also handles indexed parameters (e.g., `pint.models.parameter.prefixParameter`
        and `pint.models.parameter.maskParameter`) with an index beyond those currently
        initialized.

        Parameters
        ----------
        alias : str
            Alias name to be translated

        Returns
        -------
        pint_par : str
            PINT parameter name the given alias maps to. If there is no matching
            PINT parameters, it will raise a `UnknownParameter` error.
        first_init_par : str
            The parameter name that is first initialized in a component. If the
            paramere is non-indexable, it is the same as ``pint_par``, otherwrise
            it returns the parameter with the first index. For example, the
            ``first_init_par`` for 'T2EQUAD25' is 'EQUAD1'

        Notes
        -----
        Providing a indexable parameter without the index attached, it returns
        the PINT parameter with first index (i.e. ``0`` or ``1``). If with index,
        the function returns the matched parameter with the index provided.
        The index format has to match the PINT defined index format. For instance,
        if PINT defines a parameter using leading-zero indexing, the provided
        index has to use the same leading-zeros, otherwrise, returns a `UnknownParameter`
        error.

        Examples
        --------
        >>> from pint.models.timing_model import AllComponents
        >>> ac = AllComponents()
        >>> ac.alias_to_pint_param('RA')
        ('RAJ', 'RAJ')

        >>> ac.alias_to_pint_param('T2EQUAD')
        ('EQUAD1', 'EQUAD1')

        >>> ac.alias_to_pint_param('T2EQUAD25')
        ('EQUAD25', 'EQUAD1')

        >>> ac.alias_to_pint_param('DMX_0020')
        ('DMX_0020', 'DMX_0001')

        >>> ac.alias_to_pint_param('DMX20')
        UnknownParameter: Can not find matching PINT parameter for 'DMX020'

        """
        pint_par = self._param_alias_map.get(alias, None)
        # If it is not in the map, double check if it is a repeatable par.
        if pint_par is None:
            try:
                prefix, idx_str, idx = split_prefixed_name(alias)
                # assume the index 1 parameter is in the alias map
                # count length of idx_str and dectect leading zeros
                # TODO fix the case for searching `DMX`
                num_lzero = len(idx_str) - len(str(idx))
                if num_lzero > 0:  # Has leading zero
                    fmt = len(idx_str)
                else:
                    fmt = 0
                first_init_par = None
                # Handle the case of start index from 0 and 1
                for start_idx in [0, 1]:
                    first_init_par_alias = prefix + f"{start_idx:0{fmt}}"
                    first_init_par = self._param_alias_map.get(
                        first_init_par_alias, None
                    )
                    if first_init_par:
                        # Find the first init par move to the next step
                        pint_par = split_prefixed_name(first_init_par)[0] + idx_str
                        break
            except PrefixError:
                pint_par = None

        else:
            first_init_par = pint_par
        if pint_par is None:
            raise UnknownParameter(
                "Can not find matching PINT parameter for '{}'".format(alias)
            )
        return pint_par, first_init_par

    def param_to_unit(self, name):
        """Return the unit associated with a parameter

        This is a wrapper function over the property ``_param_unit_map``.  It
        also handles aliases and indexed parameters (e.g., `pint.models.parameter.prefixParameter`
        and `pint.models.parameter.maskParameter`) with an index beyond those currently
        initialized.

        This can be used without an existing :class:`~pint.models.TimingModel`.

        Parameters
        ----------
        name : str
            Name of PINT parameter or alias

        Returns
        -------
        astropy.u.Unit
        """
        pintname, firstname = self.alias_to_pint_param(name)
        if pintname == firstname:
            # not a prefix parameter
            return self._param_unit_map[pintname]
        prefix, idx_str, idx = split_prefixed_name(pintname)
        component = self.param_component_map[firstname][0]
        if getattr(self.components[component], firstname).unit_template is None:
            return self._param_unit_map[firstname]
        return u.Unit(getattr(self.components[component], firstname).unit_template(idx))


class TimingModelError(ValueError):
    """Generic base class for timing model errors."""

    pass


class MissingParameter(TimingModelError):
    """A required model parameter was not included.

    Parameters
    ----------
    module
        name of the model class that raised the error
    param
        name of the missing parameter
    msg
        additional message

    """

    def __init__(self, module, param, msg=None):
        super().__init__(msg)
        self.module = module
        self.param = param
        self.msg = msg

    def __str__(self):
        result = self.module + "." + self.param
        if self.msg is not None:
            result += "\n  " + self.msg
        return result


class AliasConflict(TimingModelError):
    """If the same alias is used for different parameters."""

    pass


class UnknownParameter(TimingModelError):
    """Signal that a parameter name does not match any PINT parameters and their aliases."""

    pass


class UnknownBinaryModel(TimingModelError):
    """Signal that the par file requested a binary model not in PINT."""

    def __init__(self, message, suggestion=None):
        super().__init__(message)
        self.suggestion = suggestion

    def __str__(self):
        base_message = super().__str__()
        if self.suggestion:
            return f"{base_message} Perhaps use {self.suggestion}?"
        return base_message


class MissingBinaryError(TimingModelError):
    """Error for missing BINARY parameter."""

    pass
