"""Timing model objects.

Defines the basic timing model interface classes.
"""
from __future__ import absolute_import, division, print_function

import abc
import copy
import inspect
from collections import OrderedDict, defaultdict
from functools import wraps
from warnings import warn

import astropy.time as time
import astropy.units as u
import numpy as np
import six
from astropy import log
from scipy.optimize import brentq

import pint
from pint.models.parameter import (
    AngleParameter,
    MJDParameter,
    boolParameter,
    floatParameter,
    Parameter,
    maskParameter,
    strParameter,
)
from pint.phase import Phase
from pint.toa import TOAs
from pint.utils import PrefixError, interesting_lines, lines_of, split_prefixed_name

__all__ = ["DEFAULT_ORDER", "TimingModel"]
# Parameters or lines in parfiles we don't understand but shouldn't
# complain about. These are still passed to components so that they
# can use them if they want to.
#
# Other unrecognized parameters produce warnings and possibly indicate
# errors in the par file.
#
# Comparisons with keywords in par file lines is done in a case insensitive way.
ignore_params = set(
    [
        "TRES",
        "TZRMJD",
        "TZRFRQ",
        "TZRSITE",
        "NITS",
        "IBOOT",
        "BINARY",
        "CHI2R",
        "MODE",
        "PLANET_SHAPIRO2",
        #    'NE_SW', 'NE_SW2',
    ]
)

ignore_prefix = set(["DMXF1_", "DMXF2_", "DMXEP_"])  # DMXEP_ for now.

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
]


class MissingTOAs(ValueError):
    def __init__(self, parameter_names):
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        if len(parameter_names) == 1:
            msg = f"Parameter {parameter_names[0]} does not correspond to any TOAs"
        elif len(parameter_names) > 1:
            msg = (
                f"Parameters {' '.join(parameter_names)} do not correspond to any TOAs"
            )
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


class TimingModel(object):
    """Timing model object built from Components.

    This object is the primary object to represent a timing model in PINT.  It
    is normally constructed with :func:`pint.models.model_builder.get_model`,
    and it contains a variety of Component objects, each representing a
    physical process that either introduces delays in the pulse arrival time or
    introduces shifts in the pulse arrival phase.  These components have
    parameters, described by :class:`pint.models.parameter.Parameter` objects,
    and methods. Both the parameters and the methods are accessible through
    this object using attribute access, for example as ``model.F0`` or
    ``model.coords_as_GAL()``.

    Components in a TimingModel objects are accessible through the
    ``model.components`` property, using their class name to index the
    TimingModel, as ``model.components["Spindown"]``. They can be added and
    removed with methods on this object, and for many of them additional
    parameters in families (``DMXEP_1234``) can be added.

    Parameters in a TimingModel object are listed in the ``model.params`` and
    ``model.params_ordered`` objects. Each Parameter can be set as free or
    frozen using its ``.frozen`` attribute, and a list of the free parameters
    is available through the ``model.free_params`` property; this can also
    be used to set which parameters are free. Several methods are available
    to get and set some or all parameters in the forms of dictionaries.

    TimingModel objects also support a number of functions for computing
    various things like orbital phase, and barycentric versions of TOAs,
    as well as the various derivatives and matrices needed to support fitting.

    TimingModel objects can be written out to ``.par`` files using
    :func:`pint.models.timing_model.TimingModel.as_parfile`.

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
    :class:`pint.models.parameter.Parameter` delay or phase and its derivatives are implemented
    as TimingModel Methods.

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
            boolParameter(
                name="DILATEFREQ",
                value=False,
                description="Whether or not TEMPO2 should apply gravitational redshift and time dilation to obseerving frequency (Y/N; PINT only supports N)",
            ),
            "",
        )
        self.add_param_from_top(
            floatParameter(
                name="DMDATA",
                value=0.0,
                units="",
                description="Was the fit done using per-TOA DM information?",
            ),
            "",
        )
        self.add_param_from_top(
            floatParameter(
                name="NTOA",
                value=0.0,
                units="",
                description="Number of TOAs used in the fitting",
            ),
            "",
        )
        self.add_param_from_top(
            floatParameter(
                name="CHI2",
                value=0.0,
                units="",
                description="Chi-squared value obtained during fitting",
            ),
            "",
        )

        for cp in components:
            self.add_component(cp, validate=False)

    def __repr__(self):
        return "{}(\n  {}\n)".format(
            self.__class__.__name__,
            ",\n  ".join(str(v) for k, v in sorted(self.components.items())),
        )

    def __str__(self):
        return self.as_parfile()

    def validate(self):
        """Validate component setup.

        The checks include required parameters and parameter values.
        """
        if self.DILATEFREQ.value:
            warn("PINT does not support 'DILATEFREQ Y'")
        if self.TIMEEPH.value not in [None, "FB90"]:
            warn("PINT only supports 'TIMEEPH FB90'")
        if self.T2CMETHOD.value not in [None, "IAU2000B"]:  # FIXME: really?
            warn("PINT only supports 'T2CMETHOD IAU2000B'")
        if self.UNITS.value not in [None, "TDB"]:
            raise ValueError("PINT only supports 'UNITS TDB'")
        for cp in self.components.values():
            cp.validate()

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
            "Attribute {} not found in TimingModel or any Component".format(name)
        )

    @property_exists
    def params(self):
        """List of all parameter names in this model and all its components (order is arbitrary)."""
        # FIXME: any reason not to just use params_ordered here?
        p = self.top_level_params
        for cp in self.components.values():
            p = p + cp.params
        return p

    @property_exists
    def params_ordered(self):
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
            if cat in compdict:
                cp = compdict[cat]
                for cpp in cp:
                    pstart += cpp.params
                used_cats.add(cat)
            else:
                continue

        pend = []
        for cat in last_order:
            if cat in compdict:
                cp = compdict[cat]
                for cpp in cp:
                    pend += cpp.parms
                used_cats.add(cat)
            else:
                continue

        # Now collect any components that haven't already been included in the list
        pmid = []
        for cat in compdict:
            if cat in used_cats:
                continue
            else:
                cp = compdict[cat]
                for cpp in cp:
                    pmid += cpp.params
                used_cats.add(cat)

        return pstart + pmid + pend

    @property_exists
    def free_params(self):
        """List of all the free parameters in the timing model. Can be set to change which are free.

        These are ordered as ``self.params_ordered`` does.

        Upon setting, order does not matter, and aliases are accepted.
        ValueError is raised if a parameter is not recognized.

        On setting, parameter aliases are converted with
        :func:`pint.models.timing_model.TimingModel.match_param_aliases`.
        """
        return [p for p in self.params_ordered if not getattr(self, p).frozen]

    @free_params.setter
    def free_params(self, params):
        params_true = {self.match_param_aliases(p) for p in params}
        for p in self.params:
            getattr(self, p).frozen = p not in params_true
            params_true.discard(p)
        if params_true:
            raise ValueError(
                "Parameter(s) are familiar but not in the model: {}".format(params)
            )

    def match_param_aliases(self, alias):
        """Return the parameter corresponding to this alias."""
        for p in self.params:
            if p == alias:
                return p
            if alias in getattr(self, p).aliases:
                return p
        raise ValueError("{} is not recognized as a parameter or alias".format(alias))

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
            ps = self.params_ordered
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
                raise ValueError("Unknown kind '{}'".format(kind))
        return c

    def set_param_values(self, fitp):
        """Set the model parameters to the value contained in the input dict.

        Ex. model.set_param_values({'F0':60.1,'F1':-1.3e-15})
        """
        # In Powell fitter this sometimes fails because after some iterations the values change from
        # plain float to Quantities. No idea why.
        for k, v in fitp.items():
            p = getattr(self, k)
            if isinstance(v, Parameter):
                if v.value is None:
                    raise ValueError("Parameter {} is unset".format(v))
                p.value = v.value
            elif isinstance(v, u.Quantity):
                p.value = v.to_value(p.units)
            else:
                p.value = v

    def set_param_uncertainties(self, fitp):
        """Set the model parameters to the value contained in the input dict."""
        for k, v in fitp.items():
            p = getattr(self, k)
            if isinstance(v, u.Quantity):
                p.uncertainty = v
            else:
                p.uncertainty = v * p.units

    @property_exists
    def components(self):
        """All the components in a dictionary indexed by name."""
        comps = {}
        for ct in self.component_types:
            for cp in getattr(self, ct + "_list"):
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
            If a TOAs instance is passed, the barycenting will happen
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
        b.update_binary_object()
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
            anoms = bbi.nu()
        else:
            raise ValueError("anom='%s' is not a recognized type of anomaly" % anom)
        # Make sure all angles are between 0-2*pi
        anoms = np.fmod(anoms.value, 2 * np.pi)
        if radians:  # return with radian units
            return anoms * u.rad
        else:  # return as unitless cycles from 0-1
            return anoms / (2 * np.pi)

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
            return np.fmod((nu + bbi.omega()).value, 2 * np.pi) - np.pi / 2

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
            ts = np.linspace(bt, bt + self.PB.value, 11)
            # Compute the true anomalies and omegas for those times
            nus = self.orbital_phase(ts, anom="true")
            omegas = bbi.omega()
            x = np.fmod((nus + omegas).value, 2 * np.pi) - np.pi / 2
            # find the lowest index where x is just below 0
            for lb in range(len(x)):
                if x[lb] < 0 and x[lb + 1] > 0:
                    break
            # Now use scipy to find the root
            scs.append(brentq(funct, ts[lb], ts[lb + 1]))
        if len(scs) == 1:
            return scs[0]  # Return a float
        else:
            return np.asarray(scs)  # otherwise return an array

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
        if "NoiseComponent" in self.component_types:
            for nc in self.NoiseComponent_list:
                # recursive if necessary
                if nc.introduces_correlated_errors:
                    return True
        return False

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
        """List of dm derivative functions."""
        return self.get_deriv_funcs("DelayComponent", "dm")

    @property_exists
    def d_phase_d_delay_funcs(self):
        """List of d_phase_d_delay functions."""
        Dphase_Ddelay = []
        for cp in self.PhaseComponent_list:
            Dphase_Ddelay += cp.phase_derivs_wrt_delay
        return Dphase_Ddelay

    def get_deriv_funcs(self, component_type, derivative_type=""):
        """Return dictionary of derivative functions."""
        # TODO, this function can be a more generical function collector.
        deriv_funcs = defaultdict(list)
        if not derivative_type == "":
            derivative_type += "_"
        for cp in getattr(self, component_type + "_list"):
            try:
                df = getattr(cp, derivative_type + "deriv_funcs")
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
        raise AttributeError("{} not found in any component".format(name))

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
                "Class '%s' is not a Component type class."
                % component.__class__.__name__
            )
        elif len(comp_base) < 3:
            raise TypeError(
                "'%s' class is not a subclass of 'Component' class."
                % component.__class__.__name__
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
                raise AttributeError("No '%s' in the timing model." % component)
            comp = comps[component]
        else:  # When component is an component instance.
            if component not in list(comps.values()):
                raise AttributeError(
                    "No '%s' in the timing model." % component.__class__.__name__
                )
            else:
                comp = component
        comp_type = self.get_component_type(comp)
        host_list = getattr(self, comp_type + "_list")
        order = host_list.index(comp)
        return comp, order, host_list, comp_type

    def add_component(self, component, order=DEFAULT_ORDER, force=False, validate=True):
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
        if comp_type in self.component_types:
            comp_list = getattr(self, comp_type + "_list")
            cur_cps = []
            for cp in comp_list:
                cur_cps.append((order.index(cp.category), cp))
            # Check if the component has been added already.
            if component.__class__ in (x.__class__ for x in comp_list):
                log.warning(
                    "Component '%s' is already present but was added again."
                    % component.__class__.__name__
                )
                if not force:
                    raise ValueError(
                        "Component '%s' is already present and will not be "
                        "added again. To force add it, use force=True option."
                        % component.__class__.__name__
                    )
        else:
            self.component_types.append(comp_type)
            cur_cps = []

        # link new component to TimingModel
        component._parent = self

        # If the categore is not in the order list, it will be added to the end.
        if component.category not in order:
            new_cp = tuple((len(order) + 1, component))
        else:
            new_cp = tuple((order.index(component.category), component))
        # add new component
        cur_cps.append(new_cp)
        cur_cps.sort(key=lambda x: x[0])
        new_comp_list = [c[1] for c in cur_cps]
        setattr(self, comp_type + "_list", new_comp_list)
        # Set up components
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

    def _locate_param_host(self, components, param):
        """Search for the parameter host component.

        Parameters
        ----------
        components: list
            Searching component list.
        param: str
            Target parameter.

        Return
        ------
        List of tuples. The first element is the component object that have the
        target parameter, the second one is the parameter object. If it is a
        prefix-style parameter, it will return one example of such parameter.
        """
        result_comp = []
        for cp in components:
            if param in cp.params:
                result_comp.append((cp, getattr(cp, param)))
            else:
                # search for prefixed parameter
                prefixs = cp.param_prefixs
                try:
                    prefix, index_str, index = split_prefixed_name(param)
                except PrefixError:
                    prefix = param

                if prefix in prefixs.keys():
                    result_comp.append(cp, getattr(cp, prefixs[param][0]))

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
        param: str
            Parameter name
        target_component: str
            Parameter host component name. If given as "" it would add
            parameter to the top level `TimingModel` class
        setup: bool, optional
            Flag to run setup() function.
        """
        if target_component == "":
            setattr(self, param.name, param)
            self.top_level_params += [param.name]
        else:
            if target_component not in list(self.components.keys()):
                raise AttributeError(
                    "Can not find component '%s' in " "timing model." % target_component
                )
            self.components[target_component].add_param(param, setup=setup)

    def remove_param(self, param):
        """Remove a parameter from timing model.

        Parameters
        ----------
        param: str
            The name of parameter to be removed.
        """
        param_map = self.get_params_mapping()
        if param not in list(param_map.keys()):
            raise AttributeError("Can not find '%s' in timing model." % param)
        if param_map[param] == "timing_model":
            delattr(self, param)
            self.top_level_params.remove(param)
        else:
            target_component = param_map[param]
            self.components[target_component].remove_param(param)

    def get_params_mapping(self):
        """Report whick component each parameter name comes from."""
        param_mapping = {}
        for p in self.top_level_params:
            param_mapping[p] = "timing_model"
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
           A dictionary with prefix pararameter real index as key and parameter
           name as value.
        """
        parnames = [x for x in self.params if x.startswith(prefix)]
        mapping = dict()
        for parname in parnames:
            par = getattr(self, parname)
            if par.is_prefix and par.prefix == prefix:
                mapping[par.index] = parname
        return mapping

    def param_help(self):
        """Print help lines for all available parameters in model."""
        return "".join(
            "{:<40}{}\n".format(cp, getattr(self, par).help_line())
            for par, cp in self.get_params_mapping().items()
        )

    def delay(self, toas, cutoff_component="", include_last=True):
        """Total delay for the TOAs.

        Parameters
        ----------
        toas: toa.table
            The toas for analysis delays.
        cutoff_component: str
            The delay component name that a user wants the calculation to stop
            at.
        include_last: bool
            If the cutoff delay component is included.

        Return the total delay which will be subtracted from the given
        TOA to get time of emission at the pulsar.
        """
        delay = np.zeros(toas.ntoas) * u.second
        if cutoff_component == "":
            idx = len(self.DelayComponent_list)
        else:
            delay_names = [x.__class__.__name__ for x in self.DelayComponent_list]
            if cutoff_component in delay_names:
                idx = delay_names.index(cutoff_component)
                if include_last:
                    idx += 1
            else:
                raise KeyError("No delay component named '%s'." % cutoff_component)

        # Do NOT cycle through delay_funcs - cycle through components until cutoff
        for dc in self.DelayComponent_list[:idx]:
            for df in dc.delay_funcs_component:
                delay += df(toas, delay)
        return delay

    def phase(self, toas, abs_phase=False):
        """Return the model-predicted pulse phase for the given TOAs."""
        # First compute the delays to "pulsar time"
        delay = self.delay(toas)
        phase = Phase(np.zeros(toas.ntoas), np.zeros(toas.ntoas))
        # Then compute the relevant pulse phases
        for pf in self.phase_funcs:
            phase += Phase(pf(toas, delay))

        # If the absolute phase flag is on, use the TZR parameters to compute
        # the absolute phase.
        if abs_phase:
            if "AbsPhase" not in list(self.components.keys()):
                # if no absolute phase (TZRMJD), add the component to the model and calculate it
                from pint.models import absolute_phase

                self.add_component(absolute_phase.AbsPhase(), validate=False)
                self.make_TZR_toa(
                    toas
                )  # TODO:needs timfile to get all toas, but model doesn't have access to timfile. different place for this?
                self.validate()
            tz_toa = self.get_TZR_toa(toas)
            tz_delay = self.delay(tz_toa)
            tz_phase = Phase(np.zeros(len(toas.table)), np.zeros(len(toas.table)))
            for pf in self.phase_funcs:
                tz_phase += Phase(pf(tz_toa, tz_delay))
            return phase - tz_phase
        else:
            return phase

    def total_dm(self, toas):
        """Calculate dispersion measure from all the dispersion type of components."""
        # Here we assume the unit would be the same for all the dm value function.
        # By doing so, we do not have to hard code an unit here.
        dm = self.dm_funcs[0](toas)

        for dm_f in self.dm_funcs[1::]:
            dm += dm_f(toas)
        return dm

    def toa_covariance_matrix(self, toas):
        """Get the TOA covariance matrix for noise models.

        If there is no noise model component provided, a diagonal matrix with
        TOAs error as diagonal element will be returned.
        """
        ntoa = toas.ntoas
        tbl = toas.table
        result = np.zeros((ntoa, ntoa))
        # When there is no noise model.
        if len(self.covariance_matrix_funcs) == 0:
            result += np.diag(tbl["error"].quantity.to(u.s).value ** 2)
            return result

        for nf in self.covariance_matrix_funcs:
            result += nf(toas)
        return result

    def dm_covariance_matrix(self, toas):
        """Get the DM covariance matrix for noise models.

        If there is no noise model component provided, a diagonal matrix with
        TOAs error as diagonal element will be returned.
        """
        dms, valid_dm = toas.get_flag_value("pp_dm")
        dmes, valid_dme = toas.get_flag_value("pp_dme")
        dms = np.array(dms)[valid_dm]
        n_dms = len(dms)
        dmes = np.array(dmes)[valid_dme]
        result = np.zeros((n_dms, n_dms))
        # When there is no noise model.
        if len(self.dm_covariance_matrix_funcs) == 0:
            result += np.diag(dmes.value ** 2)
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
        toas: `pint.toa.TOAs` object
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
        toas: `pint.toa.TOAs` object
            The input data object for DM uncertainty.
        """
        dm_error, valid = toas.get_flag_value("pp_dme")
        dm_error = np.array(dm_error)[valid] * u.pc / u.cm ** 3
        result = np.zeros(len(dm_error)) * u.pc / u.cm ** 3
        # When there is no noise model.
        if len(self.scaled_dm_uncertainty_funcs) == 0:
            result += dm_error
            return result

        for nf in self.scaled_dm_uncertainty_funcs:
            result += nf(toas)
        return result

    def noise_model_designmatrix(self, toas):
        result = []
        if len(self.basis_funcs) == 0:
            return None

        for nf in self.basis_funcs:
            result.append(nf(toas)[0])
        return np.hstack([r for r in result])

    def noise_model_basis_weight(self, toas):
        result = []
        if len(self.basis_funcs) == 0:
            return None

        for nf in self.basis_funcs:
            result.append(nf(toas)[1])
        return np.hstack([r for r in result])

    def noise_model_dimensions(self, toas):
        """Number of basis functions for each noise model component.

        Returns a dictionary of correlated-noise components in the noise
        model.  Each entry contains a tuple (offset, size) where size is the
        number of basis funtions for the component, and offset is their
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
                nbf = 0
                for bf in bfs:
                    nbf += len(bf(toas)[1])
                result[nc.category] = (ntot, nbf)
                ntot += nbf

        return result

    def jump_flags_to_params(self, toas):
        """Convert jump flags in toas.table["flags"] to jump parameters in the model.

        The flags processed are ``jump`` and ``gui_jump``.
        """
        from . import jump

        for flag_dict in toas.table["flags"]:
            if "jump" in flag_dict.keys():
                break
            elif "gui_jump" in flag_dict.keys():
                break
        else:
            log.info("No jump flags to process")
            return None
        for flag_dict in toas.table["flags"]:
            if "jump" in flag_dict.keys():
                jump_nums = [
                    flag_dict["jump"] if "jump" in flag_dict.keys() else np.nan
                    for flag_dict in toas.table["flags"]
                ]
                if "PhaseJump" not in self.components:
                    log.info("PhaseJump component added")
                    a = jump.PhaseJump()
                    a.setup()
                    self.add_component(a)
                    self.remove_param("JUMP1")
                for num in np.arange(1, np.nanmax(jump_nums) + 1):
                    if "JUMP" + str(int(num)) not in self.params:
                        param = maskParameter(
                            name="JUMP",
                            index=int(num),
                            key="jump",
                            key_value=int(num),
                            value=0.0,
                            units="second",
                            uncertainty=0.0,
                        )
                        self.add_param_from_top(param, "PhaseJump")
                        getattr(self, param.name).frozen = False
                if 0 in jump_nums:
                    for flag_dict in toas.table["flags"]:
                        if "jump" in flag_dict.keys() and flag_dict["jump"] == 0:
                            flag_dict["jump"] = int(np.nanmax(jump_nums) + 1)
                    param = maskParameter(
                        name="JUMP",
                        index=int(np.nanmax(jump_nums) + 1),
                        key="jump",
                        key_value=int(np.nanmax(jump_nums) + 1),
                        value=0.0,
                        units="second",
                        uncertainty=0.0,
                    )
                    self.add_param_from_top(param, "PhaseJump")
                    getattr(self, param.name).frozen = False
        # convert string list key_value from file into int list for jumps
        # previously added thru pintk
        for flag_dict in toas.table["flags"]:
            if "gui_jump" in flag_dict.keys():
                num = flag_dict["gui_jump"]
                jump = getattr(self.components["PhaseJump"], "JUMP" + str(num))
                jump.key_value = list(map(int, jump.key_value))
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
        astropy.Quantity
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
        """Return the derivative of phase wrt TOA.

        Parameters
        ----------
        toas : PINT TOAs class
            The toas when the derivative of phase will be evaluated at.
        sample_step : float optional
            Finite difference steps. If not specified, it will take 1/10 of the
            spin period.

        """
        copy_toas = copy.deepcopy(toas)
        if sample_step is None:
            pulse_period = 1.0 / (self.F0.quantity)
            sample_step = pulse_period * 1000
        sample_dt = [-sample_step, 2 * sample_step]

        sample_phase = []
        for dt in sample_dt:
            dt_array = [dt.value] * copy_toas.ntoas * dt._unit
            deltaT = time.TimeDelta(dt_array)
            copy_toas.adjust_TOAs(deltaT)
            phase = self.phase(copy_toas)
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
        pass

    def d_phase_d_param(self, toas, delay, param):
        """Return the derivative of phase with respect to the parameter."""
        # TODO need to do correct chain rule stuff wrt delay derivs, etc
        # Is it safe to assume that any param affecting delay only affects
        # phase indirectly (and vice-versa)??
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
                "Derivative function for '%s' is not provided"
                " or not registered. " % param
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
        if ori_value == 0:
            h = 1.0 * step
        else:
            h = ori_value * step
        parv = [par.value - h, par.value + h]

        phase_i = (
            np.zeros((toas.ntoas, 2), dtype=np.longdouble) * u.dimensionless_unscaled
        )
        phase_f = (
            np.zeros((toas.ntoas, 2), dtype=np.longdouble) * u.dimensionless_unscaled
        )
        for ii, val in enumerate(parv):
            par.value = val
            ph = self.phase(toas)
            phase_i[:, ii] = ph.int
            phase_f[:, ii] = ph.frac
        res_i = -phase_i[:, 0] + phase_i[:, 1]
        res_f = -phase_f[:, 0] + phase_f[:, 1]
        result = (res_i + res_f) / (2.0 * h * unit)
        # shift value back to the original value
        par.quantity = ori_value
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
            log.warning("Parameter '%s' is not used by timing model." % param)
            return np.zeros(toas.ntoas) * (u.second / par.units)
        unit = par.units
        if ori_value == 0:
            h = 1.0 * step
        else:
            h = ori_value * step
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
        """Return the derivative of dm with respect to the parameter."""
        par = getattr(self, param)
        result = np.zeros(len(data)) << (u.pc / u.cm ** 3 / par.units)
        dm_df = self.dm_derivs.get(param, None)
        if dm_df is None:
            if param not in self.params:  # Maybe add differentitable params
                raise AttributeError("Parameter {} does not exist".format(param))
            else:
                return result

        for df in dm_df:
            result += df(data, param).to(
                result.unit, equivalencies=u.dimensionless_angles()
            )
        return result

    def designmatrix(self, toas, acc_delay=None, incfrozen=False, incoffset=True):
        """Return the design matrix.

        The design matrix is the matrix with columns of d_phase_d_param/F0
        or d_toa_d_param; it is used in fitting and calculating parameter
        covariances.

        """
        params = ["Offset"] if incoffset else []
        params += [
            par for par in self.params if incfrozen or not getattr(self, par).frozen
        ]

        F0 = self.F0.quantity  # 1/sec
        ntoas = toas.ntoas
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
                M[:, ii] = 1.0
                units.append(u.s / u.s)
            else:
                # NOTE Here we have negative sign here. Since in pulsar timing
                # the residuals are calculated as (Phase - int(Phase)), which is different
                # from the conventional definition of least square definition (Data - model)
                # We decide to add minus sign here in the design matrix, so the fitter
                # keeps the conventional way.
                q = -self.d_phase_d_param(toas, delay, param)
                M[:, ii] = q
                units.append(u.Unit("") / getattr(self, param).units)
        mask = []
        for ii, un in enumerate(units):
            if params[ii] == "Offset":
                continue
            units[ii] = un * u.second
            mask.append(ii)
        M[:, mask] /= F0.value
        return M, params, units

    def compare(self, othermodel, nodmx=True, threshold_sigma=3.0, verbosity="max"):
        """Print comparison with another model

        Parameters
        ----------
        othermodel
            TimingModel object to compare to
        nodmx : bool
            If True (which is the default), don't print the DMX parameters in
            the comparison
        threshold_sigma : float
            Pulsar parameters for which diff_sigma > threshold will be printed
            with an exclamation point at the end of the line
        verbosity : string
            Dictates amount of information returned. Options include "max",
            "med", and "min", which have the following results:
                "max"     - print all lines from both models whether they are fit or not (note that nodmx will override this); DEFAULT
                "med"     - only print lines for parameters that are fit
                "min"     - only print lines for fit parameters for which diff_sigma > threshold
                "check"   - only print significant changes with astropy.log.warning, not as string (note that all other modes will still print this)

        Returns
        -------
        str
            Human readable comparison, for printing.
            Formatted as a five column table with titles of
            PARAMETER NAME | Model 1 | Model 2 | Diff_Sigma1 | Diff_Sigma2
            where Model 1/2 refer to self and othermodel Timing Model objects,
            and Diff_SigmaX is the difference in a given parameter as reported by the two models,
            normalized by the uncertainty in model X. If model X has no reported uncertainty,
            nothing will be printed. When either Diff_Sigma value is greater than threshold_sigma,
            an exclamation point (!) will be appended to the line. If the uncertainty in the first model
            if smaller than the second, an asterisk (*) will be appended to the line. Also, astropy
            warnings and info statements will be printed.

        Note
        ----
            Prints astropy.log warnings for parameters that have changed significantly
            and/or have increased in uncertainty.
        """
        import sys
        from copy import deepcopy as cp

        import uncertainties.umath as um
        from uncertainties import ufloat

        s = "{:14s} {:>28s} {:>28s} {:14s} {:14s}\n".format(
            "PARAMETER", "Model 1", "Model 2 ", "Diff_Sigma1", "Diff_Sigma2"
        )
        s += "{:14s} {:>28s} {:>28s} {:14s} {:14s}\n".format(
            "---------", "----------", "----------", "----------", "----------"
        )
        log.info("Comparing ephemerides for PSR %s" % self.PSR.value)
        log.info("Threshold sigma = %f" % threshold_sigma)
        log.info("Creating a copy of Model 2")
        if verbosity == "max":
            log.info("Maximum verbosity - printing all parameters")
        elif verbosity == "med":
            log.info("Medium verbosity - printing parameters that are fit")
        elif verbosity == "min":
            log.info(
                "Minimum verbosity - printing parameters that are fit and significantly changed"
            )
        elif verbosity == "check":
            log.info("Check verbosity - only warnings/info with be displayed")
        othermodel = cp(othermodel)

        if (
            "POSEPOCH" in self.params_ordered
            and "POSEPOCH" in othermodel.params_ordered
        ):
            if (
                self.POSEPOCH.value is not None
                and self.POSEPOCH.value != othermodel.POSEPOCH.value
            ):
                log.info("Updating POSEPOCH in Model 2 to match Model 1")
                othermodel.change_posepoch(self.POSEPOCH.value)
        if "PEPOCH" in self.params_ordered and "PEPOCH" in othermodel.params_ordered:
            if (
                self.PEPOCH.value is not None
                and self.PEPOCH.value != othermodel.PEPOCH.value
            ):
                log.info("Updating PEPOCH in Model 2 to match Model 1")
                othermodel.change_pepoch(self.PEPOCH.value)
        if "DMEPOCH" in self.params_ordered and "DMEPOCH" in othermodel.params_ordered:
            if (
                self.DMEPOCH.value is not None
                and self.DMEPOCH.value != othermodel.DMEPOCH.value
            ):
                log.info("Updating DMEPOCH in Model 2 to match Model 1")
                othermodel.change_posepoch(self.DMEPOCH.value)
        for pn in self.params_ordered:
            par = getattr(self, pn)
            if par.value is None:
                continue
            newstr = ""
            try:
                otherpar = getattr(othermodel, pn)
            except AttributeError:
                otherpar = None
            if isinstance(par, strParameter):
                newstr += "{:14s} {:>28s}".format(pn, par.value)
                if otherpar is not None:
                    newstr += " {:>28s}\n".format(otherpar.value)
                else:
                    newstr += " {:>28s}\n".format("Missing")
            elif isinstance(par, AngleParameter):
                if par.frozen:
                    # If not fitted, just print both values
                    newstr += "{:14s} {:>28s}".format(pn, str(par.quantity))
                    if otherpar is not None:
                        newstr += " {:>28s}\n".format(str(otherpar.quantity))
                    else:
                        newstr += " {:>28s}\n".format("Missing")
                else:
                    # If fitted, print both values with uncertainties
                    if par.units == u.hourangle:
                        uncertainty_unit = pint.hourangle_second
                    else:
                        uncertainty_unit = u.arcsec
                    newstr += "{:14s} {:>16s} +/- {:7.2g}".format(
                        pn,
                        str(par.quantity),
                        par.uncertainty.to(uncertainty_unit).value,
                    )
                    if otherpar is not None:
                        try:
                            newstr += " {:>16s} +/- {:7.2g}".format(
                                str(otherpar.quantity),
                                otherpar.uncertainty.to(uncertainty_unit).value,
                            )
                        except AttributeError:
                            # otherpar must have no uncertainty
                            if otherpar.quantity is not None:
                                newstr += " {:>28s}".format(str(otherpar.quantity))
                            else:
                                newstr += " {:>28s}".format("Missing")
                    else:
                        newstr += " {:>28s}".format("Missing")
                    try:
                        diff = otherpar.value - par.value
                        diff_sigma = diff / par.uncertainty.value
                        if abs(diff_sigma) != np.inf:
                            newstr += " {:>10.2f}".format(diff_sigma)
                            if abs(diff_sigma) > threshold_sigma:
                                newstr += " !"
                            else:
                                newstr += "  "
                        else:
                            newstr += "           "
                        diff_sigma2 = diff / otherpar.uncertainty.value
                        if abs(diff_sigma2) != np.inf:
                            newstr += " {:>10.2f}".format(diff_sigma2)
                            if abs(diff_sigma2) > threshold_sigma:
                                newstr += " !"
                    except (AttributeError, TypeError):
                        pass
                    if otherpar is not None:
                        if (
                            par.uncertainty is not None
                            and otherpar.uncertainty is not None
                        ):
                            if 1.01 * par.uncertainty < otherpar.uncertainty:
                                newstr += " *"
                    newstr += "\n"
            else:
                # Assume numerical parameter
                if nodmx and pn.startswith("DMX"):
                    continue
                if par.frozen:
                    # If not fitted, just print both values
                    newstr += "{:14s} {:28f}".format(pn, par.value)
                    if otherpar is not None and otherpar.value is not None:
                        try:
                            newstr += " {:28SP}".format(
                                ufloat(otherpar.value, otherpar.uncertainty.value)
                            )
                        except (ValueError, AttributeError):
                            newstr += " {:28f}".format(otherpar.value)
                        if otherpar.value != par.value:
                            sys.stdout.flush()
                            if par.name == "START" or par.name == "FINISH":
                                log.info(
                                    "Parameter %s not fit, but has changed between these models"
                                    % par.name
                                )
                            else:
                                log.warning(
                                    "Parameter %s not fit, but has changed between these models"
                                    % par.name
                                )
                            log.handlers[0].flush()
                            newstr += " !"
                        if (
                            par.uncertainty is not None
                            and otherpar.uncertainty is not None
                        ):
                            if 1.01 * par.uncertainty < otherpar.uncertainty:
                                newstr += " *"
                        newstr += "\n"
                    else:
                        newstr += " {:>28s}\n".format("Missing")
                else:
                    # If fitted, print both values with uncertainties
                    newstr += "{:14s} {:28SP}".format(
                        pn, ufloat(par.value, par.uncertainty.value)
                    )
                    if otherpar is not None and otherpar.value is not None:
                        try:
                            newstr += " {:28SP}".format(
                                ufloat(otherpar.value, otherpar.uncertainty.value)
                            )
                        except AttributeError:
                            # otherpar must have no uncertainty
                            if otherpar.value is not None:
                                newstr += " {:28f}".format(otherpar.value)
                            else:
                                newstr += " {:>28s}".format("Missing")
                    else:
                        newstr += " {:>28s}".format("Missing")
                    if "Missing" in newstr:
                        ind = np.where(np.array(newstr.split()) == "Missing")[0][0]
                        log.info("Parameter %s missing from model %i" % (par.name, ind))
                    try:
                        diff = otherpar.value - par.value
                        diff_sigma = diff / par.uncertainty.value
                        if abs(diff_sigma) != np.inf:
                            newstr += " {:>10.2f}".format(diff_sigma)
                            if abs(diff_sigma) > threshold_sigma:
                                newstr += " !"
                        else:
                            newstr += "           "
                        diff_sigma2 = diff / otherpar.uncertainty.value
                        if abs(diff_sigma2) != np.inf:
                            newstr += " {:>10.2f}".format(diff_sigma2)
                            if abs(diff_sigma2) > threshold_sigma:
                                newstr += " !"
                    except (AttributeError, TypeError):
                        pass
                    if otherpar is not None:
                        if (
                            par.uncertainty is not None
                            and otherpar.uncertainty is not None
                        ):
                            if par.uncertainty < otherpar.uncertainty:
                                newstr += " *"
                    newstr += "\n"

            if "!" in newstr and not par.frozen:
                try:
                    sys.stdout.flush()
                    log.warning(
                        "Parameter %s has changed significantly (%f sigma)"
                        % (newstr.split()[0], float(newstr.split()[-2]))
                    )
                    log.handlers[0].flush()
                except ValueError:
                    sys.stdout.flush()
                    log.warning(
                        "Parameter %s has changed significantly (%f sigma)"
                        % (newstr.split()[0], float(newstr.split()[-3]))
                    )
                    log.handlers[0].flush()
            if "*" in newstr:
                sys.stdout.flush()
                log.warning(
                    "Uncertainty on parameter %s has increased (unc2/unc1 = %2.2f)"
                    % (newstr.split()[0], float(otherpar.uncertainty / par.uncertainty))
                )
                log.handlers[0].flush()

            if verbosity == "max":
                s += newstr
            elif verbosity == "med":
                if not par.frozen:
                    s += newstr
            elif verbosity == "min":
                if "!" in newstr and not par.frozen:
                    s += newstr
            elif verbosity != "check":
                raise AttributeError(
                    'Options for verbosity are "max" (default), "med", "min", and "check"'
                )
        # Now print any parameters in othermodel that were missing in self.
        mypn = self.params_ordered
        for opn in othermodel.params_ordered:
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
            log.info("Parameter %s missing from model 1" % opn)
            if verbosity == "max":
                s += "{:14s} {:>28s}".format(opn, "Missing")
                s += " {:>28s}".format(str(otherpar.quantity))
                s += "\n"
        if verbosity != "check":
            return s.split("\n")

    def read_parfile(self, file, validate=True):
        """Read values from the specified parfile into the model parameters.

        Parameters
        ----------
        file : str or list or file-like
            The parfile to read from. May be specified as a filename,
            a list of lines, or a readable file-like object.
        """
        repeat_param = defaultdict(int)
        param_map = self.get_params_mapping()
        comps = self.components.copy()
        comps["timing_model"] = self
        wants_tcb = None
        stray_lines = []
        for li in interesting_lines(lines_of(file), comments=("#", "C ")):
            k = li.split()
            name = k[0].upper()

            if name == "UNITS":
                if name in repeat_param:
                    raise ValueError("UNITS is repeated in par file")
                else:
                    repeat_param[name] += 1
                if len(k) > 1 and k[1] == "TDB":
                    wants_tcb = False
                else:
                    wants_tcb = li
                self.UNITS.value = k[1]
                continue

            if name == "EPHVER":
                if len(k) > 1 and k[1] != "2" and wants_tcb is None:
                    wants_tcb = li
                log.warning("EPHVER %s does nothing in PINT" % k[1])
                # actually people expect EPHVER 5 to work
                # even though it's supposed to imply TCB which doesn't
                continue

            if name == "START":
                if name in repeat_param:
                    raise ValueError("START is repeated in par file")
                self.START.value = k[1]
                continue

            if name == "FINISH":
                if name in repeat_param:
                    raise ValueError("FINISH is repeated in par file")
                self.FINISH.value = k[1]
                continue

            repeat_param[name] += 1
            if repeat_param[name] > 1:
                k[0] = k[0] + str(repeat_param[name])
                li = " ".join(k)

            used = []
            for p, c in param_map.items():
                if getattr(comps[c], p).from_parfile_line(li):
                    used.append((c, p))
            if len(used) > 1:
                log.warning(
                    "More than one component made use of par file "
                    "line {!r}: {}".format(li, used)
                )
            if used:
                continue

            if name in ignore_params:
                log.debug("Ignoring parfile line '%s'" % (li,))
                continue

            try:
                prefix, f, v = split_prefixed_name(name)
                if prefix in ignore_prefix:
                    log.debug("Ignoring prefix parfile line '%s'" % (li,))
                    continue
            except PrefixError:
                pass

            stray_lines.append(li)

        if wants_tcb:
            raise ValueError(
                "Only UNITS TDB supported by PINT but parfile has {}".format(wants_tcb)
            )
        if stray_lines:
            for l in stray_lines:
                log.warning("Unrecognized parfile line {!r}".format(l))
            for name, param in getattr(self, "discarded_components", []):
                log.warning(
                    "Model component {} was rejected because we "
                    "didn't find parameter {}".format(name, param)
                )
            # Disable here for now. TODO  need to modified.
            # log.info("Final object: {}".format(repr(self)))

        self.setup()
        # The "validate" functions contain tests for required parameters or
        # combinations of parameters, etc, that can only be done
        # after the entire parfile is read
        if validate:
            self.validate()

    def as_parfile(
        self,
        start_order=["astrometry", "spindown", "dispersion"],
        last_order=["jump_delay"],
    ):
        """Represent the entire model as a parfile string."""
        self.validate()
        result_begin = ""
        result_end = ""
        result_middle = ""
        cates_comp = self.get_components_by_category()
        printed_cate = []
        for p in self.top_level_params:
            result_begin += getattr(self, p).as_parfile_line()
        for cat in start_order:
            if cat in list(cates_comp.keys()):
                cp = cates_comp[cat]
                for cpp in cp:
                    result_begin += cpp.print_par()
                printed_cate.append(cat)
            else:
                continue

        for cat in last_order:
            if cat in list(cates_comp.keys()):
                cp = cates_comp[cat]
                for cpp in cp:
                    result_end += cpp.print_par()
                printed_cate.append(cat)
            else:
                continue

        for cat in list(cates_comp.keys()):
            if cat in printed_cate:
                continue
            else:
                cp = cates_comp[cat]
                for cpp in cp:
                    result_middle += cpp.print_par()
                printed_cate.append(cat)

        return result_begin + result_middle + result_end

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
            if "TNEQ" in str(par.name) or par.frozen:
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
        raise KeyError("TimingModel does not have parameter {}".format(name))

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


class ModelMeta(abc.ABCMeta):
    """Ensure timing model registration.

    When a new subclass of Component is created, record its identity in
    a class attribute ``component_types``, provided that the class has
    an attribute ``register``. This makes sure all timing model components
    are listed in ``Component.component_types``.

    """

    def __init__(cls, name, bases, dct):
        regname = "component_types"
        if "register" in dct:
            if cls.register:
                getattr(cls, regname)[name] = cls
        super(ModelMeta, cls).__init__(name, bases, dct)


@six.add_metaclass(ModelMeta)
class Component(object):
    """A base class for timing model components."""

    component_types = {}
    """An index of all registered subtypes.

    Note that classes are registered when their modules are imported,
    so ensure all classes of interest are imported before this list
    is checked.

    """

    def __init__(self):
        self.params = []
        self._parent = None
        self.deriv_funcs = {}
        self.component_special_params = []

    def __repr__(self):
        return "{}(\n    {})".format(
            self.__class__.__name__,
            ",\n    ".join(str(getattr(self, p)) for p in self.params),
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

        Return
        ------
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

    def get_prefix_mapping(self, prefix):
        """Get the index mapping for the prefix parameters.

        Parameters
        ----------
        prefix : str
           Name of prefix.

        Returns
        -------
        dict
           A dictionary with prefix pararameter real index as key and parameter
           name as value.
        """
        parnames = [x for x in self.params if x.startswith(prefix)]
        mapping = dict()
        for parname in parnames:
            par = getattr(self, parname)
            if par.is_prefix and par.prefix == prefix:
                mapping[par.index] = parname
        return mapping

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
        mapping = dict()
        for parname in parnames:
            par = getattr(self, parname)
            if par.is_prefix and par.prefix == prefix:
                mapping[par.index] = parname
        return mapping

    def match_param_aliases(self, alias):
        """Return the parameter corresponding to this alias."""
        for p in self.params:
            if p == alias:
                return p
            if alias in getattr(self, p).aliases:
                return p
        raise ValueError("{} is not recognized as a parameter or alias".format(p))

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

    def print_par(self):
        result = ""
        for p in self.params:
            result += getattr(self, p).as_parfile_line()
        return result


class DelayComponent(Component):
    def __init__(self):
        super(DelayComponent, self).__init__()
        self.delay_funcs_component = []


class PhaseComponent(Component):
    def __init__(self):
        super(PhaseComponent, self).__init__()
        self.phase_funcs_component = []
        self.phase_derivs_wrt_delay = []


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
        super(MissingParameter, self).__init__(msg)
        self.module = module
        self.param = param
        self.msg = msg

    def __str__(self):
        result = self.module + "." + self.param
        if self.msg is not None:
            result += "\n  " + self.msg
        return result
