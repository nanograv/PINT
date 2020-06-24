"""Timing model objects.

Defines the basic timing model interface classes.
"""
from __future__ import absolute_import, division, print_function

import abc
import copy
from collections import defaultdict

import astropy.time as time
import astropy.units as u
import numpy as np
import six
from astropy import log

import pint
from pint.models.parameter import strParameter, maskParameter
from pint.phase import Phase
from pint.utils import (
    PrefixError,
    interesting_lines,
    lines_of,
    split_prefixed_name,
    get_component_type,
)
from pint.models.parameter import (
    AngleParameter,
    prefixParameter,
    strParameter,
    floatParameter,
)
from pint.models.model_sector import builtin_sector_map


__all__ = ["DEFAULT_ORDER", "TimingModel", "Component"]
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
        "START",
        "FINISH",
        "CLK",
        "EPHVER",
        "UNITS",
        "TIMEEPH",
        "T2CMETHOD",
        "CORRECT_TROPOSPHERE",
        "DILATEFREQ",
        "NTOA",
        "CLOCK",
        "TRES",
        "TZRMJD",
        "TZRFRQ",
        "TZRSITE",
        "NITS",
        "IBOOT",
        "BINARY",
        "CHI2R",
        "MODE",
        "INFO",
        "PLANET_SHAPIRO2",
        #    'NE_SW', 'NE_SW2',
    ]
)

ignore_prefix = set(["DMXF1_", "DMXF2_", "DMXEP_"])  # DMXEP_ for now.

DEFAULT_ORDER = [
    "astrometry",
    "jump_delay",
    "solar_system_shapiro",
    "solar_wind",
    "dispersion_constant",
    "dispersion_dmx",
    "pulsar_system",
    "frequency_dependent",
    "absolute_phase",
    "spindown",
    "phase_jump",
    "wave",
]


class TimingModel(object):
    """High level interface class for timing models and components.

    Base-level object provides an interface for implementing pulsar timing
    models. A timing model contains different model components, for example
    astrometry delays and spindown phase. All the components will be stored in
    a dictionary by category. Each category is kept as an ordered list.

    Parameters
    ----------
    name: str, optional
        The name of the timing model.
    components: list of Component, optional
        The model components for timing model. The order of the components in
        timing model will follow the order of input.

    external_sector_map: dict, optional
        The sector map for non-built-in sectors. It follows the convention,
        {'Component type': ModelSector sub-class for this component}. Default is
        '{}'. The built-in sectors are listed in the
        `timing_model.builtin_sector_map`.

    Notes
    -----
    PINT models pulsar pulse time of arrival at observer from its emission process and
    propagation to observer. Emission generally modeled as pulse 'Phase' and propagation.
    'time delay'. In pulsar timing different astrophysics phenomenons are separated to
    time model components for handling a specific emission or propagation effect.

    All timing model component classes should subclass this timing model base class.
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

    def __init__(self, name="", components=[], external_sector_map={}):
        if not isinstance(name, str):
            raise ValueError(
                "First parameter should be the model name, was {!r}".format(name)
            )
        self.model_sectors = {}
        self.name = name
        self.introduces_correlated_errors = False
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
            strParameter(name="UNITS", description="Units (TDB assumed)"), ""
        )

        for cp in components:
            self.add_component(
                cp, validate=False, external_sector_map=external_sector_map
            )

    def __repr__(self):
        return "{}(\n  {}\n)".format(
            self.__class__.__name__,
            ",\n  ".join(str(v) for k, v in sorted(self.components.items())),
        )

    def __str__(self):
        return self.as_parfile()

    def setup(self):
        """Run setup methods od all components."""
        for cp in self.components.values():
            cp.setup()

    def validate(self):
        """ Validate component setup.
            The checks includes:
            - Required parameters
            - Parameter values
        """
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
        try:
            if six.PY2:
                return super(TimingModel, self).__getattribute__(name)
            else:
                return super().__getattribute__(name)
        except AttributeError:
            # Note that there is a complex series of fallbacks that can wind up
            # here - for example if a property inadvertently raises an
            # AttributeError it looks like it's missing entirely
            errmsg = "'TimingModel' object and its component has no attribute"
            errmsg += " '%s'." % name
            if six.PY2:
                sector_methods = super(TimingModel, self).__getattribute__(
                    "sector_methods"
                )
            else:
                sector_methods = super().__getattribute__("sector_methods")

            if name in sector_methods.keys():
                sector = sector_methods[name]
                return getattr(sector, name)

            try:
                if six.PY2:
                    cp = super(TimingModel, self).__getattribute__("search_cmp_attr")(
                        name
                    )
                else:
                    cp = super().__getattribute__("search_cmp_attr")(name)
                if cp is not None:
                    return super(cp.__class__, cp).__getattribute__(name)
                else:
                    raise AttributeError(errmsg)
            except:
                raise AttributeError(errmsg)

    @property
    def sector_methods(self):
        """ Get all the sector method and reorganise it by {method_name: sector_name}
        """
        md = {}
        if six.PY2:
            sectors = super(TimingModel, self).__getattribute__("model_sectors")
        else:
            sectors = super().__getattribute__("model_sectors")
        for k, v in sectors.items():
            for method in v._methods:
                md[method] = v
        return md

    @property
    def params(self):
        """List of all parameter names in this model and all its components (order is arbitrary)."""
        p = self.top_level_params
        for cp in self.components.values():
            p = p + cp.params
        return p

    @property
    def params_ordered(self):
        """List of all parameter names in this model and all its components, in a sensible order."""

        # Define the order of components in the list
        # Any not included will be printed between the first and last set.
        start_order = ["astrometry", "spindown", "dispersion"]
        last_order = ["jump_delay"]
        compdict = self.get_components_by_category()
        used_cats = []
        pstart = copy.copy(self.top_level_params)
        for cat in start_order:
            if cat in list(compdict.keys()):
                cp = compdict[cat]
                for cpp in cp:
                    pstart += cpp.params
                used_cats.append(cat)
            else:
                continue

        pend = []
        for cat in last_order:
            if cat in list(compdict.keys()):
                cp = compdict[cat]
                for cpp in cp:
                    pend += cpp.parms
                used_cats.append(cat)
            else:
                continue

        # Now collect any components that haven't already been included in the list
        pmid = []
        for cat in list(compdict.keys()):
            if cat in used_cats:
                continue
            else:
                cp = compdict[cat]
                for cpp in cp:
                    pmid += cpp.params
                used_cats.append(cat)

        return pstart + pmid + pend

    @property
    def components(self):
        """All the components indexed by name."""
        comps = {}
        for k, v in self.model_sectors.items():
            for cp in v.component_list:
                comps[cp.__class__.__name__] = cp
        return comps

    @property
    def component_types(self):
        return self.model_sectors.keys()

    def search_cmp_attr(self, name):
        """Search for an attribute in all components.

        Return the component, or None.

        If multiple components have same attribute, it will return the first
        component.

        """
        for cp in list(self.components.values()):
            try:
                super(cp.__class__, cp).__getattribute__(name)
                return cp
            except AttributeError:
                continue

    def map_component(self, component):
        """ Get the location of component.

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
        comp_type = get_component_type(comp)
        host_list = self.model_sectors[comp_type].component_list
        order = host_list.index(comp)
        return comp, order, host_list, comp_type

    def _make_sector(self, component, external_sector_map={}):
        # Map a sector subclass from component type;
        all_sector_map = builtin_sector_map
        all_sector_map.update(external_sector_map)
        if not isinstance(component, (list, tuple)):
            components = [
                component,
            ]
        # Assuem the first component's type is the type for all other components.
        com_type = get_component_type(components[0])
        try:
            sector_cls = all_sector_map[com_type]
        except KeyError:
            raise ValueError("Can not find the Sector class for" " {}".format(com_type))
        sector = sector_cls(component)
        if sector.sector_name in self.model_sectors.keys():
            log.warn("Sector {} is already in the timing model. skip...")
            return
        common_method = set(sector._methods).intersection(self.sector_methods.keys())
        if len(common_method) != 0:
            raise ValueError(
                "Sector {}'s methods have the same method with the current "
                "sector methods. But sector methods should be unique."
                "Please check .sector_methods for the current sector"
                " methods.".format(sector.sector_name)
            )
        self.model_sectors[sector.sector_name] = sector
        # Link sector to the Timing model class.
        self.model_sectors[sector.sector_name]._parent = self

    def add_component(
        self,
        component,
        order=DEFAULT_ORDER,
        force=False,
        validate=True,
        external_sector_map={},
    ):
        """Add a component into TimingModel.

        Parameters
        ----------
        component : `pint.modle.Component` object.
            The component to be added to the timing model.
        order : list, optional.
            The component category order list. Default is the DEFAULT_ORDER.
        force : bool, optional.
            If true, add a duplicate component. Default is False.
        validate: bool, optional.
            If true, validate the component. Default is True.
        external_sector_map: dict, optional.
            Non built-in sector maps. Default is {}.
        """
        comp_type = get_component_type(component)
        print(self.model_sectors)
        if comp_type in self.model_sectors.keys():
            comp_list = self.model_sectors[comp_type].component_list
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
            self._make_sector(component, external_sector_map=external_sector_map)
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
        self.model_sectors[comp_type].component_list = new_comp_list
        # Set up components
        self.setup()
        # Validate inputs
        if validate:
            self.validate()

    def remove_component(self, component):
        """ Remove one component from the timing model.

        Parameters
        ----------
        component: str or `Component` object
            Component name or component object.
        """
        cp, co_order, host, cp_type = self.map_component(component)
        host.remove(cp)

    def _locate_param_host(self, components, param):
        """ Search for the parameter host component.

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

    def replicate(self, components=[], copy_component=False):
        raise NotImplementError()

    def get_components_by_category(self):
        """Return a dict of this model's component objects keyed by the category name"""
        categorydict = defaultdict(list)
        for cp in self.components.values():
            categorydict[cp.category].append(cp)
        # Convert from defaultdict to dict
        return dict(categorydict)

    def add_param_from_top(self, param, target_component, setup=False):
        """ Add a parameter to a timing model component.

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
                    "Can not find component '%s' in "
                    "timging model." % target_component
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
        """Report which component each parameter name comes from."""
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
            if par.is_prefix == True and par.prefix == prefix:
                mapping[par.index] = parname
        return mapping

    def param_help(self):
        """Print help lines for all available parameters in model."""
        return "".join(
            "{:<40}{}\n".format(cp, getattr(self, par).help_line())
            for par, cp in self.get_params_mapping().items()
        )

    def jump_flags_to_params(self, toas):
        """convert jump flags in toas.table["flags"] to jump parameters in the model"""
        from . import jump

        for flag_dict in toas.table["flags"]:
            if "jump" in flag_dict.keys():
                break
        else:
            log.info("No jump flags to process")
            return None
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
        self.components["PhaseJump"].setup()

    def d_phase_d_tpulsar(self, toas):
        """Return the derivative of phase wrt time at the pulsar.

        NOT implemented yet.
        """
        raise NotImplementedError()

    def designmatrix(
        self, toas, acc_delay=None, scale_by_F0=True, incfrozen=False, incoffset=True
    ):
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

        if scale_by_F0:
            mask = []
            for ii, un in enumerate(units):
                if params[ii] == "Offset":
                    continue
                units[ii] = un * u.second
                mask.append(ii)
            M[:, mask] /= F0.value
        return M, params, units, scale_by_F0

    def covariance_matrix(self, toas):
        """This a function to get the TOA covariance matrix for noise models.
           If there is no noise model component provided, a diagonal matrix with
           TOAs error as diagonal element will be returned.
        """
        ntoa = toas.ntoas
        tbl = toas.table
        result = np.zeros((ntoa, ntoa))
        # When there is no noise model.
        if "NoiseComponent" not in self.component_types:
            result += np.diag(tbl["error"].quantity.to(u.s).value ** 2)
            return result

        for nf in self.covariance_matrix_funcs:
            result += nf(toas)
        return result

    @property
    def has_correlated_errors(self):
        """Whether or not this model has correlated errors."""
        if "NoiseComponent" in self.component_types:
            for nc in self.model_sectors["NoiseComponent"].component_list:
                # recursive if necessary
                if nc.introduces_correlated_errors:
                    return True
        return False

    def scaled_sigma(self, toas):
        """This a function to get the scaled TOA uncertainties noise models.
           If there is no noise model component provided, a vector with
           TOAs error as values will be returned.
        """
        ntoa = toas.ntoas
        tbl = toas.table
        result = np.zeros(ntoa) * u.us
        # When there is no noise model.
        if "NoiseComponent" not in self.component_types:
            result += tbl["error"].quantity
            return result

        for nf in self.scaled_sigma_funcs:
            result += nf(toas)
        return result

    def compare(self, othermodel, nodmx=True):
        """Print comparison with another model

        Parameters
        ----------
        othermodel
            TimingModel object to compare to
        nodmx : bool
            If True (which is the default), don't print the DMX parameters in the comparison

        Returns
        -------
        str
            Human readable comparison, for printing
        """

        from uncertainties import ufloat
        import uncertainties.umath as um

        s = "{:14s} {:>28s} {:>28s} {:14s} {:14s}\n".format(
            "PARAMETER", "Self   ", "Other   ", "Diff_Sigma1", "Diff_Sigma2"
        )
        s += "{:14s} {:>28s} {:>28s} {:14s} {:14s}\n".format(
            "---------", "----------", "----------", "----------", "----------"
        )
        for pn in self.params_ordered:
            par = getattr(self, pn)
            if par.value is None:
                continue
            try:
                otherpar = getattr(othermodel, pn)
            except AttributeError:
                # s += "Parameter {} missing in other model\n".format(par.name)
                otherpar = None
            if isinstance(par, strParameter):
                s += "{:14s} {:>28s}".format(pn, par.value)
                if otherpar is not None:
                    s += " {:>28s}\n".format(otherpar.value)
                else:
                    s += " {:>28s}\n".format("Missing")
            elif isinstance(par, AngleParameter):
                if par.frozen:
                    # If not fitted, just print both values
                    s += "{:14s} {:>28s}".format(pn, str(par.quantity))
                    if otherpar is not None:
                        s += " {:>28s}\n".format(str(otherpar.quantity))
                    else:
                        s += " {:>28s}\n".format("Missing")
                else:
                    # If fitted, print both values with uncertainties
                    if par.units == u.hourangle:
                        uncertainty_unit = pint.hourangle_second
                    else:
                        uncertainty_unit = u.arcsec
                    s += "{:14s} {:>16s} +/- {:7.2g}".format(
                        pn,
                        str(par.quantity),
                        par.uncertainty.to(uncertainty_unit).value,
                    )
                    if otherpar is not None:
                        try:
                            s += " {:>16s} +/- {:7.2g}".format(
                                str(otherpar.quantity),
                                otherpar.uncertainty.to(uncertainty_unit).value,
                            )
                        except AttributeError:
                            # otherpar must have no uncertainty
                            if otherpar.quantity is not None:
                                s += " {:>28s}".format(str(otherpar.quantity))
                            else:
                                s += " {:>28s}".format("Missing")
                    else:
                        s += " {:>28s}".format("Missing")
                    try:
                        diff = otherpar.value - par.value
                        diff_sigma = diff / par.uncertainty.value
                        s += " {:>10.2f}".format(diff_sigma)
                        diff_sigma2 = diff / otherpar.uncertainty.value
                        s += " {:>10.2f}".format(diff_sigma2)
                    except (AttributeError, TypeError):
                        pass
                    s += "\n"
            else:
                # Assume numerical parameter
                if nodmx and pn.startswith("DMX"):
                    continue
                if par.frozen:
                    # If not fitted, just print both values
                    s += "{:14s} {:28f}".format(pn, par.value)
                    if otherpar is not None and otherpar.value is not None:
                        try:
                            s += " {:28SP}\n".format(
                                ufloat(otherpar.value, otherpar.uncertainty.value)
                            )
                        except:
                            s += " {:28f}\n".format(otherpar.value)
                    else:
                        s += " {:>28s}\n".format("Missing")
                else:
                    # If fitted, print both values with uncertainties
                    s += "{:14s} {:28SP}".format(
                        pn, ufloat(par.value, par.uncertainty.value)
                    )
                    if otherpar is not None and otherpar.value is not None:
                        try:
                            s += " {:28SP}".format(
                                ufloat(otherpar.value, otherpar.uncertainty.value)
                            )
                        except AttributeError:
                            # otherpar must have no uncertainty
                            if otherpar.value is not None:
                                s += " {:28f}".format(otherpar.value)
                            else:
                                s += " {:>28s}".format("Missing")
                    else:
                        s += " {:>28s}".format("Missing")
                    try:
                        diff = otherpar.value - par.value
                        diff_sigma = diff / par.uncertainty.value
                        s += " {:>10.2f}".format(diff_sigma)
                        diff_sigma2 = diff / otherpar.uncertainty.value
                        s += " {:>10.2f}".format(diff_sigma2)
                    except (AttributeError, TypeError):
                        pass
                    s += "\n"
        # Now print any parametrs in othermodel that were missing in self.
        mypn = self.params_ordered
        for opn in othermodel.params_ordered:
            if opn in mypn:
                continue
            if nodmx and opn.startswith("DMX"):
                continue
            try:
                otherpar = getattr(othermodel, opn)
            except AttributeError:
                otherpar = None
            s += "{:14s} {:>28s}".format(opn, "Missing")
            s += " {:>28s}".format(str(otherpar.quantity))
            s += "\n"
        return s

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
            log.info("Final object: {}".format(repr(self)))

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
        """ Validate loaded values."""
        pass

    @property
    def category(self):
        """Category is a feature the class, so delegate."""
        return self.__class__.category

    def __getattr__(self, name):
        try:
            return super(Component, self).__getattribute__(name)
        except AttributeError:
            try:
                p = super(Component, self).__getattribute__("_parent")
                if p is None:
                    raise AttributeError(
                        "'%s' object has no attribute '%s'."
                        % (self.__class__.__name__, name)
                    )
                else:
                    return self._parent.__getattr__(name)
            except:
                raise AttributeError(
                    "'%s' object has no attribute '%s'."
                    % (self.__class__.__name__, name)
                )

    @property
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

    def get_params_of_type(self, param_type):
        """ Get all the parameters in timing model for one specific type
        """
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
            if par.is_prefix == True and par.prefix == prefix:
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
            self.register_deriv_funcs(func, param.name)

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
                "Tried to remove parameter {} but it is not listed: {}".formmat(
                    param_name, self.params
                )
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
            if par.is_prefix == True and par.prefix == prefix:
                mapping[par.index] = parname
        return mapping

    def match_param_aliases(self, alias):
        # TODO need to search the parent class as well
        p_aliases = {}
        # if alias is a parameter name, return itself
        if self._parent is not None:
            search_target = self._parent
        else:
            search_target = self
        if alias in search_target.params:
            return alias
        # get all the aliases
        for p in self._parent.params:
            par = getattr(self, p)
            if par.aliases != []:
                p_aliases[p] = par.aliases
        # match alias
        for pa, pav in zip(p_aliases.keys(), p_aliases.values()):
            if alias in pav:
                return pa
        # if not found any thing.
        return ""

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
        if pn == "":
            raise ValueError("Parameter '%s' in not in the model." % param)

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

    def print_par(self,):
        result = ""
        for p in self.params:
            result += getattr(self, p).as_parfile_line()
        return result


class DelayComponent(Component):
    def __init__(self,):
        super(DelayComponent, self).__init__()
        self.delay_funcs_component = []


class PhaseComponent(Component):
    def __init__(self,):
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
