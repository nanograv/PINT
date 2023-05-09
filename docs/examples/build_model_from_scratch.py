# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: .env
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Building a timing model from scratch
#
# This example includes:
#  * Constructing a timing model object from scratch
#  * Adding and deleting components
#  * Assigning parameter values
#  * Adding prefix-able parameters

# %%
import astropy.units as u  # Astropy units is a very useful module.
import pint.logging

try:
    from IPython.display import display
except ImportError:
    # Older IPython
    from IPython.core.display_functions import display

# setup logging
pint.logging.setup(level="INFO")
from pint.models import (
    parameter as p,
)  # We would like to add parameters to the model, so we need parameter module.
from pint.models.timing_model import (
    TimingModel,
    Component,
)  # Interface for timing model
import pint
from astropy.time import Time  # PINT uses astropy Time objects to represent times

# %% [markdown]
# Typically, timing models are built by reading a par file with the `get_model()` function, but it is possible to construct them entirely programmatically from scratch. Also, once you have a `TimingModel` object (no matter how you built it), you can modify it by adding or removing parameters or entire components. This example show how this is done.
#
# We are going to build the model for "NGC6440E.par" from scratch
#
# ### First let us see all the possible components we can use
#
# All built-in component classes can be viewed from `Component` class, which uses the meta-class to collect the built-in component class. For how to make a component class, see example "make_component_class" (in preparation)

# %%
# list all the existing components
# all_components is a dictionary, with the component name as the key and component class as the value.
all_components = Component.component_types
# Print the component class names.
_ = [print(x) for x in all_components]  # The "_ =" just suppresses excess output

# %% [markdown]
# ### Choose your components
#
# Let's start from a relatively simple model, with
# `AbsPhase`: The absolute phase of the pulsar, typical parameters, `TZRMJD`, `TZRFREQ`...
# `AstrometryEquatorial`: The ICRS equatorial coordinate, parameters, `RAJ`, `DECJ`, `PMRA`, `PMDEC`...
# `Spindown`: The pulsar spin-down model, parameters, `F0`, `F1`...
#
# We will add a dispersion model as a demo.

# %%
selected_components = ["AbsPhase", "AstrometryEquatorial", "Spindown"]
component_instances = []

# Initiate the component instances
for cp_name in selected_components:
    component_class = all_components[cp_name]  # Get the component class
    component_instance = component_class()  # Instantiate a component object
    component_instances.append(component_instance)


# %% [markdown]
# ### Construct timing model (i.e., `TimingModel` instance)
#
# `TimingModel` class provides the storage and interface for the components. It also manages the components internally.

# %%
# Construct timing model instance, given a name and a list of components to include (that we just created above)
tm = TimingModel("NGC6400E", component_instances)

# %% [markdown]
# ### View the components in the timing model instance
#
# To view all the components in `TimingModel` instance, we can use the property `.components`, which returns a dictionary (name as the key, component instance as the value).
#
# Internally, the components are stored in a list(ordered list, you will see why this is important below) according to their types. All the delay type of components (subclasses of `DelayComponent` class) are stored in the `DelayComponent_list`, and the phase type of components (subclasses of `PhaseComponent` class) in the `PhaseComponent_list`.

# %%
# print the components in the timing model
for cp_name, cp_instance in tm.components.items():
    print(cp_name, cp_instance)

# %% [markdown]
# ### Useful methods of `TimingModel`
#
# * `TimingModel.components()`           : List all the components in the timing model.
# * `TimingModel.add_component()`        : Add component into the timing model.
# * `TimingModel.remove_component()`     : Remove a component from the timing model.
# * `TimingModel.params()`               : List all the parameters in the timing model from all components.
# * `TimingModel.setup()`                : Setup the components (e.g., register the derivatives).
# * `TimingModel.validate()`             : Validate the components see if the parameters are setup properly.
# * `TimingModel.delay()`                : Compute the total delay.
# * `TimingModel.phase()`                : Compute the total phase.
# * `TimingModel.delay_funcs()`          : List all the delay functions from all the delay components.
# * `TimingModel.phase_funcs()`          : List all the phase functions from all the phase components.
# * `TimingModel.get_component_type()`   : Get all the components from one category
# * `TimingModel.map_component()`        : Get the component location. It returns the component's instance, order in                                          the list, host list and its type.
# * `TimingModel.get_params_mapping()`   : Report which component each parameter comes from.
# * `TimingModel.get_prefix_mapping()`   : Get the index mapping for one prefix parameters.
# * `TimingModel.param_help()`           : Print the help line for all available parameters.
#

# %% [markdown]
# ### Component order
#
# Since the times that are given to a delay component include all the delays from the previously-evaluted delay components, the order of delay components is important. For example, the solar system delays need to be applied to get to barycentric time, which is needed to evaluate the binary delays, then the binary delays must be applied to get to pulsar proper time.
#
# PINT provides a default ordering for the components. In most cases this should be correct, but can be modified by expert users for a particular purpose.
#
# Here is the default order:

# %%
from pint.models.timing_model import DEFAULT_ORDER

_ = [print(order) for order in DEFAULT_ORDER]

# %% [markdown]
# ### Add parameter values
#
# Initially, the parameters have no values or the default values, so we must add them before validating the model.
#
# Please note, PINT's convention for fitting flag is defined in the `Parameter.frozen` attribute. `Parameter.frozen = True` means "do **not** fit this parameter". This is the opposite of TEMPO/TEMPO2 .par file flag where "1" means the parameter is fitted.

# %%
# We build a dictionary with a key for each parameter we want to set.
# The dictionary entries can be either
#  {'pulsar name': (parameter value, TEMPO_Fit_flag, uncertainty)} akin to a TEMPO par file form
# or
# {'pulsar name': (parameter value, )} for parameters that can't be fit
# NOTE: The values here are assumed to be in the default units for each parameter
# Notice that we assign values with units, and pint defines a special hourangle_second unit that can be use for
# right ascensions.  Also, angles can be specified as strings that will be parsed by astropy.
params = {
    "PSR": ("1748-2021E",),
    "RAJ": ("17:48:52.75", 1, 0.05 * pint.hourangle_second),
    "DECJ": ("-20:21:29.0", 1, 0.4 * u.arcsec),
    "F0": (61.485476554 * u.Hz, 1, 5e-10 * u.Hz),
    "PEPOCH": (Time(53750.000000, format="mjd", scale="tdb"),),
    "POSEPOCH": (Time(53750.000000, format="mjd", scale="tdb"),),
    "TZRMJD": (Time(53801.38605120074849, format="mjd", scale="tdb"),),
    "TZRFRQ": (1949.609 * u.MHz,),
    "TZRSITE": (1,),
}

# Assign the parameters
for name, info in params.items():
    par = getattr(tm, name)  # Get parameter object from name
    par.quantity = info[0]  # set parameter value
    if len(info) > 1:
        if info[1] == 1:
            par.frozen = False  # Frozen means not fit.
        par.uncertainty = info[2]
# %% [markdown]
# ### Set up and Validating the model
#
# Setting up the model builds the necessary model attributes, and validating model checks if there is any
# important parameter values missing, and if the parameters are assigned correctly. If there is anything
# not assigned correctly, it will raise an exception.
# %%
tm.setup()
tm.validate()
# You should see all the assigned parameters.
# Printing a TimingModel object shows the parfile representation
print(tm)
# %% [markdown]
# The validate function is also integrated into the add_component() function. When adding a component it will validate the timing model by default; however, it can be switched off by setting flag `validate=False`. We will use this flag in the next section.

# %% [markdown]
# ### Add a component to the timing model
#
# We will add the dispersion component to the timing model. The steps are:
# 1. Instantiate the Dispersion class
# 2. Add dispersion instance into the timing model, with validation as False.
#     <p> Since the dispersion model's parameter have not set yet, validation would fail. We will validate it after the parameters filled in.</p>
# 3. Add parameters and set values
# 4. Validate the timing model.

# %%
dispersion_class = all_components["DispersionDM"]
dispersion = dispersion_class()  # Make the dispersion instance.

# Using validate=False here allows a component being added first and validate later.
tm.add_component(dispersion, validate=False)
# %% [markdown]
# Let us examine the components in the timing model.

# %%
# print the components out, DispersionDM should be there.
print("All components in timing model:")
display(tm.components)

print("\n")
print("Delay components in the DelayComponent_list (order matters!):")

# print the delay component order, dispersion should be after the astrometry
display(tm.DelayComponent_list)

# %% [markdown]
# The DM value can be set as we set the parameters above.

# %%
tm.DM.quantity = 223.9 * u.pc / u.cm**3
tm.DM.frozen = False  # Frozen means not fit.
tm.DM.uncertainty = 0.3 * u.pc / u.cm**3

# %% [markdown]
# Run validate again and just make sure everything is setup good.

# %%
tm.validate()  # If this fails, that means the DM model was not setup correctly.

# %% [markdown]
# Now the dispersion model component is added and you are now set for your analysis.

# %% [markdown]
# ### Delete a component
#
# Deleting a component will remove the component from component list.

# %%
# Remove by name
tm.remove_component("DispersionDM")
# %%
display(tm.components)

# %% [markdown]
# Dispersion model should disappear from the timing model.

# %% [markdown]
# ### Add prefix-style parameters
#
# Prefix style parameters are used in certain models (e.g., DMX_nnnn or Fn).
#
# Let us use the `DispersionDMX` model to demonstrate how it works.

# %%
tm.add_component(all_components["DispersionDMX"]())
# %%
_ = [print(cp) for cp in tm.components]
# "DispersionDMX" should be there.

# %% [markdown]
# ### Display the existing DMX parameters
#
# What do we have in DMX model.
#
# Note, `Component` class also has the attribute `params`, which is only for the parameters in the component.

# %%
print(tm.components["DispersionDMX"].params)

# %% [markdown]
# ### Add DMX parameters
#
# Since we already have DMX_0001, we will add DMX_0003, just want to show that for DMX model, DMX index('0001' part) does not have to follow the consecutive order.
#
# The prefix type of parameters have to use `prefixParameter` class from `pint.models.parameter` module.
# %%
# Add prefix parameters
dmx_0003 = p.prefixParameter(
    parameter_type="float", name="DMX_0003", value=None, units=u.pc / u.cm**3
)

tm.components["DispersionDMX"].add_param(dmx_0003, setup=True)
# tm.add_param_from_top(dmx_0003, "DispersionDMX", setup=True)
# # Component should given by its name string. use setup=True make sure new parameter get registered.

# %% [markdown]
# ### Check if the parameter and component setup correctly.

# %%
display(tm.params)
display(tm.delay_deriv_funcs.keys())  # the derivative function should be added.

# %% [markdown]
# However only adding DMX_0003 is not enough, since one DMX parameter also need a DMX range, `DMXR1_0003`, `DMXR2_0003` in this case. Without them, the validation will fail. So let us add them as well.

# %%
dmxr1_0003 = p.prefixParameter(
    parameter_type="mjd", name="DMXR1_0003", value=None, units=u.day
)  # DMXR1 is a type of MJD parameter internally.
dmxr2_0003 = p.prefixParameter(
    parameter_type="mjd", name="DMXR2_0003", value=None, units=u.day
)  # DMXR1 is a type of MJD parameter internally.

tm.components["DispersionDMX"].add_param(dmxr1_0003, setup=True)
tm.components["DispersionDMX"].add_param(dmxr2_0003, setup=True)


# %%
tm.params

# %% [markdown]
# Then validate it.

# %%
tm.validate()

# %% [markdown]
# ### Remove a parameter
#
# Remove a parameter is just use the `remove_param()` function.

# %%
tm.remove_param("DMX_0003")
tm.remove_param("DMXR1_0003")
tm.remove_param("DMXR2_0003")
display(tm.params)
# %% [markdown]
# ### Add higher order derivatives of spin frequency to timing model
#
# Adding higher order derivatives of spin frequency (e.g., F2, F3, F4) is a common use case. `Fn` is a prefixParameter, but unlike the `DMX_` parameters, all indexes up to the maximum order must exist, since it represents the coefficients of a Taylor expansion.
#
# Let us list the current spindown model parameters:

# %%
display(tm.components["Spindown"].params)

# %% [markdown]
# Let us add `F1` and `F2` to the model. Both `F1` and `F2` needs a very high
# precision, we use longdouble=True flag to specify the `F2` value to be a longdouble type.
#
# Note, if we add `F3` directly without `F2`, the validation will fail.

# %%
f1 = p.prefixParameter(
    parameter_type="float", name="F1", value=0.0, units=u.Hz / (u.s), longdouble=True
)

f2 = p.prefixParameter(
    parameter_type="float",
    name="F2",
    value=0.0,
    units=u.Hz / (u.s) ** 2,
    longdouble=True,
)

# %%
tm.components["Spindown"].add_param(f1, setup=True)
tm.components["Spindown"].add_param(f2, setup=True)

# %%
tm.validate()

# %%
display(tm.params)

# %% [markdown]
# Now `F2` can be used in the timing model.

# %%
tm.F1.quantity = -1.181e-15 * u.Hz / u.s
tm.F1.uncertainty = 1e-18 * u.Hz / u.s
tm.F2.quantity = 2e-10 * u.Hz / u.s**2
display(tm.F2)

# %%
