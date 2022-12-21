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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Understanding Timing Models

# %% [markdown]
# ## Build a timing model starting from a par file

# %%
from pint.models import get_model
from pint.models.timing_model import Component
import pint.config
import pint.logging

# setup logging
pint.logging.setup(level="INFO")

# %% [markdown]
# One can build a timing model via `get_model()` method. This will read the par file and instantiate all the delay and phase components, using the default ordering.

# %%
par = "B1855+09_NANOGrav_dfg+12_TAI.par"
m = get_model(pint.config.examplefile(par))


# %% [markdown]
# Each of the parameters in the model can be accessed as an attribute of the `TimingModel` object.
# Behind the scenes PINT figures out which component the parameter is stored in.
#
# Each parameter has attributes like the quantity (which includes units), and a description (see the Understanding Parameters notebook for more detail)

# %%
print(m.F0.quantity)
print(m.F0.description)

# %% [markdown]
# We can now explore the structure of the model

# %%
# This gives a list of all of the component types (so far there are only delay and phase components)
m.component_types

# %%
dir(m)

# %% [markdown]
# The TimingModel class stores lists of the delay model components and phase components that make up the model

# %%
# When this list gets printed, it shows the parameters that are associated with each component as well.
m.DelayComponent_list

# %%
# Now let's look at the phase components. These include the absolute phase, the spindown model, and phase jumps
m.PhaseComponent_list

# %% [markdown]
# We can add a component to an existing model

# %%
from pint.models.astrometry import AstrometryEcliptic

# %%
a = AstrometryEcliptic()  # init the AstrometryEcliptic instance

# %%
# Add the component to the model
# It will be put in the default order
# We set validate=False since we have not set the parameter values yet, which would cause validate to fail
m.add_component(a, validate=False)

# %%
m.DelayComponent_list  # The new instance is added to delay component list

# %% [markdown]
# There are two ways to remove a component from a model. This simplest is to use the `remove_component` method to remove it by name.

# %%
# We will not do this here, since we'll demonstrate a different method below.
# m.remove_component("AstrometryEcliptic")

# %% [markdown]
# Alternatively, you can have more control using the `map_component()` method, which takes either a string with component name,
# or a Component instance and returns a tuple containing the Component instance, its order in the relevant component list,
# the list of components of this type in the model, and the component type (as a string)

# %%
component, order, from_list, comp_type = m.map_component("AstrometryEcliptic")
print("Component : ", component)
print("Type      : ", comp_type)
print("Order     : ", order)
print("List      : ")
_ = [print(c) for c in from_list]


# %%
# Now we can remove the component by directly manipulating the list
from_list.remove(component)

# %%
m.DelayComponent_list  # AstrometryEcliptic has been removed from delay list.

# %% [markdown]
# To switch the order of a component, just change the order of the component list
#
# **NB: that this should almost never be done!  In most cases the default order of the delay components is correct. Experts only!**

# %%
# Let's look at the order of the components in the delay list first
_ = [print(dc.__class__) for dc in m.DelayComponent_list]

# %%
# Now let's swap the order of DispersionDMX and Dispersion
component, order, from_list, comp_type = m.map_component("DispersionDMX")
new_order = 3
from_list[order], from_list[new_order] = from_list[new_order], from_list[order]

# %%
# Print the classes to see the order switch
_ = [print(dc.__class__) for dc in m.DelayComponent_list]

# %% [markdown]
# Delays are always computed in the order of the DelayComponent_list

# %%
# First get the toas
from pint.toa import get_TOAs

t = get_TOAs(pint.config.examplefile("B1855+09_NANOGrav_dfg+12.tim"), model=m)

# %%
# compute the total delay
total_delay = m.delay(t)
total_delay

# %% [markdown]
# One can get the delay up to some component. For example, if you want the delay computation stop after the Solar System Shapiro delay.
#
# By default the delay of the specified component *is* included. This can be changed by the keyword parameter `include_last=False`.

# %%
to_jump_delay = m.delay(t, cutoff_component="SolarSystemShapiro")
to_jump_delay

# %% [markdown]
# Here is a list of all the Component types that PINT knows about

# %%
Component.component_types

# %% [markdown]
# When PINT builds a model from a par file, it has to infer what components to include in the model.
# This is done by the `component_special_params` of each `Component`. A component will be instantiated
# when one of its special parameters is present in the par file.

# %%
from collections import defaultdict

special = defaultdict(list)
for comp, tp in Component.component_types.items():
    for p in tp().component_special_params:
        special[p].append(comp)


special
