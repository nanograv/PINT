# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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
# # Understanding Parameters

# %% {"jupyter": {"outputs_hidden": false}}
import pint.models
import pint.models.parameter as pp
import astropy.units as u
from astropy.time import Time
import pint.config
import pint.logging

pint.logging.setup(level="INFO")

# %% {"jupyter": {"outputs_hidden": false}}
# Load a model to play with
model = pint.models.get_model(
    pint.config.examplefile("B1855+09_NANOGrav_dfg+12_TAI.par")
)

# %% {"jupyter": {"outputs_hidden": false}}
# This model has a large number of parameters of various types
model.params

# %% [markdown]
# ## Attributes of Parameters
#
# Each parameter has attributes that specify the name and type of the parameter, its units, and the uncertainty.
# The `par.quantity` and `par.uncertainty` are both astropy quantities with units. If you need the bare values,
# access `par.value` and `par.uncertainty_value`, which will be numerical values in the units of `par.units`
#
# Let's look at those for each of the types of parameters in this model.

# %% {"jupyter": {"outputs_hidden": false}}
printed = []
for p in model.params:
    par = getattr(model, p)
    if type(par) in printed:
        continue
    print("Name           ", par.name)
    print("Type           ", type(par))
    print("Quantity       ", par.quantity, type(par.quantity))
    print("Value          ", par.value)
    print("units          ", par.units)
    print("Uncertainty    ", par.uncertainty)
    print("Uncertainty_value", par.uncertainty_value)
    print("Summary        ", par)
    print("Parfile Style  ", par.as_parfile_line())
    print()
    printed.append(type(par))

# %% [markdown]
# Note that DMX_nnnn is an example of a `prefixParameter`. These are parameters that are indexed by a numerical value and a componenent can have an arbitrary number of them.
# In some cases, like `Fn` they are coefficients of a Taylor expansion and so all indices up to the maximum must be present. For others, like `DMX_nnnn` some indices can be missing without a problem.
#
# `prefixParameter`s can be used to hold indexed parameters of various types ( float, bool, str, MJD, angle ). Each one will instantiate a parameter of that type as `par.param_comp`.
# When you print the parameter it looks like the `param_comp` type.

# %%
# Note that for each instance of a prefix parameter is of type `prefixParameter`
print("Type = ", type(model.DMX_0016))
print("param_comp type = ", type(model.DMX_0016.param_comp))
print("Printing gives : ", model.DMX_0016)

# %% [markdown]
# ## Constructing a parameter
#
# You can make a Parameter instance by calling its constructor

# %% {"jupyter": {"outputs_hidden": false}}
# You can specify the vaue as a number
t = pp.floatParameter(name="TEST", value=100, units="Hz", uncertainty=0.03)
print(t)

# %% {"jupyter": {"outputs_hidden": false}}
# Or as a string that will be parsed
t2 = pp.floatParameter(name="TEST", value="200", units="Hz", uncertainty=".04")
print(t2)

# %% {"jupyter": {"outputs_hidden": false}}
# Or as an astropy Quantity with units (this is the preferred method!)
t3 = pp.floatParameter(
    name="TEST", value=0.3 * u.kHz, units="Hz", uncertainty=4e-5 * u.kHz
)
print(t3)
print(t3.quantity)
print(t3.value)
print(t3.uncertainty)
print(t3.uncertainty_value)

# %% [markdown]
# ## Setting Parameters
#
# The value of a parameter can be set in multiple ways. As usual, the preferred method is to set it using an astropy Quantity, so units will be checked and respected

# %% {"jupyter": {"outputs_hidden": false}}
par = model.F0
# Here we set it using a Quantity in kHz. Because astropy Quantities are used, it does the right thing!
par.quantity = 0.3 * u.kHz
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)

# %% {"jupyter": {"outputs_hidden": false}}
# Here we set it with a bare number, which is interpreted as being in the units `par.units`
print(par)
par.quantity = 200
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)

# %% {"jupyter": {"outputs_hidden": false}}
# If you try to set the parameter to a quantity that isn't compatible with the units, it raises an exception
try:
    print(par)
    par.value = 100 * u.second  # SET F0 to seconds as time.
    print("Quantity       ", par.quantity, type(par.quantity))
    print("Value          ", par.value)
    print(par)
except u.UnitConversionError as e:
    print("Exception raised:", e)
else:
    raise ValueError("That was supposed to raise an exception!")

# %% [markdown]
# ### MJD parameters
#
# These parameters hold a date as an astropy `Time` object. Numbers will be interpreted as MJDs in the default time scale of the parameter (which is UTC for the TZRMJD parameter)

# %% {"jupyter": {"outputs_hidden": false}}
par = model.TZRMJD
print(par)
par.quantity = 54000
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity

# %% {"jupyter": {"outputs_hidden": false}}
# And of course, you can set them with a `Time` object
par.quantity = Time.now()
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity

# %%
# I wonder if this should get converted to UTC?
par.quantity = Time(58000.0, format="mjd", scale="tdb")
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity

# %% [markdown]
# ### AngleParameters
#
# These store quanities as angles using astropy coordinates

# %% {"jupyter": {"outputs_hidden": false}}
# The unit for RAJ is hourangle
par = model.RAJ
print(par)
par.quantity = 12
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)

# %% {"jupyter": {"outputs_hidden": false}}
# Best practice is to set using a quantity with units
print(par)
par.quantity = 30.5 * u.hourangle
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity

# %% {"jupyter": {"outputs_hidden": false}}
# But a string will work
par.quantity = "20:30:00"
print("Quantity       ", par.quantity, type(par.quantity))
print("Value          ", par.value)
print(par)
par.quantity

# %% {"jupyter": {"outputs_hidden": false}}
# And the units can be anything that is convertable to hourangle
print(par)
par.quantity = 30 * u.deg
print("Quantity       ", par.quantity, type(par.quantity))
print("Quantity in deg", par.quantity.to(u.deg))
print("Value          ", par.value)
print(par)
par.quantity

# %% {"jupyter": {"outputs_hidden": false}}
# Here, setting RAJ to an incompatible unit will raise an exception
try:
    # Example for RAJ
    print(par)
    par.quantity = 30 * u.hour  # Here hour is in the unit of time, not hourangle
    print("Quantity       ", par.quantity, type(par.quantity))
    print(par)
    par.quantity
except u.UnitConversionError as e:
    print("Exception raised:", e)
else:
    raise ValueError("That was supposed to raise an exception!")
