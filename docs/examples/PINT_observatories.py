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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PINT Observatories
# Basic loading and use of observatories in PINT, including loading custom observatories.
#
# PINT needs to know where telescopes are, what clock corrections are necessary, and
# a bunch of other information in order to correctly process TOAs.  In many cases this will be seemlessly handled
# when you load in a set of TOAs.  But if you want to look at how/where the observatories are defined or add your own, this is the place to learn.


# %%
# import the library
import pint.observatory

# %% [markdown]
# What observatories are present in PINT?  How can we identify them?  They have both default names and aliases.

# %%
for name, aliases in pint.observatory.Observatory.names_and_aliases().items():
    print(f"Observatory '{name}' is also known as {aliases}")

# %% [markdown]
# Let's get the GBT.  When we `print`, we find out some basic info about the observatory name/aliases, location, and and other info present (e.g., where the data are from):

# %%
gbt = pint.observatory.get_observatory("gbt")
print(gbt)

# %% [markdown]
# The observatory also includes info on things like the clock file:

# %%
print(f"GBT clock file is named '{gbt.clock_files}'")

# %% [markdown]
# Some special locations are also present, like the solar system barycenter.  You can access explicitly through the `pint.observatory.special_locations` module, but if you just try to get one it will automatically import what is needed

# %%
ssb = pint.observatory.get_observatory("ssb")

# %% [markdown]
# If you want to know where the observatories are defined, you can find that too:

# %%
print(
    f"Observatory definitions are in '{pint.observatory.topo_obs.observatories_json}'"
)

# %% [markdown]
# That is the default location, although you can overwrite those definitions by setting `$PINT_CLOCK_OVERRIDE`.  You can also define a new observatory andn load it in.  We use `JSON` to do this:

# %%
# We want to create a file-like object containing the new definition.  So defined as a string, and then use StringIO
import io

notthegbt = r"""
    {
        "notgbt": {
        "tempo_code": "1",
        "itoa_code": "GB",
        "clock_file": "time_gbt.dat",
        "itrf_xyz": [
            882589.289,
            4924872.368,
            3943729.418
        ],
        "origin": "The Robert C. Byrd Green Bank Telescope, except with one coordinate changed.\nThis data was obtained by Joe Swiggum from Ryan Lynch in 2021 September.\n"
        }
    }
    """
pint.observatory.topo_obs.load_observatories(io.StringIO(notthegbt))

# %% [markdown]
# If we had defined the GBT again, it would have complained unless we used `overwrite=True`.  But since this has a new name it's OK.  Now let's try to use it:

# %%
notgbt = pint.observatory.get_observatory("notgbt")
print(notgbt)
