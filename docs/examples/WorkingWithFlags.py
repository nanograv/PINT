# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Working With TOA Flags
#
# ``PINT`` provides methods for working conveniently with TOA flags.  You can add, delete, and modify flags, and use them to select TOAs.
#

# %%
from pint.toa import get_TOAs
import pint.config
import pint.logging

# %%
pint.logging.setup("WARNING")

# %% [markdown]
# Get a test dataset.  This file has no flags to start with.

# %%
t = get_TOAs(pint.config.examplefile("NGC6440E.tim"), ephem="DE440")
print(t)

# %% [markdown]
# The TOAs are stored internally as an ``astropy.Table``:

# %%
print(t.table)

# %% [markdown]
# You can look at the flags directly (note that the leading ``-`` from a file has been stripped for internal storage).  The initial file did not have flags, so these (``format``, ``ddm``, and ``clkcorr``) were added by ``PINT`` when reading:

# %%
print(t["flags"])

# %% [markdown]
# Which is just looking at the ``'flags'`` column in the TOA table:

# %%
print(t.table["flags"])

# %% [markdown]
# To extract the values of a flag, you can just treat it like an array slice

# %%
print(t["ddm"])

# %% [markdown]
# However, flags are stored as strings, so you might use a function that also allows type conversions:

# %%
ddm, _ = t.get_flag_value("ddm", as_type=float)
print(ddm)

# %% [markdown]
# It's also easy to add flags.  In this case we will just add to the first 10 TOAs:

# %%
t[:10, "fish"] = "carp"
print(t["flags"])

# %% [markdown]
# If we now try to get those values:

# %%
print(t["fish"])

# %% [markdown]
# it will return the flag value or an empty string when it is not set.  On the other hand:

# %%
fish, idx = t.get_flag_value("fish")
print(fish)
print(idx)

# %% [markdown]
# Will also return an array of indices when the flag is set.
# Now let's set some other fish:

# %%
t[10:15, "fish"] = "bass"

# %% [markdown]
# and we can select only the TOAs that meet some criteria:

# %%
basstoas = t[t["fish"] == "bass"]
print(len(basstoas))

# %% [markdown]
# Which can be combined with other selection criteria:

# %%
print(t[(t["fish"] == "carp") & (t["mjd_float"] > 53679)])

# %% [markdown]
# You can set the value of a flag to an empty string, which will delete it:

# %%
t["fish"] = ""
print(t["flags"])

# %%
