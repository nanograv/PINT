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
# # Check the state of PINT's clock corrections
#
# In order to do precision pulsar timing, it is necessary to know how the observatory clocks differ from a global time standard so that TOAs can be corrected. This requires PINT to have access to a record of measured differences. This record needs to be updated when new data is available. This notebook demonstrates how you can check the status of the clock corrections in your version of PINT. The version in the documentation also records the state of the PINT distribution at the moment the documentation was generated (which is when the code was last changed).

# %%
import tempfile
from glob import glob

import pint.observatory
import pint.observatory.topo_obs
import pint.logging

# hide annoying INFO messages?
pint.logging.setup("WARNING")

# %%
pint.observatory.list_last_correction_mjds()

# %% [markdown]
# Let's export the clock corrections as they currently stand so we can save
# these exact versions for reproducibility purposes.
#
# %%
d = tempfile.mkdtemp()
pint.observatory.topo_obs.export_all_clock_files(d)
for f in sorted(glob(f"{d}/*")):
    print(f)
