# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Check the state of PINT's clock corrections
#
# In order to do precision pulsar timing, it is necessary to know how the observatory clocks differ from a global time standard so that TOAs can be corrected. This requires PINT to have access to a record of measured differences. This record needs to be updated when new data are available. This notebook demonstrates how you can check the status of the clock corrections in your version of PINT. The version in the documentation also records the state of the PINT distribution at the moment the documentation was generated (which is when the code was last changed).

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

# %% [markdown]
# # Understand Observatory Clock Corrections
#
# Observatory objects hold information needed for PINT to convert site arrival times (SAT) to UTC. The first is the clock correction file as described above.  For most observatories, these files correct the times from UTC(observatory) to UTC(GPS) since the corrections are determined using a GPS time standard at the site.  To convert UTC(GPS) to UTC, the corrections from [BIPM Circular T](https://www.bipm.org/en/time-ftp/circular-t) must be applied. PINT gets these corrections from the file gps2utc.clk.

# %%
site = pint.observatory.get_observatory("gbt")

# %%
# This lists the clock correction files that will be applied to TOAs from this site
site.clock_files

# %%
# This boolean indicate whether the UTC(GPS) to UTC correction will be applied to TOAs from this site
site.apply_gps2utc

# %% [markdown]
# Note these should be considered immutable properties of the site in general. However, if for testing or other purposes you want to disable the GPS correction, you can overwrite the observatory in the registry with a modified one, as follows. But this is **not recommended** in most cases because it modifies the observatory registry and will affect all TOAs that use that observatory in the current python session (see issue #[1708](https://github.com/nanograv/PINT/issues/1708)).

# %%
site = pint.observatory.get_observatory("gbt", apply_gps2utc=False)
site.apply_gps2utc

# %%
# Note how this will be the case going forward, even if you don't specify that kwarg,
# because the site is pulled from the Observatory registry (i.e. the change "sticks")
site2 = pint.observatory.get_observatory("gbt")
site2.apply_gps2utc

# %%
