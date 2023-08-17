#!/usr/bin/env python

import pint.toa

# Get .tim file from here:
# curl -O https://data.nanograv.org/static/data/J0740+6620.cfr+19.tim

# This will load the TOAs, compute the positions of the Earth and planets, and apply clock corrections and build the table.
thanktoas = pint.toa.get_TOAs(
    "J0740+6620.cfr+19.tim",
    ephem="DE436",
    planets=True,
    usepickle=False,
    include_gps=True,
    bipm_version="BIPM2015",
    include_bipm=True,
)
print()
print(f"Number of TOAs: {str(thanktoas.ntoas)}")
print()
