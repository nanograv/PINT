import os
import pytest
import io

import astropy.units as u
import numpy as np

from pint.models import get_model, get_model_and_toas
import pint.toa as toa
from pinttestdata import datadir


def test_DDS_delay():
    """Make a copy of a DD model and switch to DDS"""
    parfileB1855 = os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_modified.par")
    timB1855 = os.path.join(datadir, "B1855+09_NANOGrav_dfg+12.tim")
    t = toa.get_TOAs(timB1855, ephem="DE405", planets=False, include_bipm=False)
    mDD = get_model(parfileB1855)
    with open(parfileB1855) as f:
        lines = f.readlines()
    outlines = ""
    for line in lines:
        if not (line.startswith("SINI") or line.startswith("BINARY")):
            outlines += f"{line}"
        else:
            if line.startswith("SINI"):
                d = line.split()
                sini = float(d[1])
                shapmax = -np.log(1 - sini)
                outlines += f"SHAPMAX {shapmax}\n"
            elif line.startswith("BINARY"):
                outlines += "BINARY DDS\n"
    mDDS = get_model(io.StringIO(outlines))
    DD_delay = mDD.binarymodel_delay(t, None)
    DDS_delay = mDDS.binarymodel_delay(t, None)
    assert np.allclose(DD_delay, DDS_delay)
