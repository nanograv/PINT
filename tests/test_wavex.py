from io import StringIO
import pytest
import numpy as np

from astropy import units as u
from pint.models import get_model, get_model_and_toas
from pint.fitter import Fitter
from pint.toa import get_TOAs
from pint.simulation import make_fake_toas_uniform
import pint.utils
from pinttestdata import datadir

par1 = """
    PSR              B1937+21
    LAMBDA   301.9732445337270
    BETA      42.2967523367957
    PMLAMBDA           -0.0175
    PMBETA             -0.3971
    PX                  0.1515
    POSEPOCH        55321.0000
    F0    641.9282333345536244  1  0.0000000000000132
    F1     -4.330899370129D-14  1  2.149749089617D-22
    PEPOCH        55321.000000
    DM               71.016633
    UNITS                  TDB
    """

# Introduce a par file with WaveX already present

par2 = """
    PSR              B1937+21
    LAMBDA   301.9732445337270
    BETA      42.2967523367957
    PMLAMBDA           -0.0175
    PMBETA             -0.3971
    PX                  0.1515
    POSEPOCH        55321.0000
    F0    641.9282333345536244  1  0.0000000000000132
    F1     -4.330899370129D-14  1  2.149749089617D-22
    PEPOCH        55321.000000
    DM               71.016633
    WXEPOCH       55321.000000
    WXFREQ_0001            0.1
    WXSIN_0001              1
    WXCOS_0001              1 
    UNITS                  TDB
    """
