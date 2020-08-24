""" Test for pint solar wind module
"""

import os
import pytest
import numpy as np
import copy
import sys

import astropy.units as u
from pint.models import get_model
from pint.fitter import WidebandTOAFitter
from pint.toa import get_TOAs
from pinttestdata import datadir

os.chdir(datadir)


class TestSolarWind:
    def test_solar_wind_dm(self):
        self.par = ""
        self.tim = ""
