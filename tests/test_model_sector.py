""" Various tests for ModelSector and its interaction with TimingModel
"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy as np
import astropy.units as u
from copy import deepcopy

from pint.models.timing_model import (
    TimingModel,
    Component,
)
from pint.models.model_sector import (
    ModelSector,
    DelaySector,
)


class TestSector:
    def setup(self):
        self.all_components = Component.component_types

    def test_sector_init(self):
        sector = DelaySector(self.all_components['AstrometryEquatorial']())

        assert sector.__class__.__name__ == 'DelaySector'
        assert len(sector.component_list) == 1
        assert hasattr(sector, 'delay')
        assert hasattr(sector, 'delay_funcs')

    def test_copy(self):
        sector = DelaySector(self.all_components['AstrometryEquatorial']())
        copy_sector = deepcopy(sector)

        assert id(sector) != id(copy_sector)
