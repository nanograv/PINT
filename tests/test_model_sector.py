""" Various tests for ModelSector and its interaction with TimingModel
"""
from __future__ import absolute_import, division, print_function

import pytest
import os
import numpy as np
import astropy.units as u

from pint.models.timing_model import (
    TimingModel,
    ModelSector,
    Component
)


class TestSector:
    def setup(self):
        self.all_components = Component.component_types

    def test_sector_init(self):
        sector = ModelSector(self.all_components['AstrometryEquatorial']())
        
        assert sector.__class__.__name__ == 'DelaySector'
        assert len(sector.component_list) == 1

     
