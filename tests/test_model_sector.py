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


class test_sector:
    def setup(self):
        self.all_components = Component.component_types

    def test_init(self):
        sector = ModelSector(self.all_components['AstrometryEquatorial'])
