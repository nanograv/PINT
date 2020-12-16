"""Various tests for the maskParameter"""

import os
import unittest
import pytest
from io import StringIO
import astropy.time as time
import astropy.units as u
import numpy as np

from pint.models.model_builder import get_model
from pint.models.parameter import maskParameter
from pint.toa import get_TOAs
from pinttestdata import datadir

import copy


class TestParameters(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        os.chdir(datadir)
        self.toas = get_TOAs("J1713+0747_NANOGrav_11yv0_short.tim")

    def test_mjd_mask(self):
        mp = maskParameter("test1", key="mjd", key_value=[54000, 54100])
        assert mp.key == "mjd"
        assert mp.key_value == [54000, 54100]
        assert mp.value == None
        select_toas = mp.select_toa_mask(self.toas)
        assert np.all(self.toas.table["mjd_float"][select_toas] <= 54100)
        assert np.all(self.toas.table["mjd_float"][select_toas] >= 54000)
        mp_str_keyval = maskParameter("test2", key="mjd", key_value=["54000", "54100"])
        assert mp_str_keyval.key_value == [54000, 54100]
        mp_value_switch = maskParameter("test1", key="mjd", key_value=[54100, 54000])
        assert mp_value_switch.key_value == [54000, 54100]
        with pytest.raises(ValueError):
            mp_str_keyval = maskParameter("test2", key="mjd", key_value=["54000"])

    def test_freq_mask(self):
        mp = maskParameter("test2", key="freq", key_value=[1400, 1430])
        assert mp.key_value == [1400 * u.MHz, 1430 * u.MHz]
        select_toas = mp.select_toa_mask(self.toas)
        assert np.all(self.toas.table["freq"][select_toas] <= 1430 * u.MHz)
        assert np.all(self.toas.table["freq"][select_toas] >= 1400 * u.MHz)
        mp_str = maskParameter("test2", key="freq", key_value=["1400", "2000"])
        assert mp_str.key_value == [1400 * u.MHz, 2000 * u.MHz]
        mp_switch = maskParameter("test2", key="freq", key_value=[2000, 1400])
        assert mp_switch.key_value == [1400 * u.MHz, 2000 * u.MHz]
        mp_quantity = maskParameter(
            "test2", key="freq", key_value=[2000 * u.MHz, 1400 * u.MHz]
        )
        assert mp_quantity.key_value == [1400 * u.MHz, 2000 * u.MHz]
        with pytest.raises(ValueError):
            mp_wrong_keyvalue = maskParameter("test2", key="freq", key_value=[1400])

    def test_tel_mask(self):
        mp_ao = maskParameter("test2", key="tel", key_value="ao")
        assert mp_ao.key_value == ["arecibo"]
        select_toas = mp_ao.select_toa_mask(self.toas)
        print(self.toas.table["obs"][select_toas])
        assert np.all(self.toas.table["obs"][select_toas] == "arecibo")
        mp_gbt = maskParameter("test2", key="tel", key_value=["gbt"])
        assert mp_gbt.key_value == ["gbt"]
        select_toas = mp_gbt.select_toa_mask(self.toas)
        assert np.all(self.toas.table["obs"][select_toas] == "gbt")

        with pytest.raises(ValueError):
            mp_wrong_keyvalue = maskParameter(
                "test2", key="tel", key_value=["gbt", "ao"]
            )

    def test_name_mask(self):
        # Not sure about the use case for name mask
        mp_name = maskParameter("test2", key="name", key_value=10)
        assert mp_name.key_value == ["10"]

    def test_flag_mask(self):
        mp_flag = maskParameter("test2", key="-fe", key_value=430)
        assert mp_flag.key_value == ["430"]
        mp_flag2 = maskParameter("test2", key="-fe", key_value="430")
        assert mp_flag2.key_value == ["430"]
        with pytest.raises(ValueError):
            mp_wrong_key = maskParameter("test2", key="fe", key_value="430")
        mp_flag3 = maskParameter("test2", key="-fe", key_value="L-wide")
        assert mp_flag3.key_value == ["L-wide"]
        select_toas = mp_flag3.select_toa_mask(self.toas)
        assert np.all(self.toas.table["fe"][select_toas] == "L-wide")

    def test_read_from_par(self):
        temp_par = """
                F0    10 1 0.0001
                JUMP -fe L-wide 0.01 1 0.001
                JUMP -fe S-wide 0.02 1 0.001
                JUMP mjd 55000 56000 0.03 1 0.001
                JUMP freq 1400 1440 0.04 1 0.001
                JUMP tel ao 0.07 1 0.001
                   """
        model = get_model(StringIO(temp_par))
        assert len(model.jumps) == 5
        assert len(model.jump_phase(self.toas, 0.0)) == self.toas.ntoas
        jump_phase0 = model.F0.quantity * (
            model.JUMP1.quantity + model.JUMP4.quantity + model.JUMP5.quantity
        )

        assert (
            model.jump_phase(self.toas, 0.0)[0] - jump_phase0
        ) < 1e-16 * jump_phase0.unit
