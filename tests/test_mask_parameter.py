"""Various tests for the maskParameter"""

import os
import pytest
from io import StringIO
import astropy.units as u
import numpy as np

from pint.models.model_builder import get_model
from pint.models.parameter import maskParameter
from pint.toa import get_TOAs
from pinttestdata import datadir


@pytest.fixture
def toas():
    """Load toas."""
    os.chdir(datadir)
    return get_TOAs("J1713+0747_NANOGrav_11yv0_short.tim")


def test_mjd_mask(toas):
    mp = maskParameter("test1", key="mjd", key_value=[54000, 54100])
    assert mp.key == "mjd"
    assert mp.key_value == [54000, 54100]
    assert mp.value is None
    select_toas = mp.select_toa_mask(toas)
    assert len(select_toas) > 0
    raw_selection = np.where(
        (toas.table["mjd_float"] <= 54100) * (toas.table["mjd_float"] >= 54000)
    )
    assert np.all(select_toas == raw_selection[0])
    assert np.all(toas.table["mjd_float"][select_toas] <= 54100)
    assert np.all(toas.table["mjd_float"][select_toas] >= 54000)
    mp_str_keyval = maskParameter("test2", key="mjd", key_value=["54000", "54100"])
    assert mp_str_keyval.key_value == [54000, 54100]
    mp_value_switch = maskParameter("test1", key="mjd", key_value=[54100, 54000])
    assert mp_value_switch.key_value == [54000, 54100]
    with pytest.raises(ValueError):
        mp_str_keyval = maskParameter("test2", key="mjd", key_value=["54000"])


def test_freq_mask(toas):
    mp = maskParameter("test2", key="freq", key_value=[1400, 1430])
    assert mp.key_value == [1400 * u.MHz, 1430 * u.MHz]
    select_toas = mp.select_toa_mask(toas)
    assert len(select_toas) > 0
    raw_selection = np.where(
        (toas.table["freq"] <= 1430 * u.MHz) * (toas.table["freq"] >= 1400 * u.MHz)
    )
    assert np.all(select_toas == raw_selection[0])
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


def test_tel_mask(toas):
    mp_ao = maskParameter("test2", key="tel", key_value="ao")
    assert mp_ao.key_value == ["arecibo"]
    select_toas = mp_ao.select_toa_mask(toas)
    assert len(select_toas) > 0
    raw_selection = np.where(toas.table["obs"] == "arecibo")
    assert np.all(select_toas == raw_selection[0])
    mp_gbt = maskParameter("test2", key="tel", key_value=["gbt"])
    assert mp_gbt.key_value == ["gbt"]
    select_toas = mp_gbt.select_toa_mask(toas)
    assert np.all(toas.table["obs"][select_toas] == "gbt")
    mp_special_obs1 = maskParameter("test2", key="tel", key_value="@")
    assert mp_special_obs1.key_value == ["barycenter"]
    mp_special_obs2 = maskParameter("test2", key="tel", key_value=0)
    assert mp_special_obs2.key_value == ["geocenter"]
    with pytest.raises(ValueError):
        mp_wrong_keyvalue = maskParameter("test2", key="tel", key_value=["gbt", "ao"])


def test_name_mask(toas):
    # Not sure about the use case for name mask
    mp_name = maskParameter(
        "test2", key="name", key_value="53393.000009.3.000.000.9y.x.ff"
    )
    assert mp_name.key_value == ["53393.000009.3.000.000.9y.x.ff"]
    select_toas = mp_name.select_toa_mask(toas)
    assert len(select_toas) > 0
    raw_selection = np.where(toas.table["name"] == "53393.000009.3.000.000.9y.x.ff")
    assert np.all(select_toas == raw_selection[0])
    with pytest.raises(ValueError):
        mp_wrong_keyvalue = maskParameter(
            "test2", key="name", key_value=["name1", "name2"]
        )


def test_flag_mask(toas):
    mp_flag = maskParameter("test2", key="-fe", key_value=430)
    assert mp_flag.key_value == ["430"]
    mp_flag2 = maskParameter("test2", key="-fe", key_value="430")
    assert mp_flag2.key_value == ["430"]
    with pytest.raises(ValueError):
        mp_wrong_key = maskParameter("test2", key="fe", key_value="430")
    mp_flag3 = maskParameter("test2", key="-fe", key_value="L-wide")
    assert mp_flag3.key_value == ["L-wide"]
    select_toas = mp_flag3.select_toa_mask(toas)
    assert len(select_toas) > 0
    raw_selection = np.where(toas.table["fe"] == "L-wide")
    assert np.all(select_toas == raw_selection[0])


def test_read_from_par(toas):
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
    assert len(model.jump_phase(toas, 0.0)) == toas.ntoas
    jump_phase0 = model.F0.quantity * (
        model.JUMP1.quantity + model.JUMP4.quantity + model.JUMP5.quantity
    )

    assert (model.jump_phase(toas, 0.0)[0] - jump_phase0) < 1e-16 * jump_phase0.unit
