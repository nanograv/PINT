"""Test functions for adding and removing DMX ranges."""

import pytest
import io
import numpy as np
from pint.models import get_model
from pint.models.dispersion_model import DispersionDM, DispersionDMX


def test_startMJD_greaterThan_endMJD():
    """ Check for error when start MJD is larger than end MJD. """
    dm_mod = DispersionDMX()
    with pytest.raises(ValueError):
        dm_mod.add_DMX_range(58100, 58000, 3, 1, frozen=False)


def test_only_one_MJD():
    """ Check for error when one of the MJDs is None and the other isn't. """
    dm_mod = DispersionDMX()
    with pytest.raises(ValueError):
        dm_mod.add_DMX_range(None, 58000, 3, 1, frozen=False)
    with pytest.raises(ValueError):
        dm_mod.add_DMX_range(58000, None, 3, 1, frozen=False)


def test_duplicate_index():
    """ Check for error when a duplicate DMX index is used. """
    dm_mod = DispersionDMX()
    dm_mod.add_DMX_range(58000, 58100, 3, 1, frozen=False)
    with pytest.raises(ValueError):
        dm_mod.add_DMX_range(58200, 58300, 3, 1, frozen=False)


def test_remove_nonexistent_index():
    """ Check for error when a unused DMX index is removed. """
    dm_mod = DispersionDMX()
    with pytest.raises(ValueError):
        dm_mod.remove_DMX_range(3)


def test_nonetype_MJD():
    dm_mod = DispersionDMX()
    with pytest.raises(ValueError):
        dm_mod.add_DMX_range(None, 58000, 3, 1, frozen=False)
    with pytest.raises(ValueError):
        dm_mod.add_DMX_range(58000, None, 3, 1, frozen=False)


def test_dynamic_index():
    dm_mod = DispersionDMX()
    index1 = 3
    dmx1 = 1.0
    mjd_start1 = 58000.0
    mjd_end1 = 58100.0
    index2 = None
    dmx2 = 1.0
    mjd_start2 = 58200.0
    mjd_end2 = 58300.0
    dm_mod.add_DMX_range(mjd_start1, mjd_end1, index1, dmx1, frozen=False)
    dct = dm_mod.get_prefix_mapping_component("DMX_")
    ind = np.max(list(dct.keys())) + 1
    dm_mod.add_DMX_range(mjd_start2, mjd_end2, index2, dmx2, frozen=False)
    dct = dm_mod.get_prefix_mapping_component("DMX_")
    assert np.max(list(dct.keys())) == ind


def test_add_DMX():
    """ Check that the DMX model contains the DMX events after they are added. """
    dm_mod = DispersionDMX()
    index = 3
    dmx = 1.0
    mjd_start = 58000.0
    mjd_end = 58100.0
    init_len=len(dm_mod.params)
    dm_mod.add_DMX_range(mjd_start, mjd_end, index, dmx, frozen=False)
    assert len(dm_mod.params) == init_len+3
    nm = "DMX_" + "{:04d}".format(index)
    comp = getattr(dm_mod, nm)
    assert comp.value == dmx
    nm = "DMXR1_" + "{:04d}".format(index)
    comp = getattr(dm_mod, nm)
    assert comp.value == mjd_start
    nm = "DMXR2_" + "{:04d}".format(index)
    comp = getattr(dm_mod, nm)
    assert comp.value == mjd_end


def test_remove_DMX():
    """Check the DMX model no longer has the DMX components after they are removed. """
    dm_mod = DispersionDMX()
    index = 3
    dmx = 1.0
    mjd_start = 58000.0
    mjd_end = 58100.0
    dm_mod.add_DMX_range(mjd_start, mjd_end, index, dmx, frozen=False)
    dm_mod.remove_DMX_range(index)
    for pn in zip(["DMX_", "DMXR1_", "DMXR2_"]):
        nm = str(pn) + str("{:04d}".format(index))
        assert nm not in dm_mod.params


def test_model_usage():
    par_base = """
    PSR J1234+5678
    F0 1 0
    ELAT 0 0
    ELONG 0 0
    PEPOCH 57000
    DM 10 0
    DMX 15
    DMX_0001 16 1
    DMXR1_0001 58000
    DMXR2_0001 59000
    """
    index = 3
    dmx = 1.0
    mjd_start = 58000.0
    mjd_end = 58100.0

    model = get_model(
            io.StringIO(
                par_base
                )
            )
        
    
    dm_mod = model.components["DispersionDMX"]
    dm_mod.add_DMX_range(mjd_start, mjd_end, index, dmx, frozen=False)
    nm = "DMX_" + "{:04d}".format(index)
    comp = getattr(dm_mod, nm)
    assert comp.value == dmx
    nm = "DMXR1_" + "{:04d}".format(index)
    comp = getattr(dm_mod, nm)
    assert comp.value == mjd_start
    nm = "DMXR2_" + "{:04d}".format(index)
    comp = getattr(dm_mod, nm)
    assert comp.value == mjd_end
    dm_mod.remove_DMX_range(index)
    for pn in ["DMX_", "DMXR1_", "DMXR2_"]:
        nm = str(pn) + str("{:04d}".format(index))
        assert nm not in dm_mod.params
