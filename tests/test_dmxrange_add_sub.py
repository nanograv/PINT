"""Test functions for adding and removing DMX ranges.""" 

import pytest

from pint.models.dispersion_model import DispersionDM, DispersionDMX

def test_startMJD_greaterThan_endMJD():
    """ Check for error when start MJD is larger than end MJD. """
    dm_mod=DispersionDMX()
    with pytest.raises(ValueError):
        dm_mod.add_DMX_range(58100,58000,3,1,frozen=False)

def test_only_one_MJD():
    """ Check for error when one of the MJDs is None and the other isn't. """ 
    dm_mod=DispersionDMX()
    with pytest.raises(ValueError):
        dm_mod.add_DMX_range(None,58000,3,1,frozen=False)
    dm_mod=DispersionDMX()
    with pytest.raises(ValueError):
        dm_mod.add_DMX_range(58000,None,3,1,frozen=False)

def test_duplicate_index():
    """ Check for error when a duplicate DMX index is used. """ 
    dm_mod=DispersionDMX()
    dm_mod.add_DMX_range(58000,58100,3,1,frozen=False)
    with pytest.raises(ValueError):
        dm_mod.add_DMX_range(58200,58300,3,1,frozen=False)

def test_remove_nonexistent_index():
    """ Check for error when a unused DMX index is removed. """
    dm_mod=DispersionDMX()
    with pytest.raises(ValueError):
        dm_mod.remove_DMX_range(3)

def test_add_DMX():
    """ Check that the DMX model contains the DMX events after they are added. """
    dm_mod=DispersionDMX()
    index=3
    dmx=1.
    mjd_start=58000.
    mjd_end=58100.
    dm_mod.add_DMX_range(mjd_start,mjd_end,index,dmx,frozen=False)
    if len(dm_mod.params) != 7:
        assert False
    for pn,val in zip(['DMX_','DMXR1_','DMXR2_'],[dmx,mjd_start,mjd_end]):
        nm=pn+"{:04d}".format(index)
        comp=getattr(dm_mod,nm)
        if comp.value != val:
            assert False

def test_remove_DMX():
    """Check the DMX model no longer has the DMX components after they are removed. """ 
    dm_mod=DispersionDMX()
    index=3
    dmx=1.
    mjd_start=58000.
    mjd_end=58100.
    dm_mod.add_DMX_range(mjd_start,mjd_end,index,dmx,frozen=False)
    dm_mod.remove_DMX_range(3)
    for pn in zip(['DMX_','DMXR1_','DMXR2_']):
        nm=str(pn)+str("{:04d}".format(index))
        if nm in dm_mod.params:
            assert False

