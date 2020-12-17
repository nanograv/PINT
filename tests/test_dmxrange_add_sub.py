"""Test functions for adding and removing DMX ranges.""" 

import pytest

from pint.models.dispersion_model import DispersionDM, DispersionDMX

def test_startMJD_greaterThan_endMJD():
    """ Check for error when start MJD is larger than end MJD. """
    with pytest.raises(ValueError):
        dm_mod=DispersionDMX()
        dm_mod.add_DMX_range(58100,58000,3,1,frozen=False)

def test_only_one_MJD():
    """ Check for error when one of the MJDs is None and the other isn't. """ 
    with pytest.raises(ValueError):
        dm_mod=DispersionDMX()
        dm_mod.add_DMX_range(None,58000,3,1,frozen=False)
    with pytest.raises(ValueError):
        dm_mod=DispersionDMX()
        dm_mod.add_DMX_range(58000,None,3,1,frozen=False)

def test_duplicate_index()
    """ Check for error when a duplicate DMX index is used. """ 
    with pytest.raises(ValueError):
        dm_mod=DispersionDMX()
        dm_mod.add_DMX_range(58000,58100,3,1,frozen=False)
        dm_mod.add_DMX_range(58200,58300,3,1,frozen=False)

def test_remove_nonexistent_index()
    """ Check for error when a unused DMX index is removed. """
    with pytest.raises(ValueError):
        dm_mod=DispersionDMX()
        dm_mod.remove_DMX_range(58000,58100,3,1,frozen=False)

def test_add_DMX():
    """ Check that the DMX model contains the DMX events after they are added. """
    dm_mod=DispersionDMX()
    dm_mod.add_DMX_range(58000,58100,3,1,frozen=False)
    assert len(dm_mod.params) == 7

def test_remove_DMX():
    """Check the different ways of grouping components."""
    dm_mod=DispersionDMX()
    dm_mod.add_DMX_range(58000,58100,3,1,frozen=False)
    dm_mod.remove_DMX_range(3)
    assert len(dm_mod.params) == 4

