import pytest
import logging
import pytest
import pint.observatory


class TestObservatoryMetadata:
    """Test handling of observatory metadata"""

    @classmethod
    def setup_class(cls):
        # name of an observatory that PINT should know about
        # and should have metadata on
        cls.pint_obsname = "gbt"
        # name of an observatory that only astropy should know about
        cls.astropy_obsname = "keck"

        cls.log = logging.getLogger("TestObservatoryMetadata")

    def test_astropy_observatory(self):
        """
        try to instantiate the observatory in PINT from astropy and check their metadata
        """
        keck = pint.observatory.get_observatory(self.astropy_obsname)
        msg = f"Checking PINT metadata for '{self.astropy_obsname}' failed: 'astropy' not present in '{keck.origin}'"
        assert "astropy" in keck.origin, msg

    def test_pint_observatory(self):
        """
        try to instantiate the observatory in PINT  and check their metadata
        """
        gbt = pint.observatory.get_observatory(self.pint_obsname)
        msg = f"Checking PINT definition for '{self.pint_obsname}' failed: metadata is '{gbt.origin}'"
        assert (gbt.origin is not None) and (len(gbt.origin) > 0), msg

    def test_observatory_replacement(self):
        from pint.observatory.topo_obs import TopoObs

        obsname = "nonexistent"

        TopoObs(
            obsname,
            itrf_xyz=[882589.65, -4924872.32, 3943729.348],
            overwrite=True,
            origin="Inserted for testing purposes",
        )
        obs = pint.observatory.get_observatory(obsname)
        pytest.raises(
            ValueError,
            TopoObs,
            obsname,
            itrf_xyz=[882589.65, -4924872.32, 3943729.348],
            origin="This is a test - replacement",
        )
        obs = pint.observatory.get_observatory(obsname)
        msg = f"Checking that 'replacement' is not in the metadata for '{obsname}': metadata is '{obs.origin}'"
        assert "replacement" not in obs.origin, msg
        TopoObs(
            obsname,
            itrf_xyz=[882589.65, -4924872.32, 3943729.348],
            origin="This is a test - replacement",
            overwrite=True,
        )
        obs = pint.observatory.get_observatory(obsname)
        msg = f"Checking that 'replacement' is now in the metadata for '{obsname}': metadata is '{obs.origin}'"
        assert "replacement" in obs.origin, msg
