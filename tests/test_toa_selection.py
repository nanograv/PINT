import copy
import logging
import os
import pytest

import astropy.units as u
import numpy as np

# import matplotlib
# matplotlib.use('TKAgg')
# import matplotlib.pyplot as plt

import pint.models.model_builder as mb
import pint.toa as toa
from pinttestdata import datadir


class TestTOAselection:
    @classmethod
    def setup_class(cls):
        os.chdir(datadir)
        cls.parf = "B1855+09_NANOGrav_9yv1.gls.par"
        cls.timf = "B1855+09_NANOGrav_9yv1.tim"
        cls.toas = toa.get_TOAs(cls.timf, ephem="DE421", planets=False)
        cls.model = mb.get_model(cls.parf)
        cls.sort_toas = copy.deepcopy(cls.toas)
        cls.sort_toas.table.sort("mjd_float")
        cls.sort_toas.table["index"] = range(cls.sort_toas.ntoas)

    def add_dmx_section(self, toas_table):
        DMX_mapping = self.model.get_prefix_mapping("DMX_")
        DMXR1_mapping = self.model.get_prefix_mapping("DMXR1_")
        DMXR2_mapping = self.model.get_prefix_mapping("DMXR2_")
        if "DMX_section" not in toas_table.keys():
            toas_table["DMX_section"] = np.zeros_like(toas_table["index"])
            epoch_ind = 1
            while epoch_ind in DMX_mapping:
                # Get the parameters
                r1 = getattr(self.model, DMXR1_mapping[epoch_ind]).quantity
                r2 = getattr(self.model, DMXR2_mapping[epoch_ind]).quantity
                msk = np.logical_and(
                    toas_table["mjd_float"] >= r1.mjd, toas_table["mjd_float"] <= r2.mjd
                )
                toas_table["DMX_section"][msk] = epoch_ind
                epoch_ind += 1
        return toas_table

    def get_dmx_old(self, toas):
        # old way of select DMX
        DMX_mapping = self.model.get_prefix_mapping("DMX_")
        DMXR1_mapping = self.model.get_prefix_mapping("DMXR1_")
        DMXR2_mapping = self.model.get_prefix_mapping("DMXR2_")
        tbl = toas.table
        if "DMX_section" not in tbl.keys():
            tbl = self.add_dmx_section(tbl)
        # Get DMX delays
        dmx_old = np.zeros(len(tbl)) * self.model.DM.units
        DMX_group = tbl.group_by("DMX_section")
        for ii, key in enumerate(DMX_group.groups.keys):
            keyval = key.as_void()[0]
            if keyval != 0:
                dmx = getattr(self.model, DMX_mapping[keyval]).quantity
                ind = DMX_group.groups[ii]["index"]
                dmx_old[ind] = dmx
        return dmx_old

    def test_boolean_selection(self):
        # This tests boolean TOA selection and unselection
        assert self.toas.ntoas == 4005
        self.toas.select(self.toas.get_errors() < 1.19 * u.us)
        assert self.toas.ntoas == 2001
        self.toas.select(self.toas.get_freqs() > 1.0 * u.GHz)
        assert self.toas.ntoas == 1918
        self.toas.unselect()
        assert self.toas.ntoas == 2001
        self.toas.unselect()
        assert self.toas.ntoas == 4005

    def test_dmx_selection(self):
        dmx_old = self.get_dmx_old(self.toas).value
        # New way in the code.
        dmx_new = self.model.dmx_dm(self.toas).value
        assert np.allclose(dmx_old, dmx_new)

    def test_change_toas(self):
        assert not np.allclose(
            self.toas.table["mjd_float"], self.sort_toas.table["mjd_float"]
        )
        dmx_old = self.get_dmx_old(self.sort_toas).value
        dmx_new = self.model.dmx_dm(self.sort_toas).value
        assert np.allclose(dmx_old, dmx_new)

    def test_hash(self):
        dmx_old = self.get_dmx_old(self.toas).value
        dmx_new = self.model.dmx_dm(self.toas).value
        self.model.dmx_toas_selector.use_hash = True
        assert np.allclose(dmx_old, dmx_new)
        assert self.model.dmx_toas_selector.use_hash
        dmx_old = self.get_dmx_old(self.sort_toas).value
        dmx_new = self.model.dmx_dm(self.sort_toas).value
        assert np.allclose(dmx_old, dmx_new)

    def test_change_condition(self):
        log = logging.getLogger("TestTOAselection.test_change_condition")
        dmx_old = self.get_dmx_old(self.toas).value
        dmx_new = self.model.dmx_dm(self.toas).value
        indx0004 = self.model.dmx_toas_selector.select_result["DMX_0004"]
        indx0005 = self.model.dmx_toas_selector.select_result["DMX_0005"]
        for l in indx0004:
            log.debug("indx0004= %s", str(l))
        for l in indx0005:
            log.debug("indx0005= %s", str(l))
        Temp1 = self.model.DMXR2_0004.value
        Temp2 = self.model.DMXR1_0005.value
        self.model.DMXR2_0004.value = self.model.DMXR2_0005.value
        self.model.DMXR1_0005.value = self.model.DMXR2_0005.value
        dmx_old = self.get_dmx_old(self.toas).value
        dmx_new = self.model.dmx_dm(self.toas).value
        indx0004_2 = self.model.dmx_toas_selector.select_result["DMX_0004"]
        indx0005_2 = self.model.dmx_toas_selector.select_result["DMX_0005"]
        for l in indx0004_2:
            log.debug("indx0004_2= %s", str(l))
        for l in indx0005_2:
            log.debug("indx0005_2= %s", str(l))
        self.model.DMXR2_0004.value = Temp1
        self.model.DMXR1_0005.value = Temp2
        run1 = np.concatenate((indx0004, indx0005))
        run2 = np.concatenate((indx0004_2, indx0005_2))
        run1.sort()
        run2.sort()
        assert len(indx0005_2) == 0
        assert len(run1) == len(run2)
        assert np.allclose(run1, run2)
