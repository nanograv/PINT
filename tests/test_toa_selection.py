import pint.models.model_builder as mb
import pint.toa as toa
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import astropy.units as u
import os, unittest
from pint.toa_select import TOASelect
import copy
from pinttestdata import testdir, datadir
import logging
os.chdir(datadir)

class TestTOAselection(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.parf= 'B1855+09_NANOGrav_9yv1.gls.par'
        self.timf = 'B1855+09_NANOGrav_9yv1.tim'
        self.toas = toa.get_TOAs(self.timf, ephem="DE421", planets=False)
        self.model = mb.get_model(self.parf)
        self.sort_table = copy.deepcopy(self.toas.table)
        self.sort_table.sort('mjd_float')
        self.sort_table['index'] = range(len(self.sort_table))

    def add_dmx_section(self, toas):
        DMX_mapping = self.model.get_prefix_mapping('DMX_')
        DMXR1_mapping = self.model.get_prefix_mapping('DMXR1_')
        DMXR2_mapping = self.model.get_prefix_mapping('DMXR2_')
        if 'DMX_section' not in toas.keys():
            toas['DMX_section'] = np.zeros_like(toas['index'])
            epoch_ind = 1
            while epoch_ind in DMX_mapping:
                # Get the parameters
                r1 = getattr(self.model, DMXR1_mapping[epoch_ind]).quantity
                r2 = getattr(self.model, DMXR2_mapping[epoch_ind]).quantity
                msk = np.logical_and(toas['mjd_float'] >= r1.mjd, toas['mjd_float'] <= r2.mjd)
                toas['DMX_section'][msk] = epoch_ind
                epoch_ind = epoch_ind + 1
        return toas

    def get_dmx_old(self, toas):
        # old way of select DMX
        DMX_mapping = self.model.get_prefix_mapping('DMX_')
        DMXR1_mapping = self.model.get_prefix_mapping('DMXR1_')
        DMXR2_mapping = self.model.get_prefix_mapping('DMXR2_')
        if 'DMX_section' not in toas.keys():
            toas = self.add_dmx_section(toas)
        # Get DMX delays
        dmx_old = np.zeros(len(toas)) * self.model.DM.units
        DMX_group = toas.group_by('DMX_section')
        for ii, key in enumerate(DMX_group.groups.keys):
            keyval = key.as_void()[0]
            if keyval != 0:
                dmx = getattr(self.model, DMX_mapping[keyval]).quantity
                ind = DMX_group.groups[ii]['index']
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

    def test_DMX_selection(self):
        dmx_old = self.get_dmx_old(self.toas.table).value
        # New way in the code.
        dmx_new = self.model.dmx_dm(self.toas.table).value
        assert np.allclose(dmx_old, dmx_new)

    def test_change_toas(self):
        assert not np.allclose(self.toas.table['mjd_float'], self.sort_table['mjd_float'])
        dmx_old = self.get_dmx_old(self.sort_table).value
        dmx_new = self.model.dmx_dm(self.sort_table).value
        assert np.allclose(dmx_old, dmx_new)

    def test_hash(self):
        self.model.dmx_toas_selector.use_hash = True
        dmx_old = self.get_dmx_old(self.toas.table).value
        dmx_new = self.model.dmx_dm(self.toas.table).value
        assert np.allclose(dmx_old, dmx_new)
        assert self.model.dmx_toas_selector.use_hash
        dmx_old = self.get_dmx_old(self.sort_table).value
        dmx_new = self.model.dmx_dm(self.sort_table).value
        assert np.allclose(dmx_old, dmx_new)

    def test_change_condition(self):
        log= logging.getLogger( "TestTOAselection.test_change_condition" )
        dmx_old = self.get_dmx_old(self.toas.table).value
        dmx_new = self.model.dmx_dm(self.toas.table).value
        indx0004 = self.model.dmx_toas_selector.select_result['DMX_0004']
        indx0005 = self.model.dmx_toas_selector.select_result['DMX_0005']
        for l in indx0004:
            log.debug( "indx0004= %s", str(l))
        for l in indx0005:
            log.debug( "indx0005= %s", str(l))
        Temp1 = self.model.DMXR2_0004.value
        Temp2 = self.model.DMXR1_0005.value
        self.model.DMXR2_0004.value = self.model.DMXR2_0005.value
        self.model.DMXR1_0005.value = self.model.DMXR2_0005.value
        dmx_old = self.get_dmx_old(self.toas.table).value
        dmx_new = self.model.dmx_dm(self.toas.table).value
        indx0004_2 = self.model.dmx_toas_selector.select_result['DMX_0004']
        indx0005_2 = self.model.dmx_toas_selector.select_result['DMX_0005']
        for l in indx0004_2:
            log.debug( "indx0004_2= %s", str(l))
        for l in indx0005_2:
            log.debug( "indx0005_2= %s", str(l))
        self.model.DMXR2_0004.value = Temp1
        self.model.DMXR1_0005.value = Temp2
        run1 = np.concatenate((indx0004, indx0005))
        run2 = np.concatenate((indx0004_2, indx0005_2))
        run1.sort()
        run2.sort()
        assert len(indx0005_2) == 0
        assert len(run1) == len(run2)
        assert np.allclose(run1, run2)
