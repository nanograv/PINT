from pint.models import model_builder as mb
import pint.toa as toa
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time as time
import unittest
import copy
import os

datapath = os.path.join(os.environ['PINT'], 'tests', 'datafile')


class TestD_phase_D_toa(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.parf = os.path.join(datapath, 'J1744-1134.basic.par')
        self.timf = os.path.join(datapath, 'J1744-1134.Rcvr1_2.GASP.8y.x.tim')
        self.model = mb.get_model(self.parf)
        self.toas = toa.get_TOAs(self.timf)

    def test_d_phase_d_toa(self):
        dpdtoa_model = self.model.d_phase_d_toa(self.toas)
        dpdtoa_full = self.d_phase_d_toa_full()
        dpdtoa_basic = self.d_phase_d_toa_basic()
        diff_m_f = dpdtoa_model - dpdtoa_full
        diff_m_b = dpdtoa_model - dpdtoa_basic
        max_diff_m_f = max(diff_m_f)
        max_diff_m_b = max(diff_m_b)
        emsg = str(max_diff_m_f) + "is bigger then 1e-8"
        assert max_diff_m_f < 1e-8, emsg
    def d_phase_d_toa_full(self):
        print "Test with full ssb_obs_pos calculation."
        copy_toas = copy.deepcopy(self.toas)
        pulse_period = 1.0/self.model.F0.value
        sample_step = pulse_period/10.0
        sample_dt = [-sample_step,2 * sample_step]

        sample_phase = []
        for dt in sample_dt:
            dt_array = ([dt] * copy_toas.ntoas) * u.s
            deltaT = time.TimeDelta(dt_array)
            copy_toas.adjust_TOAs(deltaT, get_posvel=True)
            phase = self.model.phase(copy_toas.table)
            sample_phase.append(phase)
        # Use finite difference method.
        # phase'(t) = (phase(t+h)-phase(t-h))/2+ 1/6*F2*h^2 + ..
        # The error should be near 1/6*F2*h^2
        dp = (sample_phase[1]-sample_phase[0])
        d_phase_d_toa = dp.int/(2*sample_step)+dp.frac/(2*sample_step)
        return d_phase_d_toa

    def d_phase_d_toa_basic(self):
        print "Test with without ssb_obs_pos calculation."
        copy_toas = copy.deepcopy(self.toas)
        pulse_period = 1.0/self.model.F0.value
        sample_step = pulse_period/10.0
        sample_dt = [-sample_step,2 * sample_step]

        sample_phase = []
        for dt in sample_dt:
            dt_array = ([dt] * copy_toas.ntoas) * u.s
            deltaT = time.TimeDelta(dt_array)
            copy_toas.adjust_TOAs(deltaT)
            phase = self.model.phase(copy_toas.table)
            sample_phase.append(phase)
        # Use finite difference method.
        # phase'(t) = (phase(t+h)-phase(t-h))/2+ 1/6*F2*h^2 + ..
        # The error should be near 1/6*F2*h^2
        dp = (sample_phase[1]-sample_phase[0])
        d_phase_d_toa = dp.int/(2*sample_step)+dp.frac/(2*sample_step)
        return d_phase_d_toa

if __name__ == '__main__':
    pass
