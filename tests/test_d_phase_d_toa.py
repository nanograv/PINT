from pint.models import model_builder as mb
import pint.toa as toa
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time as time
import numpy as np
import unittest
import copy
import os

datapath = os.path.join(os.environ['PINT'], 'tests', 'datafile')


class TestD_phase_D_toa(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.parf = os.path.join(datapath, 'test_d_phase_d_toa_F1.par')
        self.timf = os.path.join(datapath, 'test_d_phase_d_toa_F1.tim')
        self.model = mb.get_model(self.parf)
        self.toas = toa.get_TOAs(self.timf)

    def test_d_phase_d_toa(self):
        # TODO: How do you Calculate real derivative. To analysis the error?
        analog = self.ana_diff()
        emsg = ""
        for f in [0.1, 1, 10, 100, 1000, 5000]:
            step = 1.0/self.model.F0.num_value*f
            dpdtoa_model = self.model.d_phase_d_toa(self.toas, step)
            diff = dpdtoa_model - analog
            emsg += str(max(np.abs(diff))) + " "
        #dpdtoa_full = self.d_phase_d_toa_full()
        #dpdtoa_basic = self.d_phase_d_toa_basic()
        #diff_m_f = dpdtoa_model - dpdtoa_full
        #diff_m_b = dpdtoa_model - dpdtoa_basic
        #max_diff_m_f = max(diff_m_f)
        #max_diff_m_b = max(diff_m_b)
        # Calculate the error for derivative.
        #drd, step = self.d_third_derivative()
        assert False, emsg

    def ana_diff(self):
        der = self.model.F0.num_value + \
              self.model.F1.num_value * ((self.toas.table['tdbld'] - \
                                self.model.PEPOCH.num_value)* 86400.0)
        return der

    def d_phase_d_toa_full(self):
        print "Test with full ssb_obs_pos calculation."
        copy_toas = copy.deepcopy(self.toas)
        pulse_period = 1.0/self.model.F0.num_value
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
        pulse_period = 1.0/self.model.F0.num_value
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

    def d_third_derivative(self, step=None):
        print "Calculate third derivative For error."
        copy_toas = copy.deepcopy(self.toas)
        if step is None:
            pulse_period = 1.0/self.model.F0.num_value
            sample_step = pulse_period/10.0
        else:
            sample_step = step

        sample_dt = [-2 * sample_step, sample_step, 2 * sample_step,
                     sample_step]

        sample_phase = []
        for dt in sample_dt:
            dt_array = ([dt] * copy_toas.ntoas) * u.s
            deltaT = time.TimeDelta(dt_array)
            copy_toas.adjust_TOAs(deltaT)
            phase = self.model.phase(copy_toas.table)
            sample_phase.append(phase)

        diffInt = -0.5 * sample_phase[0].int + sample_phase[1].int - \
                  sample_phase[2].int + 0.5 * sample_phase[3].int

        diffFrac = -0.5 * sample_phase[0].frac + sample_phase[1].frac - \
                  sample_phase[2].frac + 0.5 * sample_phase[3].frac

        thirdDer = diffInt/(sample_step**3) + diffFrac/(sample_step**3)
        return thirdDer, sample_step

if __name__ == '__main__':
    pass
