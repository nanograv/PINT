from pint.models import model_builder as mb
import pint.models.polycos as py
import pint.utils as ut
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
        self.parf = os.path.join(datapath, 'test_d_phase_d_toa.par')
        self.timf = os.path.join(datapath, 'test_d_phase_d_toa.tim')
        self.model = mb.get_model(self.parf)
        self.toas = toa.get_TOAs(self.timf)
        self.polycos= py.Polycos()

    def test_d_phase_d_toa(self):
        # TODO: How do you Calculate real derivative. To analysis the error?
        analog = self.ana_diff()
        dpdtoa_model = self.model.d_phase_d_toa(self.toas)
        diff = dpdtoa_model - analog
        emsg = "Max difference is %s between d_phase_d_toa and realy value. "%str(max(np.abs(diff)))
        assert np.all(diff < 1e-7), emsg

    def test_d_phase_d_toa_polyco(self):
        # Use Tempo generated polyco to test the frequency
        parfile = os.path.join(datapath, 'J1744-1134.basic2.par')
        timfile = os.path.join(datapath, 'J1744-1134.Rcvr1_2.GASP.8y.x.tim')
        m = mb.get_model(parfile)
        t = toa.get_TOAs(timfile)
        mask = t.table['mjd']< time.Time(53219.0, format='mjd',scale='utc')
        polycofile = os.path.join(datapath, 'J1744-1134_polyco.dat')
        self.polycos.read_polyco_file(polycofile, 'tempo')
        mjdld = []
        for mjd in t.table['mjd'][mask]:
            mjdld.append(ut.time_to_longdouble(mjd))

        freqPolyco = self.polycos.eval_spin_freq(mjdld)
        freqModel =  m.d_phase_d_toa(t, 1000)
        diff = freqPolyco - freqModel[mask]

        assert np.all(diff<1e-7), " The result from d_phase_d_toa is not match" + \
                "polyco result."

    def test_d_phase_d_toa_coe(self):
        """Test toa at center of earth.
        """
        parfile = os.path.join(datapath, 'J1744-1134.basic2.par')
        timfile = os.path.join(datapath, 'J1744_coe.tim')
        m_coe = mb.get_model(parfile)
        t_coe = toa.get_TOAs(timfile)
        polycofile = os.path.join(datapath, 'J1744_coe.dat')
        plyc = py.Polycos()
        mjdld = []
        for mjd in t_coe.table['mjd']:
            mjdld.append(ut.time_to_longdouble(mjd))

        plyc.read_polyco_file(polycofile, 'tempo')
        freqPlyc = plyc.eval_spin_freq(mjdld)
        freqModel = m_coe.d_phase_d_toa(t_coe, 10000)
        diff = freqPlyc - freqModel
        assert np.all(diff < 1e-7), "d_phase_d_toa at center of earth is not match tempo polyco result."
        
    def ana_diff(self):
        t_reduced = (self.toas.table['tdbld'] - \
                     self.model.PEPOCH.num_value)* 86400.0
        der = self.model.F0.num_value + self.model.F1.num_value * t_reduced + \
              self.model.F2.num_value/2.0*(t_reduced**2)
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
