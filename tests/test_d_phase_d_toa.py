from pint.models import model_builder as mb
import pint.toa as toa
import matplotlib.pyplot as plt
import unittest
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
        print 'test_d_phase_d_toa at observatory.'
        dpdtoa = self.model.d_phase_d_toa(self.toas)
        plt.plot(self.toas.get_mjds(),dpdtoa)
        plt.show()

        print 'test_d_phase_d_toa at SSB.'
        dpdtoa = self.model.d_phase_d_toa(self.toas, obs='SSB')
        plt.plot(self.toas.get_mjds(),dpdtoa)
        plt.show()

        print 'test d_phase_d_toa as geocenter'
        dpdtoa = self.model.d_phase_d_toa(self.toas, obs='GEO')
        plt.plot(self.toas.get_mjds(),dpdtoa)
        plt.show()

if __name__ == '__main__':
    pass
