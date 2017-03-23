#!/usr/bin/env python
from __future__ import division, print_function
import sys, os
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import unittest
import numpy as np
import pint.scripts.fermiphase as fermiphase
from pint.observatory.fermi_obs import FermiObs
from pint.fermi_toas import load_Fermi_TOAs
import pint.toa as toa
import pint.models
from pinttestdata import testdir, datadir

parfile = os.path.join(datadir, 'PSRJ0030+0451_psrcat.par')
eventfile = os.path.join(datadir, 'J0030+0451_P8_15.0deg_239557517_458611204_ft1weights_GEO_wt.gt.0.4.fits')

eventfileraw = os.path.join(datadir, 'J0030+0451_w323_ft1weights.fits')
ft2file = os.path.join(datadir, 'lat_spacecraft_weekly_w323_p202_v001.fits')
class TestFermiPhase(unittest.TestCase):

    def test_GEO_Htest_result(self):
        saved_stdout, fermiphase.sys.stdout = fermiphase.sys.stdout, StringIO('_')
        cmd = '--plot --plotfile fermitest.png --outfile fermitest.fits {0} {1} CALC'.format(eventfile,parfile)
        fermiphase.main(cmd.split())
        lines = fermiphase.sys.stdout.getvalue()
        v = 999.0
        for l in lines.split('\n'):
            if l.startswith('Htest'):
                v = float(l.split()[2])
        # Check that H-test is greater than 1380
        self.assertTrue(v>1380)
        fermiphase.sys.stdout = saved_stdout

    def test_process_raw_LAT_times(self):
        # This just checks that the process of reading the FT2 file and
        # computing positions doesn't crash. It doesn't validate any results.
        # Should probably run comparison with GEO or BARY phases.
        modelin = pint.models.get_model(parfile)
        FermiObs(name='Fermi',ft2name=ft2file,tt2tdb_mode='none')
        tl  = load_Fermi_TOAs(eventfileraw, weightcolumn='PSRJ0030+0451')
        ts = toa.TOAs(toalist=tl)
        ts.filename = eventfileraw
        ts.compute_TDBs()
        ts.compute_posvels(ephem='DE405',planets=False)
        phss = modelin.phase(ts.table)[1]
        phases = np.where(phss < 0.0, phss + 1.0, phss)



if __name__ == '__main__':
    unittest.main()
