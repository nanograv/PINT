#! /usr/bin/env python
import time, sys, os, numpy
import pint.models as tm
from pint import toa
from astropy import log

from pinttestdata import datadir

def test_wave():
    os.chdir(datadir)
    parfile = 'J1513-5908_PKS_alldata_white.par'
    timfile = 'J1744-1134.Rcvr1_2.GASP.8y.x.tim'
    t = toa.get_TOAs(timfile, usepickle=False)

    m = tm.get_model(parfile)
    log.info("model.as_parfile():\n%s"%m.as_parfile())

    print(m.as_parfile())
    print(m)
    print(m.delay(t))

