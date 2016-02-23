# A simple test for dd binary derivatives
from pint.models.DDindependent import PSRdd as DD
import astropy.units as u
import numpy as np
import pint.models.model_builder as mb
import pint.toa as toa
import numdifftools as nd
import copy
import astropy.constants as c
import os, unittest
datapath = os.path.join(os.environ['PINT'],'tests','datafile')


class test_diff(object):
    """Setup a new class for testing derivatives.
    """
    def __init__(self,model,parname,dname):
        self.model = copy.copy(model)
        self.parn = parname
        self.par = getattr(self.model,parname)
        self.parU = self.par.unit
        self.dname = dname
        self.resU= None
    def func(self,value):
        setattr(self.model,self.parn,value*self.parU)
        dv = getattr(self.model,self.dname)
        res = dv()
        self.resU = res.unit
        return res

    def num_diff(self,step,val):
        h = val*step
        if h == 0:
            h = step
        v1 = self.func(val+h)
        v2 = self.func(val)
        return ((v1-v2)/(h*self.parU)).decompose()

def testdiff(model,dy,step):
    Pars = model.binary_params
    dervs = {}
    for p in Pars:
        if p is 'EDOT':
            stepv = 1e-19
        elif p is 'PBDOT':
            stepv = 1e-7
        else:
            stepv = step
        print p
        c = test_diff(model,p,dy)
        pv = getattr(model,p).value
        numr = c.num_diff(stepv,pv)
        try:
            anlog = model.prtl_der(dy,p)
        except:
            if dy.startswith('delay'):
                anlog = getattr(model,'d_delay'+dy[5]+'_d_par')(p)
            else:
                raise ValueError('Wrong delay')
        print "Maximum error ", (numr.value - anlog.value).max()
        dervs[p+'_num'] = numr
        dervs[p+'_anlg'] = anlog
    return dervs


parfile = os.path.join(datapath, 'B1855+09_NANOGrav_dfg+12_modified.par')
timfile = os.path.join(datapath, 'B1855+09_NANOGrav_dfg+12.tim')
ddm = mb.get_model(parfile)
t = toa.get_TOAs(timfile,planets = True)
ddob = ddm.get_dd_object(t.table)
diff = testdiff(ddob,'delayInverse',1e-7)
