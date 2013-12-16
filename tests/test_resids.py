import timing_model as tm
import toa
import fitter as f
import tmtestfuncs 
from astropy import time

reload(tm)
reload(toa)
reload(f)
reload(tmtestfuncs)

p1=tm.Parameter(name="F0",value=60.0,units="Hz")
p2=tm.Parameter(name="PEPOCH", value=53750.0000, units="MJD")

m=tm.TimingModel()
m.add_param(p1)
m.add_param(p2)
m.phase_funcs=[tmtestfuncs.F0]

t=toa.TOAs("tests/NGC6440E.tim")
r=f.resids(toas=t,model=m)

