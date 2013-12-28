import pint.models as tm
from pint.models.timing_model import Parameter
from pint import toa
import pint.fitter as f
from pint import tmtestfuncs 
from astropy import time

reload(tm)
reload(toa)
reload(f)
reload(tmtestfuncs)

p1=Parameter(name="F0",value=60.0,units="Hz")
p2=Parameter(name="PEPOCH", value=53750.0000, units="MJD")

m=tm.TimingModel()
m.add_param(p1)
m.add_param(p2)
m.phase_funcs=[tmtestfuncs.F0]

t=toa.TOAs("tempo2Test/J0000+0000.tim")
r=f.resids(toas=t,model=m)

