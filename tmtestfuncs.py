# Test timing model functions to test out the residuals class
from astropy.time import Time
import numpy

def F0(toa,model):
    
    dt = toa.get_mjds() - Time(model.PEPOCH.value,format="mjd",scale="utc")
    # Can use dt[n].jd1 and jd2 with mpmath here if necessary
    ph = numpy.array([x.sec*model.F0.value for x in dt])

    return ph




