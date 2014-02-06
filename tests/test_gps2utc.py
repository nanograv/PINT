import os
import numpy

def get_gps2utc_vals(filename):
    """
    get_gps2utc_vals(filename):

    Return a tuple of numpy arrays of MJDS and clock correction form GPS to UTC 
    in seconds.
    
    File comes from tempo2 T2runtime clock directory
    """
    filename = os.path.join(os.environ["TEMPO2"],"clock",filename)
    mjds, gps2utcCorr = numpy.loadtxt(filename,unpack = True)
    return mjds, gps2utcCorr

def add_gps2utc_corr(filename,toas):
    """
    Need to be Fixed. Integret to toa class and adding flags and expectations. 
    """
    mjds,corr  = get_gps2utc_vals(filename)
    tvals = numpy.array([tx.mjd.value for tx in toas])
    corrs = numpy.interp(tvals,mjds,corr)
    return corrs
