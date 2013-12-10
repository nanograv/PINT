def interpObsclock(t_i,method=None,fname=None):
    """Interpolate GBT clock corrections

    Keyword arguments:
    t_i -- input array like time in MJD
    method -- interpolation method supplied to scipy.interpolate.interp1d (default 'linear')
    filename -- filename of clock corrections (default './time_gbt.dat')

    Ex. t=interpObsclock(51942.59)

    N.B. interp1d with cubic seems to go crazy and take all 

    """    
    import numpy as np
    
    # Interpolation method, default is linear
    if method is None:
        method = 'linear'
    
    # Default filename of GBT clock corrections
    if fname is None:
        fname = './time_gbt.dat'

    # Load GBT timing info
    t,corr=np.loadtxt(fname,skiprows=2,usecols=(0,2),unpack=True)

    # Interpolate t vs. correction to ti
    from scipy.interpolate import interp1d
    f=interp1d(t,corr,kind=method)
    corr_i=f(t_i)

    return corr_i
