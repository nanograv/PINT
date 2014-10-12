# phase.py
# Simple class representing pulse phase as integer and fractional
# parts.
from collections import namedtuple
import numpy

class Phase(namedtuple('Phase', 'int frac')):
    """
    Class representing pulse phase internally as integer and fractional
    parts.  Can be initialized from a single numerical input (for example
    float or mpmath value), or separately provided integer and fractional
    parts, depending on what precision is required.  For example:

    >>> import mpmath
    >>> # Set the precision to 20 digits:
    >>> mpmath.mp.dps = 20
    >>> # Phase from mpmath float:
    >>> p1 = Phase(mpmath.mpf("12345.12345678901234567890"))
    >>> # Phase from integer and fractional components:
    >>> p2 = Phase(12345,0.12345678901234567890)
    >>> # Phase from standard python float:
    >>> p3 = Phase(12345.12345678901234567890)
    >>> print p1 - p2
    Phase(int=0, frac=0.0)
    >>> print p1 - p3
    Phase(int=0, frac=8.752859548266656e-13)

    """
    __slots__ = ()

    def __new__(cls, arg1, arg2=None):
        # Assume inputs are numerical, could add an extra
        # case to parse strings as input.
        if arg2 is None:
            ii = int(arg1)
            ff = float(arg1 - ii)
        else:
            ii = int(arg1) + int(arg2)
            ff = float(arg2 - int(arg2))
        if ff < -0.5:
            ff += 1.0
            ii -= 1
        elif ff > 0.5:
            ff -= 1.0
            ii += 1
        return super(Phase, cls).__new__(cls, ii, ff)

    def __neg__(self):
        return Phase(-self.int, -self.frac)

    def __add__(self, other):
        ff = self.frac + other.frac
        ii = int(ff)
        return Phase(self.int + other.int + ii, ff - ii)

    def __sub__(self, other):
        return self.__add__(other.__neg__())

class Phase_array(namedtuple('Phase_array', 'int frac')):
    """
    Phase class array version
    """
    __slots__ = ()
    def __new__(cls, arg1, arg2=None):
        # Assume inputs are numerical, could add an extra
        # case to parse strings as input.
        if arg2 is None:
            ff,ii = numpy.modf(arg1)
        else:
            arg1S = numpy.modf(arg1)
            arg2S = numpy.modf(arg2)
            ii = arg1S[1]+arg2S[1]    
            ff = arg2S[0]
        
        index = numpy.where(ff < -0.5)
        ff[index] += 1.0
        ii[index] -= 1
        index = numpy.where(ff > 0.5)
        ff[index] -= 1.0
        ii[index] += 1    
        return super(Phase_array, cls).__new__(cls, ii, ff)

    def __neg__(self):
        return Phase_array(-self.int, -self.frac)

    def __add__(self, other):
        ff = self.frac + other.frac
        ii = numpy.modf(ff)[1]
        return Phase_array(self.int + other.int + ii, ff - ii)

    def __sub__(self, other):
        return self.__add__(other.__neg__())    
