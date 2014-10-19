# phase.py
# Simple class representing pulse phase as integer and fractional
# parts.
from collections import namedtuple
import numpy
 
class Phase(namedtuple('Phase', 'int frac')):
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
        return super(Phase, cls).__new__(cls, ii, ff)

    def __neg__(self):
        return Phase(-self.int, -self.frac)

    def __add__(self, other):
        ff = self.frac + other.frac
        ii = numpy.modf(ff)[1]
        return Phase(self.int + other.int + ii, ff - ii)

    def __sub__(self, other):
        return self.__add__(other.__neg__())    
