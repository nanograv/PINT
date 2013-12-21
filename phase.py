# phase.py
# Simple class representing pulse phase as integer and fractional
# parts.
from collections import namedtuple

class Phase(namedtuple('Phase','int frac')):
    """
    Phase(arg1,arg2=None)

    Class representing pulse phase internally as integer and fractional
    parts.  Can be initialized from a single numerical input (for example 
    float or mpmath value), or separately provided integer and fractional
    parts, depending on what precision is required.  For example:

    # Set the precision to 20 digits:
    mpmath.mp.dps = 20
    # Phase from mpmath float:
    p1 = Phase(mpmath.mpf("12345.12345678901234567890"))

    # Phase from integer and fractional components:
    p2 = Phase(12345,0.12345678901234567890)

    # Phase from standard python float:
    p3 = Phase(12345.12345678901234567890)

    print p1 - p2
    >>> Phase(int=0, frac=0.0)

    print p1 - p3
    >>> Phase(int=0, frac=8.752859548266656e-13)

    """
    __slots__ = ()

    def __new__(cls,arg1,arg2=None):
        # Assume inputs are numerical, could add an extra
        # case to parse strings as input.
        if arg2 is None:
            ii = int(arg1)
            ff = float(arg1 - ii)
        else:
            ii = int(arg1) + int(arg2)
            ff = float(arg2 - int(arg2))
        return super(Phase,cls).__new__(cls,ii,ff)

    def __neg__(self):
        return Phase(-self.int,-self.frac)

    def __add__(self,other):
        ff = self.frac + other.frac
        ii = int(ff)
        return Phase(self.int + other.int + ii, ff - ii)

    def __sub__(self,other):
        return self.__add__(other.__neg__())

