from __future__ import absolute_import, division, print_function

from collections import namedtuple

import astropy.units as u
import numpy


class Phase(namedtuple("Phase", "int frac")):
    """
    Class representing pulse phase as integer (.int) and fractional (.frac) parts.

    The phase values are dimensionless Quantitys (u.dimensionless_unscaled == u.Unit("") == Unit(dimensionless))

    Ensures that the fractional part stays in [-0.5, 0.5)

    SUGGESTION(@paulray): How about adding some documentation here
    describing why the fractional part is reduced to [-0.5,0.5) instead of [0,1).

    """

    __slots__ = ()

    def __new__(cls, arg1, arg2=None):
        """Create new Phase object

        Constructs a Phase object. 
        Can be initialized with arrays or a scalar Quantity.
        Can take inputs as plain numerical types, or dimensionaless Quantitys
        Accepts either floating point argument (arg1) or pair of arguments with integer (arg1) and fractional (arg2) parts separate
        Scalars are converted to length 1 arrays so `Phase.int` and `Phase.frac` are always arrays

        Parameters
        ----------
        arg1 : array or dimensionless Quantity
        arg2 : array or dimensionless Quantity 

        Returns
        -------
        Phase : pulse phase object with arrays of dimensionless Quantitys as the int and frac parts
        """
        if not hasattr(arg1, "unit"):
            arg1 = u.Quantity(arg1)
        else:
            # This will raise an exception if the argument has any unit not convertable to Unit(dimensionless)
            arg1 = arg1.to(u.dimensionless_unscaled)
        #  If arg is scalar, convert to an array of length 1
        if arg1.shape == ():
            arg1 = arg1.reshape((1,))
        if arg2 is None:
            ff, ii = numpy.modf(arg1)
        else:
            if not hasattr(arg2, "unit"):
                arg2 = u.Quantity(arg2)
            else:
                arg2 = arg2.to(u.dimensionless_unscaled)
            if arg2.shape == ():
                arg2 = arg2.reshape((1,))
            arg1S = numpy.modf(arg1)
            arg2S = numpy.modf(arg2)
            # Prior code assumed that fractional part of arg1 was 0 if arg2 was present
            # @paulray removed that assumption here
            ff = arg1S[0] + arg2S[0]
            ii = arg1S[1] + arg2S[1]
        index = ff < -0.5
        ff[index] += 1.0
        ii[index] -= 1
        # The next line is >= so that the range is the interval [-0.5,0.5)
        # Otherwise, the same phase could be represented 0,0.5 or 1,-0.5
        index = ff >= 0.5
        ff[index] -= 1.0
        ii[index] += 1
        return super(Phase, cls).__new__(cls, ii, ff)

    def __neg__(self):
        # TODO: add type check for __neg__ and __add__
        return Phase(-self.int, -self.frac)

    def __add__(self, other):
        ff = self.frac + other.frac
        ii = numpy.modf(ff)[1]
        return Phase(self.int + other.int + ii, ff - ii)

    def __sub__(self, other):
        return self.__add__(other.__neg__())
