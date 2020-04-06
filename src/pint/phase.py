# phase.py
# Simple class representing pulse phase as integer and fractional
# parts.
# SUGGESTION(@paulray): How about adding some documentation here
# describing why the fractional part is reduced to [-0.5,0.5] instead of [0,1].
# I think I understand it, but it would be good to have it stated.
# Also, probably one of the comparisons below should be <= or >=, so the
# range is [-0.5,0.5) or (-0.5,0.5].
from __future__ import absolute_import, division, print_function

from collections import namedtuple

import astropy.units as u
import numpy

from pint import dimensionless_cycles


class Phase(namedtuple("Phase", "int frac")):
    """
    Phase class array version

    Ensures that the fractional part stays in [-0.5, 0.5)
    """

    __slots__ = ()

    def __new__(cls, arg1, arg2=None):
        # Assume inputs are numerical, could add an extra
        # case to parse strings as input.
        # if it is not a list, convert to a list
        if not hasattr(arg1, "unit"):
            arg1 = arg1 * u.cycle
        if arg1.shape == ():
            arg1 = arg1.reshape((1,))
        with u.set_enabled_equivalencies(dimensionless_cycles):
            arg1 = arg1.to(u.Unit(""))
            # Since modf does not like dimensioned quantity
            if arg2 is None:
                ff, ii = numpy.modf(arg1)
            else:
                if not hasattr(arg2, "unit"):
                    arg2 = arg2 * u.cycle
                if arg2.shape == ():
                    arg2 = arg2.reshape((1,))
                arg2 = arg2.to(u.Unit(""))
                arg1S = numpy.modf(arg1)
                arg2S = numpy.modf(arg2)
                ii = arg1S[1] + arg2S[1]
                ff = arg2S[0]
            index = numpy.where(ff < -0.5)
            ff[index] += 1.0
            ii[index] -= 1
            # The next line is >= so that the range is the interval [-0.5,0.5)
            # Otherwise, the same phase could be represented 0,0.5 or 1,-0.5
            index = numpy.where(ff >= 0.5)
            ff[index] -= 1.0
            ii[index] += 1
            return super(Phase, cls).__new__(cls, ii.to(u.cycle), ff.to(u.cycle))

    def __neg__(self):
        # TODO: add type check for __neg__ and __add__
        return Phase(-self.int, -self.frac)

    def __add__(self, other):
        ff = self.frac + other.frac
        with u.set_enabled_equivalencies(dimensionless_cycles):
            ii = numpy.modf(ff.to(u.Unit("")))[1]
            ii = ii.to(u.cycle)
            return Phase(self.int + other.int + ii, ff - ii)

    def __sub__(self, other):
        return self.__add__(other.__neg__())
