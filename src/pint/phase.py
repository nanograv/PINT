# -*- coding: utf-8 -*-
# Licensed under the GPLv3 - see LICENSE
"""Provide a Phase class with integer and fractional part."""

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle, Longitude
from astropy.time.utils import two_sum, two_product
from astropy.utils import minversion

from pint import dimensionless_cycles


__all__ = ['Phase', 'FractionalPhase']

NUMPY_LT_1_16 = not minversion('numpy', '1.16')

FRACTION_UFUNCS = {np.cos, np.sin, np.tan, np.spacing}

COMPARISON_UFUNCS = {
    np.equal, np.not_equal,
    np.less, np.less_equal, np.greater, np.greater_equal}


def day_frac(val1, val2, factor=None, divisor=None):
    """Return the sum of ``val1`` and ``val2`` as two float64s.

    The returned floats are an integer part and the fractional remainder,
    with the latter guaranteed to be within -0.5 and 0.5 (inclusive on
    either side, as the integer is rounded to even).

    The arithmetic is all done with exact floating point operations so no
    precision is lost to rounding error.  It is assumed the sum is less
    than about 1e16, otherwise the remainder will be greater than 1.0.

    Parameters
    ----------
    val1, val2 : array of float
        Values to be summed.
    factor : float, optional
        If given, multiply the sum by it.
    divisor : float, optional
        If given, divide the sum by it.

    Returns
    -------
    day, frac : float64
        Integer and fractional part of val1 + val2.
    """
    # Note that the version of astropy >=3.2;
    # See https://github.com/astropy/astropy/pull/8763
    # TODO: remove when we only support astropy >=3.2.
    #
    # Add val1 and val2 exactly, returning the result as two float64s.
    # The first is the approximate sum (with some floating point error)
    # and the second is the error of the float64 sum.
    sum12, err12 = two_sum(val1, val2)

    if factor is not None:
        sum12, carry = two_product(sum12, factor)
        carry += err12 * factor
        sum12, err12 = two_sum(sum12, carry)

    if divisor is not None:
        q1 = sum12 / divisor
        p1, p2 = two_product(q1, divisor)
        d1, d2 = two_sum(sum12, -p1)
        d2 += err12
        d2 -= p2
        q2 = (d1 + d2) / divisor  # 3-part float fine here; nothing can be lost
        sum12, err12 = two_sum(q1, q2)

    # get integer fraction
    day = np.around(sum12)
    extra, frac = two_sum(sum12, -day)
    frac += extra + err12
    # This part was missed in astropy...
    excess = np.around(frac)
    day += excess
    extra, frac = two_sum(sum12, -day)
    frac += extra + err12
    return day, frac


def _parse_string(s):
    """Parse a string into two doubles.

    Generally, first part before the decimal dot, second after,
    but takes account of possible exponents.
    """
    s = s.strip().lower().translate(s.maketrans('d', 'e'))
    if s[-1] == 'j':
        s = s[:-1]
        factor = 1j
    else:
        factor = 1
    if s[0] == '+':
        s = s[1:]
    elif s[0] == '-':
        s = s[1:]
        factor *= -1

    # Just test string is basically OK (and reference value below).
    test = float(s) * factor

    s_float, exp, s_exp = s.partition('e')
    s_count, sep, s_frac = s_float.rpartition('.')
    if exp:
        exponent = int(s_exp)
        if exponent < 0:
            n = min(len(s_count), -exponent)
            s_frac = s_count[-n:] + s_frac
            s_count = s_count[:-n]
            exponent += n
        elif exponent > 0:
            n = min(len(s_frac), exponent)
            s_count = s_count + s_frac[:n]
            s_frac = s_frac[n:]
            exponent -= n
        factor *= 10**exponent

    frac = float('0.' + s_frac) * factor
    count = float('0' + s_count) * factor

    assert count + frac == test

    return count, frac


_parse_strings = np.vectorize(_parse_string, otypes=[complex, complex])


class FractionalPhase(Longitude):
    """Phase without the cycle count, i.e., with a range of 1 cycle.

    This subclass of `~astropy.coordinates.Longitude` differs from it
    mostly in being able to take the fractional part of any
    `~scintillometry.phases.Phase` input.

    Parameters
    ----------
    angle : array, list, scalar, `~astropy.units.Quantity`,
        :class:`~astropy.coordinates.Angle` The angle value(s). If a tuple,
        will be interpreted as ``(h, m s)`` or ``(d, m, s)`` depending
        on ``unit``. If a string, it will be interpreted following the
        rules described for :class:`~astropy.coordinates.Angle`.
    unit : :class:`~astropy.units.UnitBase`, str, optional
        The unit of the value specified for the angle.  This may be any
        string that `~astropy.units.Unit` understands.  Must be an angular
        unit.  Default is 'cycle'.
    wrap_angle : :class:`~astropy.coordinates.Angle` or equivalent, optional
        Angle at which to wrap back to ``wrap_angle - 1 cycle``.
        If ``None`` (default), it will be taken to be 0.5 cycle unless
        ``angle`` has a ``wrap_angle`` attribute.

    Raises
    ------
    `~astropy.units.UnitsError`
        If a unit is not provided or it is not an angular unit.
    `TypeError`
        If the angle parameter is a :class:`~astropy.coordinates.Latitude`.
    """
    _default_wrap_angle = Angle(0.5, u.cycle)
    _equivalent_unit = _default_unit = u.cycle

    def __new__(cls, angle, unit=None, wrap_angle=None, **kwargs):
        # TODO: ideally, the Longitude/Angle/Quantity initializer by
        # default tries to convert to float also for structured arrays,
        # maybe via astype.
        if isinstance(angle, Phase):
            angle = angle['frac']

        with u.add_enabled_equivalencies(dimensionless_cycles):
            return super().__new__(cls, angle, unit=unit, wrap_angle=wrap_angle,
                                   **kwargs)


def check_imaginary(a):
    """Check whether a value is purely imaginary or purely real.

    Parameters
    ----------
    a : array_like or `~numpy.ndarray` subclass

    Returns
    -------
    real, imaginary : array of float, bool
        A float array, either just the input array if not complex
        or the real or imaginary part if complex, with the bool indicating
        whether it is the imaginary part.

    Raises
    ------
    ValueError
        If the input is complex and is not purely real or purely imaginary.
    """
    if np.iscomplexobj(a):
        if np.all(a.real == 0):
            return a.imag, True
        elif np.all(a.imag == 0):
            return a.real, False
        else:
            raise ValueError("cannot have mixed real/imaginary Phase")
    else:
        return a, False


class Phase(Angle):
    """Represent two-part phase.

    With one part the integer cycle count and the other the fractional phase.

    The class is a high-precision version of `~astropy.coordinates.Angle`
    that aims to behave just like it, but use its precision in operations
    where possible -- decaying to a `~astropy.units.Quantity` otherwise.

    For instance, addition and subtraction preserve precision, as do
    multiplication and division with dimensionless quantities.  Similarly,
    trigonometric functions use just the fractional phase.

    The phase can either be purely real or purely imaginary, not mixed.  If
    imaginary, using it in `~numpy.exp` will again preserve precision.

    Parameters
    ----------
    phase1, phase2 : array or `~astropy.units.Quantity`
        Two-part phase.  If not quantities, the assumed units are cycles.
    copy : bool, optional
        Ensure a copy is made.  Only relevant if ``phase1`` is a `Phase`
        and ``phase2`` is not given.
    subok : bool, optional
        If `False` (default), the returned array will be forced to be a
        `Phase`.  Otherwise, `Phase` subclasses will be passed through.
        Only relevant if ``phase1`` or ``phase2`` is a `Phase` subclass.

    Notes
    -----
    The machinery to keep precision is not complete; in particular, reductions
    such as summing along an axis will currently lose precision.

    Strings passed in to ``phase1`` or ``phase2`` are first converted to
    standard doubles, which may lead to loss of precision.  For long strings,
    use the `~scintillometry.phases.Phase.from_string` class method instead.
    """

    _equivalent_unit = _unit = _default_unit = u.cycle
    _phase_dtype = np.dtype({'names': ['int', 'frac'],
                             'formats': ['f8']*2})

    def __new__(cls, phase1, phase2=None, copy=True, subok=False):
        if isinstance(phase1, Phase):
            if phase2 is not None:
                phase1 = phase1 + phase2
                copy = False
            if not subok and type(phase1) is not cls:
                phase1 = phase1.view(cls)
            return phase1.copy() if copy else phase1

        with u.add_enabled_equivalencies(dimensionless_cycles):
            phase1 = Angle(phase1, cls._unit, copy=False)

        if phase2 is not None:
            if isinstance(phase2, Phase):
                phase2 = phase2 + phase1
                if not subok and type(phase2) is not cls:
                    phase2 = phase2.view(cls)
                return phase2

            with u.add_enabled_equivalencies(dimensionless_cycles):
                phase2 = Angle(phase2, cls._unit, copy=False)

        return cls.from_angles(phase1, phase2)

    @classmethod
    def from_angles(cls, phase1, phase2=None, factor=None, divisor=None,
                    out=None):
        """Create a Phase instance from two angles.

        The two angles will be added, and possibly multiplied by a factor or
        divided by a divisor, preserving precision using two doubles, one for
        the integer part and one for the remainder.

        Note that this class method is mostly meant for internal use.

        Parameters
        ----------
        phase1 : `~astropy.units.Quantity`
             With angular units.
        phase2 : `~astropy.units.Quantity` or `None`
             With angular units.
        factor : float or complex
             Posisble factor to multiply the angles with.
        divisor : float or complex
             Posisble divisor to divide the angles by.

        Raises
        ------
        ValueError
            If the result is not purely real or purely imaginary
        """
        phase1, imaginary = check_imaginary(phase1)
        if phase2 is not None:
            phase2, im2 = check_imaginary(phase2)
            if im2 is not imaginary:
                raise ValueError("phase1 and phase2 must either be both "
                                 "real or both imaginary.")
        if factor is not None:
            factor, imf = check_imaginary(factor)
            imaginary ^= imf
        if divisor is not None:
            divisor, imd = check_imaginary(divisor)
            if imd and not imaginary:
                divisor = -divisor
            imaginary ^= imd

        # TODO: would be nice if day_frac had an out parameter.
        phase1_value = phase1.to_value(cls._unit)
        if phase2 is None:
            phase2_value = 0.
        else:
            phase2_value = phase2.to_value(cls._unit)
        count, fraction = day_frac(phase1_value, phase2_value,
                                   factor=factor, divisor=divisor)
        if out is None:
            value = np.empty(count.shape, cls._phase_dtype)
            out = value.view(cls)
        else:
            value = out.view(np.ndarray)
        value['int'] = count
        value['frac'] = fraction
        out.imaginary = imaginary
        return out

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.imaginary = getattr(obj, 'imaginary', False)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if result.dtype is self.dtype:
            return result

        if item == 'frac':
            result = result.view(Angle if self.imaginary else FractionalPhase)
        else:
            assert item == 'int'
            result = result.view(Angle)

        if self.imaginary:
            result = result * 1j

        return result

    def __iter__(self):
        if self.isscalar:
            raise TypeError(
                "'{cls}' object with a scalar value is not iterable"
                .format(cls=self.__class__.__name__))

        # Override Quantity.__iter__ since that iterates over self.value.
        def phase_iter():
            for i in range(len(self)):
                yield self[i]

        return phase_iter()

    def _set_unit(self, unit):
        if unit is None or unit != self._unit:
            raise u.UnitTypeError(
                "{0} instances require units of '{1}'"
                .format(type(self).__name__, self._unit)
                + (", but no unit was given." if unit is None else
                   ", so cannot set it to '{0}'.".format(unit)))

        super()._set_unit(unit)

    def __repr__(self):
        v = self.view(np.ndarray)
        return "{0}({1}{3} cycle, {2}{3} cycle)".format(
            self.__class__.__name__, v['int'], v['frac'],
            " * 1j" if self.imaginary else '')

    def __str__(self):
        return self.to_string()

    def __format__(self, format_spec):
        """Format a phase, special-casing the float format.

        For the 'f' format, precision is kept, and the unit is suppressed.
        For everything else, the Quantity formatter is used.
        """
        if format_spec.endswith('f'):
            # Check that formatting works at all...
            test = format(self.value, format_spec)
            pre, dot, post = test.partition('.')
            if post:
                precise = self.to_string(precision=len(post))
                pre, _, post = precise.partition('.')
                # Just to ensure no bad rounding happened
                pre = format(float(pre), format_spec).partition('.')[0]
                return pre + dot + post

        return self.cycle.__format__(format_spec)

    def to_string(self, unit=None, decimal=True, sep='fromunit',
                  precision=None, alwayssign=False, pad=False,
                  fields=3, format=None):
        """ A string representation of the Phase.

        By default, uses a decimal representation that is guaranteed to
        preserve precision to within 1e-16 cycles.  Otherwise, uses
        `astropy.coordinates.Angle.to_string`.
        """
        if not decimal or (unit is not None and unit != u.cycle):
            return self.cycle.to_string(
                unit=unit, decimal=decimal, sep=sep, precision=precision,
                alwayssign=alwayssign, pad=pad, fields=fields,
                format=format)

        if precision is None:
            func = str
        else:
            func = ("{0:1." + str(precision) + "f}").format

        def do_format(count, frac):
            neg = (count + frac) < 0
            if neg:
                count = -count
                frac = -frac
                sign = '-'
            elif alwayssign:
                sign = '+'
            else:
                sign = ''

            if frac < 0:
                frac += 1
                count -= 1

            if frac < 0.25:
                # Ensure that we do not get 1e-16, etc., yet can use numpy's
                # guarantee that the right number of digits is shown.
                frac_str = func(frac+0.25)
                f24 = int(frac_str[2:4])
                if func is str and (len(frac_str) == 3
                                    or (len(frac_str) == 4
                                        and frac_str[3] == '5')):
                    if len(frac_str) == 3:
                        f24 = '{:02d}'.format(f24 * 10 - 25)
                    else:
                        f24 = '{:1d}'.format((f24 - 5) // 10 - 2)
                else:
                    f24 = '{:02d}'.format(f24 - 25)
                frac_str = frac_str[:2] + f24 + frac_str[4:]
            else:
                frac_str = func(frac)
                if frac_str[0] == '1':
                    count += 1
            s = sign + str(int(count)) + frac_str[1:]
            if self.imaginary:
                s += 'j'
            if format == 'latex':
                s = '$' + s + '$'
            return s

        format_ufunc = np.vectorize(do_format, otypes=['U'])
        if self.imaginary:
            count, frac = self['int'].value.imag, self['frac'].value.imag
        else:
            count, frac = self['int'].value, self['frac'].value

        result = format_ufunc(count, frac)

        if result.ndim == 0:
            result = result[()]
        return result

    @classmethod
    def from_string(cls, string):
        """Create Phase instance from a long string.

        The string has to be a standard decimal string, i.e., no attempt is
        made to parse an angle.
        """
        string = np.asanyarray(string)
        if string.dtype.kind not in 'SU':
            raise ValueError('require string input.')
        count, frac = _parse_strings(string)
        return cls(count, frac)

    @property
    def int(self):
        """Rounded cycle count."""
        return self['int']

    @property
    def frac(self):
        """Fractional phase, between -0.5 and 0.5 cycles."""
        return self['frac']

    @property
    def cycle(self):
        """Full cycle, including phase, as a regular Angle.

        The result will use a standard double, and thus likely loose precision.
        """
        return self['int'] + self['frac']

    def to_value(self, unit=None, equivalencies=[]):
        """The numerical value, possibly in a different unit.

        The result will use a standard double, and thus likely loose precision.
        """
        return self.cycle.to_value(unit, equivalencies)

    value = property(to_value,
                     doc="""The numerical value.

    The result will use a standard double, and thus likely loose precision.

    See also
    --------
    to_value : Get the numerical value in a given unit.
    """)

    def to(self, unit, equivalencies=[]):
        """The phase in a different unit.

        For any unit except "cycle", this will likely loose precision as
        an `~astropy.coordinates.Angle` or `~astropy.units.Quantity`
        is returned.
        """
        if unit == u.cycle:
            return self.copy()
        else:
            return self.cycle.to(unit, equivalencies=equivalencies)

    def _take_along_axis(self, indices, axis=None, keepdims=False):
        # More or less a straight copy from Time.
        if axis is None:
            return self[np.unravel_index(indices, self.shape)]

        if indices.ndim == self.ndim - 1:
            indices = np.expand_dims(indices, axis)

        if NUMPY_LT_1_16:
            ndim = self.ndim
            if axis < 0:
                axis = axis + ndim

            ai = tuple([
                (indices if i == axis else
                 np.arange(s).reshape((1,)*i + (s,) + (1,)*(ndim-i-1)))
                for i, s in enumerate(self.shape)])
            result = self[ai]

        else:
            result = np.take_along_axis(self, indices, axis)

        return result if keepdims else result.squeeze(axis)

    def argmin(self, axis=None, out=None):
        """Return indices of the minimum values along the given axis."""
        approx = np.min(self.cycle, axis, keepdims=True)
        dt = (self['int'] - approx) + self['frac']
        return dt.argmin(axis, out)

    def argmax(self, axis=None, out=None):
        """Return indices of the maximum values along the given axis."""
        approx = np.max(self.cycle, axis, keepdims=True)
        dt = (self['int'] - approx) + self['frac']
        return dt.argmax(axis, out)

    def argsort(self, axis=-1):
        """Returns the indices that would sort the phase array."""
        phase_approx = self.cycle
        phase_remainder = (self - phase_approx).cycle
        if axis is None:
            return np.lexsort((phase_remainder.ravel(), phase_approx.ravel()))
        else:
            return np.lexsort(keys=(phase_remainder, phase_approx), axis=axis)

    # Below are basically straight copies from Time
    def min(self, axis=None, out=None, keepdims=False):
        """Minimum along a given axis.

        This is similar to :meth:`~numpy.ndarray.min`, but adapted to ensure
        that the full precision is used.
        """
        if out is not None:
            raise ValueError("An `out` argument is not yet supported.")
        return self._take_along_axis(self.argmin(axis), axis, keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        """Maximum along a given axis.

        This is similar to :meth:`~numpy.ndarray.max`, but adapted to ensure
        that the full precision is used.
        """
        if out is not None:
            raise ValueError("An `out` argument is not yet supported.")
        return self._take_along_axis(self.argmax(axis), axis, keepdims)

    def ptp(self, axis=None, out=None, keepdims=False):
        """Peak to peak (maximum - minimum) along a given axis.

        This is similar to :meth:`~numpy.ndarray.ptp`, but adapted to ensure
        that the full precision is used.
        """
        if out is not None:
            raise ValueError("An `out` argument is not yet supported.")
        return (self.max(axis, keepdims=keepdims)
                - self.min(axis, keepdims=keepdims))

    def sort(self, axis=-1):
        """Return a copy sorted along the specified axis.

        This is similar to :meth:`~numpy.ndarray.sort`, but internally uses
        indexing with :func:`~numpy.lexsort` to ensure that the full precision
        given by the two doubles is kept.

        Parameters
        ----------
        axis : int or None
            Axis to be sorted.  If ``None``, the flattened array is sorted.
            By default, sort over the last axis.
        """
        return self._take_along_axis(self.argsort(axis), axis, keepdims=True)

    # Quantity lets ndarray.__eq__, __ne__ deal with structured arrays like us.
    # Override this so we can deal with it in __array_ufunc__.
    def __eq__(self, other):
        try:
            return np.equal(self, other)
        except u.UnitsError:
            return False
        except Exception:
            return NotImplemented

    def __ne__(self, other):
        try:
            return np.not_equal(self, other)
        except u.UnitsError:
            return True
        except Exception:
            return NotImplemented

    def __array_ufunc__(self, function, method, *inputs, **kwargs):
        """Wrap numpy ufuncs, taking care of units.

        Parameters
        ----------
        function : callable
            ufunc to wrap.
        method : str
            Ufunc method: ``__call__``, ``at``, ``reduce``, etc.
        inputs : tuple
            Input arrays.
        kwargs : keyword arguments
            As passed on, with ``out`` containing possible quantity output.

        Returns
        -------
        result : array_like
            Results of the ufunc, with the unit set properly,
            `~scintillometry.phases.Phase` if possible (i.e., with units of
            cycles), otherwise `~astropyy.units.Quantity` or `~numpy.ndarray`
            as appropriate.
        """
        # Do *not* use inputs.index(self) since that will use __eq__
        for i_self, input_ in enumerate(inputs):
            if input_ is self:
                break
        else:
            i_self += 1

        # Some bools for use in the if-statements below.
        # TODO: support reductions on add, minimum, maximum; others?
        basic = method == '__call__' and i_self < function.nin
        basic_real = basic and not self.imaginary
        basic_phase_out = basic

        out = kwargs.get('out', None)
        if out is not None:
            if len(out) == 1 and isinstance(out[0], Phase):
                phase_out = out[0]
                out = None
            else:
                phase_out = None
                basic_phase_out = False
        else:
            phase_out = None

        if function in {np.add, np.subtract} and basic and out is None:
            try:
                phases = [Phase(input_, copy=False, subok=True)
                          for input_ in inputs]
            except Exception:
                return NotImplemented

            if phases[0].imaginary == phases[1].imaginary:
                return self.from_angles(
                    function(phases[0]['int'], phases[1]['int']),
                    function(phases[0]['frac'], phases[1]['frac']),
                    out=phase_out)

        elif function in COMPARISON_UFUNCS and basic:
            phases = list(inputs)
            try:
                phases[1-i_self] = Phase(inputs[1-i_self], copy=False,
                                         subok=True)
            except Exception:
                return NotImplemented

            if phases[0].imaginary == phases[1].imaginary:
                # Going with values allows us to properly override out,
                # while if we stick with a Quantity, we run into a bug; see
                # https://github.com/astropy/astropy/issues/8764
                v0, v1 = phases[0].view(np.ndarray), phases[1].view(np.ndarray)
                diff = (v0['int'] - v1['int']) + (v0['frac'] - v1['frac'])
                return getattr(function, method)(diff, 0, **kwargs)

        elif ((function is np.multiply
               or function is np.divide and i_self == 0)
              and basic_phase_out):
            try:
                other = u.Quantity(inputs[1-i_self], u.dimensionless_unscaled,
                                   copy=False).value
                if function is np.multiply:
                    return self.from_angles(self['int'], self['frac'],
                                            factor=other, out=phase_out)
                else:
                    return self.from_angles(self['int'], self['frac'],
                                            divisor=other, out=phase_out)
            except Exception:
                # If not consistent with a dimensionless quantity, or mixed
                # real and complex, we follow the standard route of
                # downgrading ourself to a quantity and see if things work.
                pass

        elif (function in {np.floor_divide, np.remainder, np.divmod}
              and basic_real):
            fd_out = None
            if out is not None:
                if function is np.divmod:
                    fd_out = out[0]
                    phase_out = out[1]
                elif function is np.floor_divide:
                    fd_out = out[0]
            elif phase_out is not None and function is np.floor_divide:
                return NotImplemented

            fd = np.floor_divide(self.cycle, inputs[1], out=fd_out)
            corr = Phase.from_angles(inputs[1], factor=fd, out=phase_out)
            remainder = np.subtract(self, corr, out=corr)
            fdx = np.floor_divide(remainder.cycle, inputs[1])
            # This can likely be optimized...
            # Note: one cannot just loop, because rounding of exact 0.5.
            # TODO: check this method is really correct.
            if np.count_nonzero(fdx):
                fd += fdx
                corr = Phase.from_angles(inputs[1], factor=fd, out=corr)
                remainder = np.subtract(self, corr, out=corr)

            if function is np.floor_divide:
                return fd
            elif function is np.remainder:
                return remainder
            else:
                return fd, remainder

        elif function is np.positive and basic_phase_out:
            return self.from_angles(self['int'], self['frac'],
                                    out=phase_out)

        elif function is np.negative and basic_phase_out:
            return self.from_angles(-self['int'], -self['frac'],
                                    out=phase_out)

        elif function in {np.absolute, np.fabs} and basic_phase_out:
            # Go via view to avoid having to deal with imaginary.
            v = self.view(np.ndarray)
            return self.from_angles(u.Quantity(v['int'], u.cycle, copy=False),
                                    u.Quantity(v['frac'], u.cycle, copy=False),
                                    factor=np.sign(v['int'] + v['frac']),
                                    out=phase_out)

        elif function is np.rint and basic:
            return np.positive(self['int'], **kwargs)

        elif function in FRACTION_UFUNCS and basic_real:
            frac = self.frac.view(Angle)
            return function(frac, **kwargs)

        elif function is np.exp and basic and self.imaginary:
            # Avoid dimensionless_angles, but still get Quantity out.
            exponent = u.Quantity(self.frac.to_value(u.radian), copy=False)
            return function(exponent, **kwargs)

        # Fall-back: treat Phase as a simple Quantity.
        if basic:
            inputs = tuple((input_.cycle if isinstance(input_, Phase)
                            else input_) for input_ in inputs)
            quantity = inputs[i_self]
        else:
            quantity = self.cycle

        if phase_out is None:
            return quantity.__array_ufunc__(function, method, *inputs,
                                            **kwargs)
        else:
            # We won't be able to store in a phase directly, but might
            # as well use one of its elements to store the angle.
            kwargs['out'] = (phase_out['int'],)
            result = quantity.__array_ufunc__(function, method, *inputs,
                                              **kwargs)
            return phase_out.from_angles(result, out=phase_out)

    def _new_view(self, obj=None, unit=None):
        # If the unit is not right, we should ensure we change our two-float
        # dtype to a single float.
        if unit is not None and unit != self._unit:
            if obj is None:
                obj = self.cycle
            elif isinstance(obj, Phase):
                obj = obj.cycle

        return super()._new_view(obj, unit)

    def astype(self, dtype, order='K', casting='unsafe', subok=True,
               copy=True):
        """Copy of the array, cast to a specified type.

        As `numpy.ndarray.astype`, but using knowledge of format to cast to
        floats.
        """
        dtype = np.dtype(dtype)
        if not dtype.fields and casting in {'same_kind', 'unsafe'}:
            parts = [self[part].astype(dtype, order=order, casting=casting,
                                       subok=subok, copy=copy)
                     for part, copy in (('int', True), ('frac', False))]
            parts[0] += parts[1]
            return parts[0]

        else:
            return super().astype(dtype, order=order, casting=casting,
                                  subok=subok, copy=copy)
