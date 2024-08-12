.. highlight:: shell

How to determine the scaling factor for TCB <-> TDB conversion for a parameter
------------------------------------------------------------------------------

The TCB and TDB timescales differ by a constant factor in the definition of the second.
This means that the epochs and TOAs must be scaled according to the following expression::
    
    t_tdb = (t_tcb - IFTE_MJD0) / IFTE_K + IFTE_MJD0

Similarly, a time difference will be scaled as::
    
    dt_tdb = dt_tcb / IFTE_K

Since the definition of the second is changing, all parameters involved in the timing model
must also be transformed. In the simplest case, if a quantity x has dimensions of [T^n], it
will be transformed as::
    
    x_tdb = x_tcb / IFTE_K^n

This rule applies to the majority of parameters.

However, there are some parameters in pulsar timing which appear in the timing model multiplied 
by some constants. Examples include

    1. DM appears as DMconst * DM
    2. A1 appears as A1 / c
    3. PX appears as PX * c / AU 
    4. M2 appears as M2 * G / c^3

In these cases, it is customary to keep the multiplication factor numerically constant during 
TCB <-> TDB conversion, and to absorb the effect of converting this constant into the parameter
itself. If a parameter has such a factor, it must be specified using the `tcb2tdb_scale_factor`
argument while constructing the `Parameter` object. For example, DM will have 
`tcb2tdb_scale_factor=DMconst`.

Note that the parameter multiplied by the constant in these cases has dimensions of the 
form [T^n]. In the above cases, the value of n is as follows.

    1. DM has n = -1
    2. A1 has n = 1
    3. PX has n = -1
    4. M2 has n = 1

In general, if a parameter x appears in the timing model as C*x and if C*x has dimensionality of
the form [T^n], the scaling should be done with the "effective dimensionality" n.

If a parameter doesn't have a dimensionality of [T^n], a general rule is to reorganize the 
factors in the equation such that each group has a dimensionality [T^n]. This is ALWAYS possible
because the timing model components produce either a delay ([T^1]) or a phase ([T^0]).

A useful trick for doing this type of transformation is to express the parameters in geometrized 
units, where everything has dimensions of [T^n]. For example, a mass M will be expressed as M*(G/c^3)
(e.g., M2, MTOT), and a distance L will be expressed as L/c (e.g., A1). Please note that this may not 
work in every case, and each parameter should be treated in a case-by-case basis depending on the 
delay/phase expression they appear in.

Exceptions to this are noise parameters. The TOA uncertainties are measured in the observatory 
timescale and are not converted into TCB or TDB before computing the likelihood function/
chi-squared. Hence, we don't convert the quantities that modify TOA uncertainties, namely EFACs and``
EQUADs. Since we are not converting TOA variances, it doesn't make sense to convert TOA covariances
either. Hence, ECORRs and red and DM noise parameters are not converted. This means that 
the noise parameters must ALWAYS be re-estimated after a TCB <-> TDB conversion.

Another exception to this are FD parameters and FD jumps, which involve polynomials of logarithms.
Such functions transform in a non-linear fashion during TCB <-> TDB conversion, and we have chosen
not to apply the conversion to such parameters. They too must be re-estimated after a TCB <-> TDB 
conversion.

In such cases, the parameter can be excluded from TCB <-> TDB conversion by passing `convert_tcb2tdb=False`
into the `Parameter` class constructor.
