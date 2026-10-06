.. highlight:: shell

How PINT converts TCB and TDB parameters
----------------------------------------

PINT evaluates timing models in TDB and does not evaluate ``UNITS TCB``
models directly. The converter therefore maps a TCB parameterization onto
PINT's existing TDB forward model.

Coordinate epochs are converted with :class:`astropy.time.Time`, which uses
ERFA's IAU 2006 TCB/TDB realization, including the defining ``TDB0`` offset.
The implementation does not duplicate the IAU constants.

For intervals, define ``F = d(TDB)/d(TCB) = 1 - L_B`` and ``K = 1/F``::

    dt_tdb = F * dt_tcb

Since the definition of the second is changing, all parameters involved in the timing model
must also be transformed. In the simplest case, if a quantity x has dimensions of [T^n], it
will be transformed as::
    
    x_tdb = x_tcb / K^n

This rule applies to the majority of parameters.

However, there are some parameters in pulsar timing which appear in the timing model multiplied 
by some constants. Examples include

    1. DM appears as DMconst * DM
    2. A1 appears as A1 / c
    3. M2 appears as M2 * G / c^3

In these cases, PINT keeps the multiplication factor numerically constant during
TCB <-> TDB conversion and absorbs the conversion into the parameter. If a
parameter has such a factor, specify it using the ``tcb2tdb_scale_factor``
argument while constructing the `Parameter` object. For example, DM will have 
`tcb2tdb_scale_factor=DMconst`.

Note that the parameter multiplied by the constant in these cases has dimensions of the 
form [T^n]. In the above cases, the value of n is as follows.

    1. A1 has n = 1
    2. M2 has n = 1

In general, if a parameter x appears in the timing model as C*x and if C*x has dimensionality of
the form [T^n], the scaling should be done with the "effective dimensionality" n.

If a parameter doesn't have a dimensionality of [T^n], a general rule is to reorganize the 
factors in the equation such that each group has a dimensionality [T^n]. This is ALWAYS possible
because the timing model components produce either a delay ([T^1]) or a phase ([T^0]).

A useful trick is to express parameters in geometrized units, where everything
has dimensions of ``T^n``. This is only the default: component-owned conversion
metadata overrides it when the forward model fixes an independent variable.

Radio-frequency decision
~~~~~~~~~~~~~~~~~~~~~~~~

PINT supports ``DILATEFREQ N``. Its barycentric radio frequency applies the
Earth-motion Doppler correction but is not dilated by ``K`` during unit
conversion. The converter holds that frequency numerically fixed.

Consequently, deterministic dispersion delays scale by ``F``. For the DM
Taylor series, order ``q`` scales as::

    DM_tdb^(q) = K^(q-1) DM_tcb^(q)

Thus DM scales by ``F``, DM1 is unchanged, and DM2 scales by ``K``. This
order-aware rule is represented by component-owned metadata rather than a
global parameter-name table.

Explicit invariants
~~~~~~~~~~~~~~~~~~~

PX is left numerically unchanged because PINT does not implement a matching
TCB/TDB spatial-coordinate transformation. ``START`` and ``FINISH`` are
data-span selectors and are also unchanged. UTC interval selectors such as
DMX range boundaries remain UTC and are not coordinate epochs.

Exceptions to this are noise parameters. The TOA uncertainties are measured in the observatory 
timescale and are not converted into TCB or TDB before computing the likelihood function/
chi-squared. Hence, we don't convert the quantities that modify TOA uncertainties, namely EFACs and``
EQUADs. Since we are not converting TOA variances, it doesn't make sense to convert TOA covariances
either. Hence, ECORRs and red and DM noise parameters are not converted. This means that 
the noise parameters must ALWAYS be re-estimated after a TCB <-> TDB conversion.

FD and FDJUMP coefficients are time-valued amplitudes evaluated at PINT's
fixed radio frequency, so every coefficient scales by ``F``. No logarithmic
coefficient mixing is needed.

The DM family is the one place where ``DILATEFREQ`` changes the answer, and
the flag that matters is the one on the **TCB** side of the conversion. A
dilating engine divides the barycentric frequency by the Einstein rate; on the
TCB side that divisor carries the full ``K`` (1.55e-8), while on the TDB side
it departs from 1 by only ~5e-10. Requiring the physical delay to be the same
object in both unit systems therefore gives ``K**(q+1)`` for a dilated TCB
source and ``K**(q-1)`` for an undilated one, where ``q`` is the Taylor order.
Choosing the wrong branch costs ``K**2 - 1 = 3.1e-8`` of the dispersion delay,
which is ~3 ns at 1.4 GHz and ~40 ns at 400 MHz for DM = 50; being chromatic,
it is not absorbed by the phase gauge. TEMPO2 defaults to ``DILATEFREQ Y`` and
writes it into the TCB par files it produces, so this is the common case.

PINT evaluates undilated frequencies regardless, and converts a dilated source
with the dilated exponents. It reports the frequency-dependent components as
unaudited in that case: the exponents are right, but PINT cannot certify them
by closure, because it cannot evaluate a dilated model, and its undilated
evaluation of the converted par still differs from a dilated one by
``E**2 - 1 ~ 1e-9`` of the dispersion delay (~1 ns at 400 MHz for DM = 50).
That residue is an annual signal and no parameter value can absorb it.

Unsupported active deterministic terms are left unchanged and reported. They
do not prevent supported parameters from being converted. A conversion is
covered by the no-refit accuracy contract only when its report is accepted.

The contract is literal, and the noise parameters above are its only
carve-out. With the TOAs, the clock chain and the solar-system ephemeris held
fixed, an accepted conversion reproduces the source model's residuals to
better than 1 ns and no deterministic parameter is refitted. The one degree of
freedom left open is the overall phase gauge, the constant that ``TZRMJD``, a
subtracted mean residual or a reference ``JUMP`` fixes; timing packages pick
that constant by differing conventions, so the bound is stated on the shape of
the residuals rather than on their absolute offset.
