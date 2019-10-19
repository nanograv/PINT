from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np

from pint.models.parameter import floatParameter, prefixParameter
from pint.models.timing_model import PhaseComponent, MissingParameter


class IFunc(PhaseComponent):
    """This class implements tabulated delays.

    These mimic a tempo2 feature, which supports piecewise, linear, and sinc
    interpolation.  The implementation here currently only supports the first
    two formulae.

    For consistency with tempo2, although the IFuncs represent time series,
    they are converted to phase simply by multiplication with F0, therefore
    changing PEPOCH should be done with care.

    The format of IFuncs in an ephemeris is:
    SIFUNC X 0
    IFUNC1 MJD1 DT1 0.0
    IFUNC2 MJD2 DT2 0.0
    ...
    IFUNCN MJDN DTN 0.0

    X indicates the type of interpolation:
    0 == piecewise (no interpolation)
    1 == sinc (not supported)
    2 == linear

    Note that in tempo2, the interpolants are formed from the sideral arrival
    time.  I have chosen instead to use the barycentric time.  This should not
    make much of a difference since these functions are typically treating
    slow phase variations.
    """

    register = True
    category = "ifunc"

    def __init__(self):
        super(IFunc, self).__init__()

        self.add_param(
            floatParameter(name="SIFUNC", description="Type of interpolation", units="")
        )
        self.add_param(
            prefixParameter(
                name="IFUNC1",
                units="s",
                description="Interpolation Components (MJD+delay)",
                type_match="pair",
                long_double=True,
                parameter_type="pair",
            )
        )
        self.phase_funcs_component += [self.ifunc_phase]

    def setup(self):
        super(IFunc, self).setup()
        if self.SIFUNC.quantity is None:
            raise MissingParameter(
                "IFunc", "SIFUNC", "SIFUNC is required if IFUNC entries are present."
            )
        if (not hasattr(self, "F0")) or (self.F0.quantity is None):
            raise MissingParameter(
                "IFunc", "F0", "F0 is required if IFUNC entries are present."
            )

        # this is copied from the wave model, but I don't think this check
        # is strictly necessary.  An ephemeris could remain perfectly valid
        # if some IFUNC terms were "missing".  (The same is true for WAVE.)
        terms = list(self.get_prefix_mapping_component("IFUNC").keys())
        terms.sort()
        for i, term in enumerate(terms):
            if (i + 1) != term:
                raise MissingParameter("IFunc", "IFUNC%d" % (i + 1))

        self.num_terms = len(terms)

    def print_par(self,):
        result = self.SIFUNC.as_parfile_line()
        terms = ["IFUNC%d" % ii for ii in range(1, self.num_terms + 1)]
        for ft in terms:
            par = getattr(self, ft)
            result += par.as_parfile_line()

        return result

    def ifunc_phase(self, toas, acc_delay=None):
        names = ["IFUNC%d" % ii for ii in range(1, self.num_terms + 1)]
        terms = [getattr(self, name) for name in names]

        # the MJDs(x) and offsets (y) of the interpolation points
        x, y = np.asarray([t.quantity for t in terms]).T
        # use barycentric times for interpolation
        ts = toas.table["tdbld"]
        delays = np.zeros(len(ts))

        # Determine what type of interpolation we are doing.
        itype = int(self.SIFUNC.quantity)

        if itype == 0:
            # piecewise interpolation.  Following the tempo2 convention, the
            # interpolating value is selected as the nearest preceding point.
            # To avoid jumps, we apply the first interpolation value to any
            # TOAs preceding the first tabulated offset.
            idx = np.searchsorted(x, ts) + 1
            idx[ts < x[0]] = 0
            idx[ts >= x[1]] = len(x) - 1
            delays[:] = y[idx]
        elif itype == 2:
            idx = np.searchsorted(x, ts)
            mask = (idx > 0) & (idx < len(x))
            im = idx[mask]
            dx1 = ts[mask] - x[im - 1]
            dx2 = x[im] - ts[mask]
            delays[mask] = (y[im] * dx1 + y[im - 1] * dx2) / (dx1 + dx2)
            # now handle edge cases
            delays[idx == 0] = y[0]
            delays[idx == len(x)] = y[-1]
        else:
            raise ValueError("Interpolation type %d not supported.".format(itype))

        phase = ((delays * u.s) * self.F0.quantity * 2 * np.pi).to(u.cycle)
        return phase
