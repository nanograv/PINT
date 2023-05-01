"""Tabulated extra delays."""
import astropy.units as u
import numpy as np

from pint.models.parameter import floatParameter, prefixParameter
from pint.models.timing_model import PhaseComponent, MissingParameter


class IFunc(PhaseComponent):
    """This class implements tabulated delays.

    These mimic a tempo2 feature, which supports piecewise, linear, and sinc
    interpolation.  The implementation here currently only supports the
    first two formulae.

    For consistency with tempo2, although the IFuncs represent time series,
    they are converted to phase simply by multiplication with F0, therefore
    changing PEPOCH should be done with care.

    The format of IFuncs in an ephemeris is::

        SIFUNC X 0
        IFUNC1 MJD1 DT1 0.0
        IFUNC2 MJD2 DT2 0.0
        ...
        IFUNCN MJDN DTN 0.0

    X indicates the type of interpolation:

        - 0 == piecewise constant (no interpolation)
        - 1 == sinc (not supported)
        - 2 == linear

    NB that the trailing 0.0s are necessary for accurate tempo2 parsing.
    NB also that tempo2 has a static setting MAX_IFUNC whose default value
    is 1000.

    Note that in tempo2, the interpolants are formed from the sidereal
    arrival time, which means that different observatories actually see
    different timing noise processes!  Here, we interpret the "x axis" as
    barycentered times, so that all observatories see the same realization
    of the interpolated signal.  Because the interpolant spacing is
    typically large (days to weeks), the difference between SAT and BAT of
    a few minutes should make little difference.

    Parameters supported:

    .. paramtable::
        :class: pint.models.ifunc.IFunc
    """

    register = True
    category = "ifunc"

    def __init__(self):
        super().__init__()

        self.add_param(
            floatParameter(name="SIFUNC", description="Type of interpolation", units="")
        )
        self.add_param(
            prefixParameter(
                name="IFUNC1",
                units="s",
                description="Interpolation control point pair (MJD, delay)",
                type_match="pair",
                long_double=True,
                parameter_type="pair",
            )
        )
        self.phase_funcs_component += [self.ifunc_phase]

    def setup(self):
        super().setup()
        self.terms = list(self.get_prefix_mapping_component("IFUNC").keys())
        self.num_terms = len(self.terms)

    def validate(self):
        super().validate()
        if self.SIFUNC.quantity is None:
            raise MissingParameter(
                "IFunc", "SIFUNC", "SIFUNC is required if IFUNC entries are present."
            )
        if (not hasattr(self._parent, "F0")) or (self._parent.F0.quantity is None):
            raise MissingParameter(
                "IFunc", "F0", "F0 is required if IFUNC entries are present."
            )

        # this is copied from the wave model, but I don't think this check
        # is strictly necessary.  An ephemeris could remain perfectly valid
        # if some IFUNC terms were "missing".  (The same is true for WAVE.)
        self.terms.sort()
        for i, term in enumerate(self.terms):
            if (i + 1) != term:
                raise MissingParameter("IFunc", "IFUNC%d" % (i + 1))

    def print_par(self, format="pint"):
        result = self.SIFUNC.as_parfile_line(format=format)
        terms = ["IFUNC%d" % ii for ii in range(1, self.num_terms + 1)]
        for ft in terms:
            par = getattr(self, ft)
            result += par.as_parfile_line(format=format)

        return result

    def ifunc_phase(self, toas, delays):
        names = ["IFUNC%d" % ii for ii in range(1, self.num_terms + 1)]
        terms = [getattr(self, name) for name in names]

        # the MJDs(x) and offsets (y) of the interpolation points
        x, y = np.asarray([t.quantity for t in terms]).T
        # form barycentered times
        ts = toas.table["tdbld"] - delays.to(u.day).value
        times = np.zeros(len(ts))

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
            times[:] = y[idx]
        elif itype == 2:
            idx = np.searchsorted(x, ts)
            mask = (idx > 0) & (idx < len(x))
            im = idx[mask]
            dx1 = ts[mask] - x[im - 1]
            dx2 = x[im] - ts[mask]
            times[mask] = (y[im] * dx1 + y[im - 1] * dx2) / (dx1 + dx2)
            # now handle edge cases
            times[idx == 0] = y[0]
            times[idx == len(x)] = y[-1]
        else:
            raise ValueError(f"Interpolation type {itype} not supported.")

        return ((times * u.s) * self._parent.F0.quantity).to(u.dimensionless_unscaled)
