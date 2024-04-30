"""The DDK model - Damour and Deruelle with kinematics."""
import warnings
import numpy as np
from astropy import units as u
from loguru import logger as log

from pint.models.binary_dd import BinaryDD
from pint.models.parameter import boolParameter, floatParameter, funcParameter
from pint.models.stand_alone_psr_binaries.DDK_model import DDKmodel
from pint.models.timing_model import MissingParameter, TimingModelError


def _convert_kin(kin):
    """Convert DDK KIN to/from IAU/DT92 conventions

    Parameters
    ----------
    kin : astropy.units.Quantity

    Returns
    -------
    astropy.units.Quantity
        Value returned in other convention
    """
    return 180 * u.deg - kin


def _convert_kom(kom):
    """Convert DDK KOM to/from IAU/DT92 conventions

    Parameters
    ----------
    kom : astropy.units.Quantity

    Returns
    -------
    astropy.units.Quantity
        Value returned in other convention
    """
    return 90 * u.deg - kom


class BinaryDDK(BinaryDD):
    """Damour and Deruelle model with kinematics.

    This extends the :class:`pint.models.binary_dd.BinaryDD` model with
    "Shklovskii" and "Kopeikin" terms that account for the finite distance
    of the system from Earth, the finite size of the system, and the
    interaction of these with the proper motion.

    From Kopeikin (1995) this includes :math:`\Delta_{\pi M}` (Equation 17), the mixed annual-orbital parallax term, which changes :math:`a_1` and :math:`\omega`
    (:meth:`~pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel.delta_a1_parallax` and :meth:`~pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel.delta_omega_parallax`).

    It does not include :math:`\Delta_{\pi P}`, the pure pulsar orbital parallax term (Equation 14).

    From Kopeikin (1996) this includes apparent changes in :math:`\omega`, :math:`a_1`, and :math:`i` due to the proper motion
    (:meth:`~pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel.delta_omega_proper_motion`, :meth:`~pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel.delta_a1_proper_motion`,
    :meth:`~pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel.delta_kin_proper_motion`) (Equations 8, 9, 10).

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel`.

    It supports all the parameters defined in :class:`pint.models.pulsar_binary.PulsarBinary`
    and :class:`pint.models.binary_dd.BinaryDD` plus:

       KIN
            the inclination angle: :math:`i`
       KOM
            the longitude of the ascending node, Kopeikin (1995) Eq 9: :math:`\Omega`
       K96
            flag for Kopeikin binary model proper motion correction

    It also removes:

       SINI
            use ``KIN`` instead

    Note
    ----
    This model defines KOM with reference to east, either equatorial or ecliptic depending on how the model is defined.
    KOM and KIN are defined in the Damour & Taylor (1992) convention (DT92), where:

        KIN = 180 deg means the orbital angular momentum vector points toward the Earth, and KIN = 0 means the orbital angular momentum vector points away from the Earth.

        KOM is 0 toward the East and increases clockwise on the sky; it is measured "East through North."


    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_ddk.BinaryDDK

    References
    ----------
    - Kopeikin (1995), ApJ, 439, L5 [1]_
    - Kopeikin (1996), ApJ, 467, L93 [2]_
    - Damour & Taylor (1992), Phys Rev D, 45, 1840 [3]_

    .. [1] https://ui.adsabs.harvard.edu/abs/1995ApJ...439L...5K/abstract
    .. [2] https://ui.adsabs.harvard.edu/abs/1996ApJ...467L..93K/abstract
    .. [3] https://ui.adsabs.harvard.edu/abs/1992PhRvD..45.1840D/abstract

    """

    register = True

    def __init__(
        self,
    ):
        super().__init__()
        self.binary_model_name = "DDK"
        self.binary_model_class = DDKmodel

        self.add_param(
            floatParameter(
                name="KIN", value=0.0, units="deg", description="Inclination angle"
            )
        )
        self.add_param(
            floatParameter(
                name="KOM",
                value=0.0,
                units="deg",
                description="The longitude of the ascending node",
            )
        )
        self.add_param(
            boolParameter(
                name="K96",
                description="Flag for Kopeikin binary model proper motion"
                " correction",
            )
        )
        self.remove_param("SINI")
        self.internal_params += ["PMLONG_DDK", "PMLAT_DDK"]

        self.add_param(
            funcParameter(
                name="KINIAU",
                description="Inclination angle in the IAU convention",
                params=("KIN",),
                func=_convert_kin,
                units="deg",
            )
        )
        self.add_param(
            funcParameter(
                name="KOMIAU",
                description="The longitude of the ascending node in the IAU convention",
                params=("KOM",),
                func=_convert_kom,
                units="deg",
            )
        )
        self.add_param(
            funcParameter(
                name="SINI",
                description="Sine of inclination angle",
                params=("KIN",),
                func=np.sin,
                units="",
            )
        )

    @property
    def PMLONG_DDK(self):
        """Proper motion in longitude (RA or ecliptic longitude)"""
        if "AstrometryEquatorial" in self._parent.components:
            return self._parent.PMRA
        elif "AstrometryEcliptic" in self._parent.components:
            return self._parent.PMELONG
        else:
            raise TimingModelError(
                "No valid AstrometryEcliptic or AstrometryEquatorial component found"
            )

    @property
    def PMLAT_DDK(self):
        """Proper motion in latitude (Dec or ecliptic latitude)"""
        if "AstrometryEquatorial" in self._parent.components:
            return self._parent.PMDEC
        elif "AstrometryEcliptic" in self._parent.components:
            return self._parent.PMELAT
        else:
            raise TimingModelError(
                "No valid AstrometryEcliptic or AstrometryEquatorial component found"
            )

    def validate(self):
        """Validate parameters."""
        super().validate()
        if "AstrometryEquatorial" in self._parent.components:
            log.debug("Validating DDK model in ICRS coordinates")
            if "PMRA" not in self._parent.params or "PMDEC" not in self._parent.params:
                raise MissingParameter(
                    "DDK", "DDK model needs proper motion parameters."
                )
        elif "AstrometryEcliptic" in self._parent.components:
            log.debug("Validating DDK model in ECL coordinates")
            if (
                "PMELONG" not in self._parent.params
                or "PMELAT" not in self._parent.params
            ):
                raise MissingParameter(
                    "DDK", "DDK model needs proper motion parameters."
                )
        else:
            raise TimingModelError(
                "No valid AstrometryEcliptic or AstrometryEquatorial component found"
            )

        if not hasattr(self._parent, "PX"):
            raise MissingParameter(
                "Binary_DDK", "PX", "DDK model needs PX from" "Astrometry."
            )

        if self._parent.PX.value <= 0.0 or self._parent.PX.value is None:
            raise TimingModelError("DDK model needs a valid `PX` value.")
        if "A1DOT" in self.params and self.A1DOT.value != 0:
            warnings.warn("Using A1DOT with a DDK model is not advised.")

    def alternative_solutions(self):
        """Alternative Kopeikin solutions (potential local minima)

        There are 4 potential local minima for a DDK model where a1dot is the same
        These are given by where Eqn. 8 in Kopeikin (1996) is equal to the best-fit value.

        We first define the symmetry point where a1dot is zero (in equatorial coordinates):

        :math:`KOM_0 = \\tan^{-1} (\mu_{\delta} / \mu_{\\alpha})`

        The solutions are then:

        :math:`(KIN, KOM)`

        :math:`(KIN, 2KOM_0 - KOM - 180^{\circ})`

        :math:`(180^{\circ}-KIN, KOM+180^{\circ})`

        :math:`(180^{\circ}-KIN, 2KOM_0 - KOM)`

        All values will be between 0 and :math:`360^{\circ}`.

        Returns
        -------
        tuple :
            tuple of (KIN,KOM) pairs for the four potential solutions
        """
        x0 = self.KIN.quantity
        y0 = self.KOM.quantity
        solutions = [(x0, y0)]
        # where Eqn. 8 in Kopeikin (1996) that is equal to 0
        KOM_zero = np.arctan2(self.PMLAT_DDK.quantity, self.PMLONG_DDK.quantity).to(
            u.deg
        )
        # second one in the same banana
        solutions += [(x0, (2 * (KOM_zero) - y0 - 180 * u.deg) % (360 * u.deg))]
        # and the other banana
        solutions += [
            ((180 * u.deg - x0) % (360 * u.deg), (2 * (KOM_zero) - y0) % (360 * u.deg))
        ]
        solutions += [
            ((180 * u.deg - x0) % (360 * u.deg), (y0 + 180 * u.deg) % (360 * u.deg))
        ]
        return solutions
