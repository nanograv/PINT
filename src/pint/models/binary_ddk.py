"""The DDK model - Damour and Deruelle with kinematics."""
from pint.models.binary_dd import BinaryDD
from pint.models.parameter import boolParameter, floatParameter
from pint.models.stand_alone_psr_binaries.DDK_model import DDKmodel
from pint.models.timing_model import MissingParameter, TimingModelError


class BinaryDDK(BinaryDD):
    """Damour and Deruelle model with kinematics.

    This extends the :class:`pint.models.binary_dd.BinaryDD` model with
    "Shklovskii" and "Kopeikin" terms that account for the finite distance
    of the system from Earth, the finite size of the system, and the
    interaction of these with the proper motion.

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.DDK_model.DDKmodel`.

    It supports all the parameters defined in :class:`pint.models.pulsar_binary.PulsarBinary`
    and :class:`pint.models.pulsar_binary.BinaryDDK` plus:

        - KIN - inclination angle (deg)
        - KOM - the longitude of the ascending node, Kopeikin (1995) Eq 9. OMEGA (deg)
        - K96 - flag for Kopeikin binary model proper motion correction

    It also removes:

        - SINI - use KIN instead

    Note
    ----
    This model defines KOM with reference to celestial north regardless of the astrometry
    model. This is incompatible with tempo2, which defines KOM with reference to ecliptic
    north when using ecliptic coordinates.

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_ddk.BinaryDDK

    References
    ----------
    KOPEIKIN. 1995, 1996
    """

    register = True

    def __init__(self,):
        super(BinaryDDK, self).__init__()
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
        self.internal_params += ["PMRA_DDK", "PMDEC_DDK"]

    @property
    def PMRA_DDK(self):
        params = self._parent.get_params_as_ICRS()
        par_obj = floatParameter(
            name="PMRA",
            units="mas/year",
            value=params["PMRA"],
            description="Proper motion in RA",
        )
        return par_obj

    @property
    def PMDEC_DDK(self):
        params = self._parent.get_params_as_ICRS()
        par_obj = floatParameter(
            name="PMDEC",
            units="mas/year",
            value=params["PMDEC"],
            description="Proper motion in DEC",
        )
        return par_obj

    def validate(self):
        """Validate parameters."""
        super().validate()
        if "PMRA" not in self._parent.params or "PMDEC" not in self._parent.params:
            # Check ecliptic coordinates proper motion.
            if (
                "PMELONG" not in self._parent.params
                or "PMELAT" not in self._parent.params
            ):
                raise MissingParameter(
                    "DDK", "DDK model needs proper motion parameters."
                )

        if hasattr(self._parent, "PX"):
            if self._parent.PX.value <= 0.0 or self._parent.PX.value is None:
                raise TimingModelError("DDK model needs a valid `PX` value.")
        else:
            raise MissingParameter(
                "Binary_DDK", "PX", "DDK model needs PX from" "Astrometry."
            )
        # Should we warn if the model is using ecliptic coordinates?
        # Should we support KOM_PINT that works this way and KOM that works the way tempo2 does?
