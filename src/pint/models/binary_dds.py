"""The DDK model - Damour and Deruelle with kinematics."""
import warnings
import numpy as np
from astropy import units as u
from loguru import logger as log

from pint.models.binary_dd import BinaryDD
from pint.models.parameter import boolParameter, floatParameter
from pint.models.stand_alone_psr_binaries.DDS_model import DDSmodel
from pint.models.timing_model import MissingParameter, TimingModelError


class BinaryDDS(BinaryDD):
    """Damour and Deruelle model with alternate Shapiro delay parameterization.

    This extends the :class:`pint.models.binary_dd.BinaryDD` model with
    :math:`SHAPMAX = -\log(1-s)` instead of just :math:`s=\sin i`, which behaves better
    for :math:`\sin i` near 1.

    The actual calculations for this are done in
    :class:`pint.models.stand_alone_psr_binaries.DDS_model.DDSmodel`.

    It supports all the parameters defined in :class:`pint.models.pulsar_binary.PulsarBinary`
    and :class:`pint.models.binary_dd.BinaryDD` plus:

       SHAPMAX
            :math:`-\log(1-\sin i)`

    It also removes:

       SINI
            use ``SHAPMAX`` instead

    Parameters supported:

    .. paramtable::
        :class: pint.models.binary_ddk.BinaryDDS

    References
    ----------
    - Kramer et al. (2006), Science, 314, 97 [1]_
    - Rafikov and Lai (2006), PRD, 73, 063003 [2]_

    .. [1] https://ui.adsabs.harvard.edu/abs/2006Sci...314...97K/abstract
    .. [2] https://ui.adsabs.harvard.edu/abs/2006PhRvD..73f3003R/abstract

    """

    register = True

    def __init__(
        self,
    ):
        super().__init__()
        self.binary_model_name = "DDS"
        self.binary_model_class = DDSmodel

        self.add_param(
            floatParameter(
                name="SHAPMAX", value=0.0, description="Function of inclination angle"
            )
        )
        self.remove_param("SINI")

    def validate(self):
        """Validate parameters."""
        super().validate()
