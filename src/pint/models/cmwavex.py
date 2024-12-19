"""Chromatic variations expressed as a sum of sinusoids."""

import astropy.units as u
import numpy as np
from loguru import logger as log
from warnings import warn

from pint.models.parameter import MJDParameter, prefixParameter
from pint.exceptions import MissingParameter
from pint.models.chromatic_model import Chromatic, cmu
from pint import DMconst


class CMWaveX(Chromatic):
    """
    Fourier representation of chromatic variations.

    Used for decomposition of chromatic noise into a series of sine/cosine components with the amplitudes as fitted parameters.

    Parameters supported:

    .. paramtable::
        :class: pint.models.cmwavex.CMWaveX

    To set up a CMWaveX model, users can use the `pint.utils` function `cmwavex_setup()` with either a list of frequencies or a choice
    of harmonics of a base frequency determined by 2 * pi /Timespan
    """

    register = True
    category = "cmwavex"

    def __init__(self):
        super().__init__()
        self.add_param(
            MJDParameter(
                name="CMWXEPOCH",
                description="Reference epoch for Fourier representation of chromatic noise",
                time_scale="tdb",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_cmwavex_component(0.1, index=1, cmwxsin=0, cmwxcos=0, frozen=False)
        self.set_special_params(["CMWXFREQ_0001", "CMWXSIN_0001", "CMWXCOS_0001"])
        self.cm_value_funcs += [self.cmwavex_cm]
        self.delay_funcs_component += [self.cmwavex_delay]

    def add_cmwavex_component(
        self, cmwxfreq, index=None, cmwxsin=0, cmwxcos=0, frozen=True
    ):
        """
        Add CMWaveX component

        Parameters
        ----------

        cmwxfreq : float or astropy.quantity.Quantity
            Base frequency for CMWaveX component
        index : int, None
            Interger label for CMWaveX component. If None, will increment largest used index by 1.
        cmwxsin : float or astropy.quantity.Quantity
            Sine amplitude for CMWaveX component
        cmwxcos : float or astropy.quantity.Quantity
            Cosine amplitude for CMWaveX component
        frozen : iterable of bool or bool
            Indicates whether CMWaveX parameters will be fit

        Returns
        -------

        index : int
            Index that has been assigned to new CMWaveX component
        """

        #### If index is None, increment the current max CMWaveX index by 1. Increment using CMWXFREQ
        if index is None:
            dct = self.get_prefix_mapping_component("CMWXFREQ_")
            index = np.max(list(dct.keys())) + 1
        i = f"{int(index):04d}"

        if int(index) in self.get_prefix_mapping_component("CMWXFREQ_"):
            raise ValueError(
                f"Index '{index}' is already in use in this model. Please choose another"
            )

        if isinstance(cmwxsin, u.quantity.Quantity):
            cmwxsin = cmwxsin.to_value(cmu)
        if isinstance(cmwxcos, u.quantity.Quantity):
            cmwxcos = cmwxcos.to_value(cmu)
        if isinstance(cmwxfreq, u.quantity.Quantity):
            cmwxfreq = cmwxfreq.to_value(1 / u.d)
        self.add_param(
            prefixParameter(
                name=f"CMWXFREQ_{i}",
                description="Component frequency for Fourier representation of chromatic noise",
                units="1/d",
                value=cmwxfreq,
                parameter_type="float",
                tcb2tdb_scale_factor=u.Quantity(1),
            )
        )
        self.add_param(
            prefixParameter(
                name=f"CMWXSIN_{i}",
                description="Sine amplitudes for Fourier representation of chromatic noise",
                units=cmu,
                value=cmwxsin,
                frozen=frozen,
                parameter_type="float",
                tcb2tdb_scale_factor=DMconst,
            )
        )
        self.add_param(
            prefixParameter(
                name=f"CMWXCOS_{i}",
                description="Cosine amplitudes for Fourier representation of chromatic noise",
                units=cmu,
                value=cmwxcos,
                frozen=frozen,
                parameter_type="float",
                tcb2tdb_scale_factor=DMconst,
            )
        )
        self.setup()
        self.validate()
        return index

    def add_cmwavex_components(
        self, cmwxfreqs, indices=None, cmwxsins=0, cmwxcoses=0, frozens=True
    ):
        """
        Add CMWaveX components with specified base frequencies

        Parameters
        ----------

        cmwxfreqs : iterable of float or astropy.quantity.Quantity
            Base frequencies for CMWaveX components
        indices : iterable of int, None
            Interger labels for CMWaveX components. If None, will increment largest used index by 1.
        cmwxsins : iterable of float or astropy.quantity.Quantity
            Sine amplitudes for CMWaveX components
        cmwxcoses : iterable of float or astropy.quantity.Quantity
            Cosine amplitudes for CMWaveX components
        frozens : iterable of bool or bool
            Indicates whether sine and cosine amplitudes of CMwavex components will be fit

        Returns
        -------

        indices : list
            Indices that have been assigned to new CMWaveX components
        """

        if indices is None:
            indices = [None] * len(cmwxfreqs)
        cmwxsins = np.atleast_1d(cmwxsins)
        cmwxcoses = np.atleast_1d(cmwxcoses)
        if len(cmwxsins) == 1:
            cmwxsins = np.repeat(cmwxsins, len(cmwxfreqs))
        if len(cmwxcoses) == 1:
            cmwxcoses = np.repeat(cmwxcoses, len(cmwxfreqs))
        if len(cmwxsins) != len(cmwxfreqs):
            raise ValueError(
                f"Number of base frequencies {len(cmwxfreqs)} doesn't match number of sine ampltudes {len(cmwxsins)}"
            )
        if len(cmwxcoses) != len(cmwxfreqs):
            raise ValueError(
                f"Number of base frequencies {len(cmwxfreqs)} doesn't match number of cosine ampltudes {len(cmwxcoses)}"
            )
        frozens = np.atleast_1d(frozens)
        if len(frozens) == 1:
            frozens = np.repeat(frozens, len(cmwxfreqs))
        if len(frozens) != len(cmwxfreqs):
            raise ValueError(
                "Number of base frequencies must match number of frozen values"
            )
        #### If indices is None, increment the current max CMWaveX index by 1. Increment using CMWXFREQ
        dct = self.get_prefix_mapping_component("CMWXFREQ_")
        last_index = np.max(list(dct.keys()))
        added_indices = []
        for cmwxfreq, index, cmwxsin, cmwxcos, frozen in zip(
            cmwxfreqs, indices, cmwxsins, cmwxcoses, frozens
        ):
            if index is None:
                index = last_index + 1
                last_index += 1
            elif index in list(dct.keys()):
                raise ValueError(
                    f"Attempting to insert CMWXFREQ_{index:04d} but it already exists"
                )
            added_indices.append(index)
            i = f"{int(index):04d}"

            if int(index) in dct:
                raise ValueError(
                    f"Index '{index}' is already in use in this model. Please choose another"
                )
            if isinstance(cmwxfreq, u.quantity.Quantity):
                cmwxfreq = cmwxfreq.to_value(u.d**-1)
            if isinstance(cmwxsin, u.quantity.Quantity):
                cmwxsin = cmwxsin.to_value(cmu)
            if isinstance(cmwxcos, u.quantity.Quantity):
                cmwxcos = cmwxcos.to_value(cmu)
            log.trace(f"Adding CMWXSIN_{i} and CMWXCOS_{i} at frequency CMWXFREQ_{i}")
            self.add_param(
                prefixParameter(
                    name=f"CMWXFREQ_{i}",
                    description="Component frequency for Fourier representation of chromatic noise",
                    units="1/d",
                    value=cmwxfreq,
                    parameter_type="float",
                    tcb2tdb_scale_factor=u.Quantity(1),
                )
            )
            self.add_param(
                prefixParameter(
                    name=f"CMWXSIN_{i}",
                    description="Sine amplitude for Fourier representation of chromatic noise",
                    units=cmu,
                    value=cmwxsin,
                    parameter_type="float",
                    frozen=frozen,
                    tcb2tdb_scale_factor=DMconst,
                )
            )
            self.add_param(
                prefixParameter(
                    name=f"CMWXCOS_{i}",
                    description="Cosine amplitude for Fourier representation of chromatic noise",
                    units=cmu,
                    value=cmwxcos,
                    parameter_type="float",
                    frozen=frozen,
                    tcb2tdb_scale_factor=DMconst,
                )
            )
        self.setup()
        self.validate()
        return added_indices

    def remove_cmwavex_component(self, index):
        """
        Remove all CMWaveX components associated with a given index or list of indices

        Parameters
        ----------
        index : float, int, list, np.ndarray
            Number or list/array of numbers corresponding to CMWaveX indices to be removed from model.
        """

        if isinstance(index, (int, float, np.int64)):
            indices = [index]
        elif isinstance(index, (list, set, np.ndarray)):
            indices = index
        else:
            raise TypeError(
                f"index most be a float, int, set, list, or array - not {type(index)}"
            )
        for index in indices:
            index_rf = f"{int(index):04d}"
            for prefix in ["CMWXFREQ_", "CMWXSIN_", "CMWXCOS_"]:
                self.remove_param(prefix + index_rf)
        self.validate()

    def get_indices(self):
        """
        Returns an array of intergers corresponding to CMWaveX component parameters using CMWXFREQs

        Returns
        -------
        inds : np.ndarray
        Array of CMWaveX indices in model.
        """
        inds = [int(p.split("_")[-1]) for p in self.params if "CMWXFREQ_" in p]
        return np.array(inds)

    # Initialize setup
    def setup(self):
        super().setup()
        # Get CMWaveX mapping and register CMWXSIN and CMWXCOS derivatives
        for prefix_par in self.get_params_of_type("prefixParameter"):
            if prefix_par.startswith("CMWXSIN_"):
                self.register_deriv_funcs(self.d_delay_d_cmparam, prefix_par)
                self.register_cm_deriv_funcs(self.d_cm_d_CMWXSIN, prefix_par)
            if prefix_par.startswith("CMWXCOS_"):
                self.register_deriv_funcs(self.d_delay_d_cmparam, prefix_par)
                self.register_cm_deriv_funcs(self.d_cm_d_CMWXCOS, prefix_par)
            self.cmwavex_freqs = list(
                self.get_prefix_mapping_component("CMWXFREQ_").keys()
            )
            self.num_cmwavex_freqs = len(self.cmwavex_freqs)

    def validate(self):
        # Validate all the CMWaveX parameters
        super().validate()
        self.setup()
        CMWXFREQ_mapping = self.get_prefix_mapping_component("CMWXFREQ_")
        CMWXSIN_mapping = self.get_prefix_mapping_component("CMWXSIN_")
        CMWXCOS_mapping = self.get_prefix_mapping_component("CMWXCOS_")
        if CMWXFREQ_mapping.keys() != CMWXSIN_mapping.keys():
            raise ValueError(
                "CMWXFREQ_ parameters do not match CMWXSIN_ parameters."
                "Please check your prefixed parameters"
            )
        if CMWXFREQ_mapping.keys() != CMWXCOS_mapping.keys():
            raise ValueError(
                "CMWXFREQ_ parameters do not match CMWXCOS_ parameters."
                "Please check your prefixed parameters"
            )
        # if len(CMWXFREQ_mapping.keys()) != len(CMWXSIN_mapping.keys()):
        #     raise ValueError(
        #         "The number of CMWXFREQ_ parameters do not match the number of CMWXSIN_ parameters."
        #         "Please check your prefixed parameters"
        #     )
        # if len(CMWXFREQ_mapping.keys()) != len(CMWXCOS_mapping.keys()):
        #     raise ValueError(
        #         "The number of CMWXFREQ_ parameters do not match the number of CMWXCOS_ parameters."
        #         "Please check your prefixed parameters"
        #     )
        if CMWXSIN_mapping.keys() != CMWXCOS_mapping.keys():
            raise ValueError(
                "CMWXSIN_ parameters do not match CMWXCOS_ parameters."
                "Please check your prefixed parameters"
            )
        if len(CMWXSIN_mapping.keys()) != len(CMWXCOS_mapping.keys()):
            raise ValueError(
                "The number of CMWXSIN_ and CMWXCOS_ parameters do not match"
                "Please check your prefixed parameters"
            )
        wfreqs = np.zeros(len(CMWXFREQ_mapping))
        for j, index in enumerate(CMWXFREQ_mapping):
            if (getattr(self, f"CMWXFREQ_{index:04d}").value == 0) or (
                getattr(self, f"CMWXFREQ_{index:04d}").quantity is None
            ):
                raise ValueError(
                    f"CMWXFREQ_{index:04d} is zero or None. Please check your prefixed parameters"
                )
            if getattr(self, f"CMWXFREQ_{index:04d}").value < 0.0:
                warn(f"Frequency CMWXFREQ_{index:04d} is negative")
            wfreqs[j] = getattr(self, f"CMWXFREQ_{index:04d}").value
        wfreqs.sort()
        # if np.any(np.diff(wfreqs) <= (1.0 / (2.0 * 364.25))):
        #     warn("Frequency resolution is greater than 1/yr")
        if self.CMWXEPOCH.value is None and self._parent is not None:
            if self._parent.PEPOCH.value is None:
                raise MissingParameter(
                    "CMWXEPOCH or PEPOCH are required if CMWaveX is being used"
                )
            else:
                self.CMWXEPOCH.quantity = self._parent.PEPOCH.quantity

    def validate_toas(self, toas):
        return super().validate_toas(toas)

    def cmwavex_cm(self, toas):
        total_cm = np.zeros(toas.ntoas) * cmu
        cmwave_freqs = self.get_prefix_mapping_component("CMWXFREQ_")
        cmwave_sins = self.get_prefix_mapping_component("CMWXSIN_")
        cmwave_cos = self.get_prefix_mapping_component("CMWXCOS_")

        base_phase = toas.table["tdbld"].data * u.d - self.CMWXEPOCH.value * u.d
        for idx, param in cmwave_freqs.items():
            freq = getattr(self, param).quantity
            cmwxsin = getattr(self, cmwave_sins[idx]).quantity
            cmwxcos = getattr(self, cmwave_cos[idx]).quantity
            arg = 2.0 * np.pi * freq * base_phase
            total_cm += cmwxsin * np.sin(arg.value) + cmwxcos * np.cos(arg.value)
        return total_cm

    def cmwavex_delay(self, toas, acc_delay=None):
        return self.chromatic_type_delay(toas)

    def d_cm_d_CMWXSIN(self, toas, param, acc_delay=None):
        par = getattr(self, param)
        freq = getattr(self, f"CMWXFREQ_{int(par.index):04d}").quantity
        base_phase = toas.table["tdbld"].data * u.d - self.CMWXEPOCH.value * u.d
        arg = 2.0 * np.pi * freq * base_phase
        deriv = np.sin(arg.value)
        return deriv * cmu / par.units

    def d_cm_d_CMWXCOS(self, toas, param, acc_delay=None):
        par = getattr(self, param)
        freq = getattr(self, f"CMWXFREQ_{int(par.index):04d}").quantity
        base_phase = toas.table["tdbld"].data * u.d - self.CMWXEPOCH.value * u.d
        arg = 2.0 * np.pi * freq * base_phase
        deriv = np.cos(arg.value)
        return deriv * cmu / par.units
