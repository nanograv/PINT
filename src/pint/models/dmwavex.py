"""DM variations expressed as a sum of sinusoids."""
import astropy.units as u
import numpy as np
from loguru import logger as log
from warnings import warn

from pint.models.parameter import MJDParameter, prefixParameter
from pint.models.timing_model import MissingParameter
from pint.models.dispersion_model import Dispersion
from pint import dmu


class DMWaveX(Dispersion):
    """
    Fourier representation of DM variations.

    Used for decomposition of DM noise into a series of sine/cosine components with the amplitudes as fitted parameters.

    Parameters supported:

    .. paramtable::
        :class: pint.models.dmwavex.DMWaveX

    To set up a DMWaveX model, users can use the `pint.utils` function `dmwavex_setup()` with either a list of frequencies or a choice
    of harmonics of a base frequency determined by 2 * pi /Timespan
    """

    register = True
    category = "dmwavex"

    def __init__(self):
        super().__init__()
        self.add_param(
            MJDParameter(
                name="DMWXEPOCH",
                description="Reference epoch for Fourier representation of DM noise",
                time_scale="tdb",
            )
        )
        self.add_dmwavex_component(0.1, index=1, dmwxsin=0, dmwxcos=0, frozen=False)
        self.set_special_params(["DMWXFREQ_0001", "DMWXSIN_0001", "DMWXCOS_0001"])
        self.dm_value_funcs += [self.dmwavex_dm]
        self.delay_funcs_component += [self.dmwavex_delay]

    def add_dmwavex_component(
        self, dmwxfreq, index=None, dmwxsin=0, dmwxcos=0, frozen=True
    ):
        """
        Add DMWaveX component

        Parameters
        ----------

        dmwxfreq : float or astropy.quantity.Quantity
            Base frequency for DMWaveX component
        index : int, None
            Interger label for DMWaveX component. If None, will increment largest used index by 1.
        dmwxsin : float or astropy.quantity.Quantity
            Sine amplitude for DMWaveX component
        dmwxcos : float or astropy.quantity.Quantity
            Cosine amplitude for DMWaveX component
        frozen : iterable of bool or bool
            Indicates whether DMWaveX parameters will be fit

        Returns
        -------

        index : int
            Index that has been assigned to new DMWaveX component
        """

        #### If index is None, increment the current max DMWaveX index by 1. Increment using DMWXFREQ
        if index is None:
            dct = self.get_prefix_mapping_component("DMWXFREQ_")
            index = np.max(list(dct.keys())) + 1
        i = f"{int(index):04d}"

        if int(index) in self.get_prefix_mapping_component("DMWXFREQ_"):
            raise ValueError(
                f"Index '{index}' is already in use in this model. Please choose another"
            )

        if isinstance(dmwxsin, u.quantity.Quantity):
            dmwxsin = dmwxsin.to_value(dmu)
        if isinstance(dmwxcos, u.quantity.Quantity):
            dmwxcos = dmwxcos.to_value(dmu)
        if isinstance(dmwxfreq, u.quantity.Quantity):
            dmwxfreq = dmwxfreq.to_value(1 / u.d)
        self.add_param(
            prefixParameter(
                name=f"DMWXFREQ_{i}",
                description="Component frequency for Fourier representation of DM noise",
                units="1/d",
                value=dmwxfreq,
                parameter_type="float",
            )
        )
        self.add_param(
            prefixParameter(
                name=f"DMWXSIN_{i}",
                description="Sine amplitudes for Fourier representation of DM noise",
                units=dmu,
                value=dmwxsin,
                frozen=frozen,
                parameter_type="float",
            )
        )
        self.add_param(
            prefixParameter(
                name=f"DMWXCOS_{i}",
                description="Cosine amplitudes for Fourier representation of DM noise",
                units=dmu,
                value=dmwxcos,
                frozen=frozen,
                parameter_type="float",
            )
        )
        self.setup()
        self.validate()
        return index

    def add_dmwavex_components(
        self, dmwxfreqs, indices=None, dmwxsins=0, dmwxcoses=0, frozens=True
    ):
        """
        Add DMWaveX components with specified base frequencies

        Parameters
        ----------

        dmwxfreqs : iterable of float or astropy.quantity.Quantity
            Base frequencies for DMWaveX components
        indices : iterable of int, None
            Interger labels for DMWaveX components. If None, will increment largest used index by 1.
        dmwxsins : iterable of float or astropy.quantity.Quantity
            Sine amplitudes for DMWaveX components
        dmwxcoses : iterable of float or astropy.quantity.Quantity
            Cosine amplitudes for DMWaveX components
        frozens : iterable of bool or bool
            Indicates whether sine and cosine amplitudes of DMwavex components will be fit

        Returns
        -------

        indices : list
            Indices that have been assigned to new DMWaveX components
        """

        if indices is None:
            indices = [None] * len(dmwxfreqs)
        dmwxsins = np.atleast_1d(dmwxsins)
        dmwxcoses = np.atleast_1d(dmwxcoses)
        if len(dmwxsins) == 1:
            dmwxsins = np.repeat(dmwxsins, len(dmwxfreqs))
        if len(dmwxcoses) == 1:
            dmwxcoses = np.repeat(dmwxcoses, len(dmwxfreqs))
        if len(dmwxsins) != len(dmwxfreqs):
            raise ValueError(
                f"Number of base frequencies {len(dmwxfreqs)} doesn't match number of sine ampltudes {len(dmwxsins)}"
            )
        if len(dmwxcoses) != len(dmwxfreqs):
            raise ValueError(
                f"Number of base frequencies {len(dmwxfreqs)} doesn't match number of cosine ampltudes {len(dmwxcoses)}"
            )
        frozens = np.atleast_1d(frozens)
        if len(frozens) == 1:
            frozens = np.repeat(frozens, len(dmwxfreqs))
        if len(frozens) != len(dmwxfreqs):
            raise ValueError(
                "Number of base frequencies must match number of frozen values"
            )
        #### If indices is None, increment the current max DMWaveX index by 1. Increment using DMWXFREQ
        dct = self.get_prefix_mapping_component("DMWXFREQ_")
        last_index = np.max(list(dct.keys()))
        added_indices = []
        for dmwxfreq, index, dmwxsin, dmwxcos, frozen in zip(
            dmwxfreqs, indices, dmwxsins, dmwxcoses, frozens
        ):
            if index is None:
                index = last_index + 1
                last_index += 1
            elif index in list(dct.keys()):
                raise ValueError(
                    f"Attempting to insert DMWXFREQ_{index:04d} but it already exists"
                )
            added_indices.append(index)
            i = f"{int(index):04d}"

            if int(index) in dct:
                raise ValueError(
                    f"Index '{index}' is already in use in this model. Please choose another"
                )
            if isinstance(dmwxfreq, u.quantity.Quantity):
                dmwxfreq = dmwxfreq.to_value(u.d**-1)
            if isinstance(dmwxsin, u.quantity.Quantity):
                dmwxsin = dmwxsin.to_value(dmu)
            if isinstance(dmwxcos, u.quantity.Quantity):
                dmwxcos = dmwxcos.to_value(dmu)
            log.trace(f"Adding DMWXSIN_{i} and DMWXCOS_{i} at frequency DMWXFREQ_{i}")
            self.add_param(
                prefixParameter(
                    name=f"DMWXFREQ_{i}",
                    description="Component frequency for Fourier representation of DM noise",
                    units="1/d",
                    value=dmwxfreq,
                    parameter_type="float",
                )
            )
            self.add_param(
                prefixParameter(
                    name=f"DMWXSIN_{i}",
                    description="Sine amplitude for Fourier representation of DM noise",
                    units=dmu,
                    value=dmwxsin,
                    parameter_type="float",
                    frozen=frozen,
                )
            )
            self.add_param(
                prefixParameter(
                    name=f"DMWXCOS_{i}",
                    description="Cosine amplitude for Fourier representation of DM noise",
                    units=dmu,
                    value=dmwxcos,
                    parameter_type="float",
                    frozen=frozen,
                )
            )
        self.setup()
        self.validate()
        return added_indices

    def remove_dmwavex_component(self, index):
        """
        Remove all DMWaveX components associated with a given index or list of indices

        Parameters
        ----------
        index : float, int, list, np.ndarray
            Number or list/array of numbers corresponding to DMWaveX indices to be removed from model.
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
            for prefix in ["DMWXFREQ_", "DMWXSIN_", "DMWXCOS_"]:
                self.remove_param(prefix + index_rf)
        self.validate()

    def get_indices(self):
        """
        Returns an array of intergers corresponding to DMWaveX component parameters using DMWXFREQs

        Returns
        -------
        inds : np.ndarray
        Array of DMWaveX indices in model.
        """
        inds = [int(p.split("_")[-1]) for p in self.params if "WXFREQ_" in p]
        return np.array(inds)

    # Initialize setup
    def setup(self):
        super().setup()
        # Get DMWaveX mapping and register DMWXSIN and DMWXCOS derivatives
        for prefix_par in self.get_params_of_type("prefixParameter"):
            if prefix_par.startswith("DMWXSIN_"):
                self.register_deriv_funcs(self.d_delay_d_dmparam, prefix_par)
                self.register_dm_deriv_funcs(self.d_dm_d_DMWXSIN, prefix_par)
            if prefix_par.startswith("DMWXCOS_"):
                self.register_deriv_funcs(self.d_delay_d_dmparam, prefix_par)
                self.register_dm_deriv_funcs(self.d_dm_d_DMWXCOS, prefix_par)
            self.dmwavex_freqs = list(
                self.get_prefix_mapping_component("DMWXFREQ_").keys()
            )
            self.num_dmwavex_freqs = len(self.dmwavex_freqs)

    def validate(self):
        # Validate all the DMWaveX parameters
        super().validate()
        self.setup()
        DMWXFREQ_mapping = self.get_prefix_mapping_component("DMWXFREQ_")
        DMWXSIN_mapping = self.get_prefix_mapping_component("DMWXSIN_")
        DMWXCOS_mapping = self.get_prefix_mapping_component("DMWXCOS_")
        if DMWXFREQ_mapping.keys() != DMWXSIN_mapping.keys():
            raise ValueError(
                "WXFREQ_ parameters do not match DMWXSIN_ parameters."
                "Please check your prefixed parameters"
            )
        if DMWXFREQ_mapping.keys() != DMWXCOS_mapping.keys():
            raise ValueError(
                "DMWXFREQ_ parameters do not match DMWXCOS_ parameters."
                "Please check your prefixed parameters"
            )
        # if len(DMWXFREQ_mapping.keys()) != len(DMWXSIN_mapping.keys()):
        #     raise ValueError(
        #         "The number of DMWXFREQ_ parameters do not match the number of DMWXSIN_ parameters."
        #         "Please check your prefixed parameters"
        #     )
        # if len(DMWXFREQ_mapping.keys()) != len(DMWXCOS_mapping.keys()):
        #     raise ValueError(
        #         "The number of DMWXFREQ_ parameters do not match the number of DMWXCOS_ parameters."
        #         "Please check your prefixed parameters"
        #     )
        if DMWXSIN_mapping.keys() != DMWXCOS_mapping.keys():
            raise ValueError(
                "DMWXSIN_ parameters do not match DMWXCOS_ parameters."
                "Please check your prefixed parameters"
            )
        if len(DMWXSIN_mapping.keys()) != len(DMWXCOS_mapping.keys()):
            raise ValueError(
                "The number of DMWXSIN_ and DMWXCOS_ parameters do not match"
                "Please check your prefixed parameters"
            )
        wfreqs = np.zeros(len(DMWXFREQ_mapping))
        for j, index in enumerate(DMWXFREQ_mapping):
            if (getattr(self, f"DMWXFREQ_{index:04d}").value == 0) or (
                getattr(self, f"DMWXFREQ_{index:04d}").quantity is None
            ):
                raise ValueError(
                    f"DMWXFREQ_{index:04d} is zero or None. Please check your prefixed parameters"
                )
            if getattr(self, f"DMWXFREQ_{index:04d}").value < 0.0:
                warn(f"Frequency DMWXFREQ_{index:04d} is negative")
            wfreqs[j] = getattr(self, f"DMWXFREQ_{index:04d}").value
        wfreqs.sort()
        # if np.any(np.diff(wfreqs) <= (1.0 / (2.0 * 364.25))):
        #     warn("Frequency resolution is greater than 1/yr")
        if self.DMWXEPOCH.value is None and self._parent is not None:
            if self._parent.PEPOCH.value is None:
                raise MissingParameter(
                    "DMWXEPOCH or PEPOCH are required if DMWaveX is being used"
                )
            else:
                self.DMWXEPOCH.quantity = self._parent.PEPOCH.quantity

    def validate_toas(self, toas):
        return super().validate_toas(toas)

    def dmwavex_dm(self, toas):
        total_dm = np.zeros(toas.ntoas) * dmu
        dmwave_freqs = self.get_prefix_mapping_component("DMWXFREQ_")
        dmwave_sins = self.get_prefix_mapping_component("DMWXSIN_")
        dmwave_cos = self.get_prefix_mapping_component("DMWXCOS_")

        base_phase = toas.table["tdbld"].data * u.d - self.DMWXEPOCH.value * u.d
        for idx, param in dmwave_freqs.items():
            freq = getattr(self, param).quantity
            dmwxsin = getattr(self, dmwave_sins[idx]).quantity
            dmwxcos = getattr(self, dmwave_cos[idx]).quantity
            arg = 2.0 * np.pi * freq * base_phase
            total_dm += dmwxsin * np.sin(arg.value) + dmwxcos * np.cos(arg.value)
        return total_dm

    def dmwavex_delay(self, toas, acc_delay=None):
        return self.dispersion_type_delay(toas)

    def d_dm_d_DMWXSIN(self, toas, param, acc_delay=None):
        par = getattr(self, param)
        freq = getattr(self, f"DMWXFREQ_{int(par.index):04d}").quantity
        base_phase = toas.table["tdbld"].data * u.d - self.DMWXEPOCH.value * u.d
        arg = 2.0 * np.pi * freq * base_phase
        deriv = np.sin(arg.value)
        return deriv * dmu / par.units

    def d_dm_d_DMWXCOS(self, toas, param, acc_delay=None):
        par = getattr(self, param)
        freq = getattr(self, f"DMWXFREQ_{int(par.index):04d}").quantity
        base_phase = toas.table["tdbld"].data * u.d - self.DMWXEPOCH.value * u.d
        arg = 2.0 * np.pi * freq * base_phase
        deriv = np.cos(arg.value)
        return deriv * dmu / par.units
