"""Delays expressed as a sum of sinusoids."""
import astropy.units as u
import numpy as np
from loguru import logger as log
from warnings import warn

from pint.models.parameter import MJDParameter, prefixParameter
from pint.models.timing_model import DelayComponent, MissingParameter


class WaveX(DelayComponent):
    """
    Implementation of the wave model as a delay correction

    Delays are expressed as a sum of sinusoids.

    Used for decomposition of timing noise into a series of sine/cosine components with the amplitudes as fitted parameters.

    Parameters supported:

    .. paramtable::
        :class: pint.models.wavex.WaveX

    This is an extension of the L13 method described in Lentati et al., 2013 doi: 10.1103/PhysRevD.87.104021
    This model is similar to the TEMPO2 WAVE model parameters and users can convert a `TimingModel` with a Wave model
    to a WaveX model and produce the same results. The main differences are that the WaveX frequencies are explicitly stated,
    they do not necessarily need to be harmonics of some base frequency, the wave amplitudes are fittable parameters, and the
    sine and cosine amplutides are reported as separate `prefixParameter`s rather than as a single `pairParameter`.

    Analogous parameters in both models have the same units:
    WAVEEPOCH is the same as WXEPOCH
    WAVEOM and WXFREQ_000N have units of 1/d
    WAVEN and WXSIN_000N/WXCOS_000N have units of seconds

    The `pint.utils` functions `translate_wave_to_wavex()` and `translate_wavex_to_wave()` can be used to go back and forth between
    two model.

    WARNING: If the choice of WaveX frequencies in a `TimingModel` doesn't correspond to harmonics of some base
    freqeuncy, it will not be possible to convert it to a Wave model.

    To set up a WaveX model, users can use the `pint.utils` function `wavex_setup()` with either a list of frequencies or a choice
    of harmonics of a base frequency determined by 2 * pi /Timespan
    """

    register = True
    category = "wavex"

    def __init__(self):
        super().__init__()
        self.add_param(
            MJDParameter(
                name="WXEPOCH",
                description="Reference epoch for Fourier representation of red noise",
                time_scale="tdb",
            )
        )
        self.add_wavex_component(0.1, index=1, wxsin=0, wxcos=0, frozen=False)
        self.set_special_params(["WXFREQ_0001", "WXSIN_0001", "WXCOS_0001"])
        self.delay_funcs_component += [self.wavex_delay]

    def add_wavex_component(self, wxfreq, index=None, wxsin=0, wxcos=0, frozen=True):
        """
        Add WaveX component

        Parameters
        ----------

        wxfreq : float or astropy.quantity.Quantity
            Base frequency for WaveX component
        index : int, None
            Interger label for WaveX component. If None, will increment largest used index by 1.
        wxsin : float or astropy.quantity.Quantity
            Sine amplitude for WaveX component
        wxcos : float or astropy.quantity.Quantity
            Cosine amplitude for WaveX component
        frozen : iterable of bool or bool
            Indicates whether wavex will be fit

        Returns
        -------

        index : int
            Index that has been assigned to new WaveX component
        """

        #### If index is None, increment the current max WaveX index by 1. Increment using WXFREQ
        if index is None:
            dct = self.get_prefix_mapping_component("WXFREQ_")
            index = np.max(list(dct.keys())) + 1
        i = f"{int(index):04d}"

        if int(index) in self.get_prefix_mapping_component("WXFREQ_"):
            raise ValueError(
                f"Index '{index}' is already in use in this model. Please choose another"
            )

        if isinstance(wxsin, u.quantity.Quantity):
            wxsin = wxsin.to_value(u.s)
        if isinstance(wxcos, u.quantity.Quantity):
            wxcos = wxcos.to_value(u.s)
        if isinstance(wxfreq, u.quantity.Quantity):
            wxfreq = wxfreq.to_value(1 / u.d)
        self.add_param(
            prefixParameter(
                name=f"WXFREQ_{i}",
                description="Component frequency for Fourier representation of red noise",
                units="1/d",
                value=wxfreq,
                parameter_type="float",
            )
        )
        self.add_param(
            prefixParameter(
                name=f"WXSIN_{i}",
                description="Sine amplitudes for Fourier representation of red noise",
                units="s",
                value=wxsin,
                frozen=frozen,
                parameter_type="float",
            )
        )
        self.add_param(
            prefixParameter(
                name=f"WXCOS_{i}",
                description="Cosine amplitudes for Fourier representation of red noise",
                units="s",
                value=wxcos,
                frozen=frozen,
                parameter_type="float",
            )
        )
        self.setup()
        self.validate()
        return index

    def add_wavex_components(
        self, wxfreqs, indices=None, wxsins=0, wxcoses=0, frozens=True
    ):
        """
        Add WaveX components with specified base frequencies

        Parameters
        ----------

        wxfreqs : iterable of float or astropy.quantity.Quantity
            Base frequencies for WaveX components
        indices : iterable of int, None
            Interger labels for WaveX components. If None, will increment largest used index by 1.
        wxsins : iterable of float or astropy.quantity.Quantity
            Sine amplitudes for WaveX components
        wxcoses : iterable of float or astropy.quantity.Quantity
            Cosine amplitudes for WaveX components
        frozens : iterable of bool or bool
            Indicates whether sine and cosine amplitudes of wavex components will be fit

        Returns
        -------

        indices : list
            Indices that have been assigned to new WaveX components
        """

        if indices is None:
            indices = [None] * len(wxfreqs)
        wxsins = np.atleast_1d(wxsins)
        wxcoses = np.atleast_1d(wxcoses)
        if len(wxsins) == 1:
            wxsins = np.repeat(wxsins, len(wxfreqs))
        if len(wxcoses) == 1:
            wxcoses = np.repeat(wxcoses, len(wxfreqs))
        if len(wxsins) != len(wxfreqs):
            raise ValueError(
                f"Number of base frequencies {len(wxfreqs)} doesn't match number of sine ampltudes {len(wxsins)}"
            )
        if len(wxcoses) != len(wxfreqs):
            raise ValueError(
                f"Number of base frequencies {len(wxfreqs)} doesn't match number of cosine ampltudes {len(wxcoses)}"
            )
        frozens = np.atleast_1d(frozens)
        if len(frozens) == 1:
            frozens = np.repeat(frozens, len(wxfreqs))
        if len(frozens) != len(wxfreqs):
            raise ValueError(
                "Number of base frequencies must match number of frozen values"
            )
        #### If indices is None, increment the current max WaveX index by 1. Increment using WXFREQ
        dct = self.get_prefix_mapping_component("WXFREQ_")
        last_index = np.max(list(dct.keys()))
        added_indices = []
        for wxfreq, index, wxsin, wxcos, frozen in zip(
            wxfreqs, indices, wxsins, wxcoses, frozens
        ):
            if index is None:
                index = last_index + 1
                last_index += 1
            elif index in list(dct.keys()):
                raise ValueError(
                    f"Attempting to insert WXFREQ_{index:04d} but it already exists"
                )
            added_indices.append(index)
            i = f"{int(index):04d}"

            if int(index) in dct:
                raise ValueError(
                    f"Index '{index}' is already in use in this model. Please choose another"
                )
            if isinstance(wxfreq, u.quantity.Quantity):
                wxfreq = wxfreq.to_value(u.d**-1)
            if isinstance(wxsin, u.quantity.Quantity):
                wxsin = wxsin.to_value(u.s)
            if isinstance(wxcos, u.quantity.Quantity):
                wxcos = wxcos.to_value(u.s)
            log.trace(f"Adding WXSIN_{i} and WXCOS_{i} at frequency WXFREQ_{i}")
            self.add_param(
                prefixParameter(
                    name=f"WXFREQ_{i}",
                    description="Component frequency for Fourier representation of red noise",
                    units="1/d",
                    value=wxfreq,
                    parameter_type="float",
                )
            )
            self.add_param(
                prefixParameter(
                    name=f"WXSIN_{i}",
                    description="Sine amplitude for Fourier representation of red noise",
                    units="s",
                    value=wxsin,
                    parameter_type="float",
                    frozen=frozen,
                )
            )
            self.add_param(
                prefixParameter(
                    name=f"WXCOS_{i}",
                    description="Cosine amplitude for Fourier representation of red noise",
                    units="s",
                    value=wxcos,
                    parameter_type="float",
                    frozen=frozen,
                )
            )
        self.setup()
        self.validate()
        return added_indices

    def remove_wavex_component(self, index):
        """
        Remove all WaveX components associated with a given index or list of indices

        Parameters
        ----------
        index : float, int, list, np.ndarray
            Number or list/array of numbers corresponding to WaveX indices to be removed from model.
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
            for prefix in ["WXFREQ_", "WXSIN_", "WXCOS_"]:
                self.remove_param(prefix + index_rf)
        self.validate()

    def get_indices(self):
        """
        Returns an array of intergers corresponding to WaveX component parameters using WXFREQs

        Returns
        -------
        inds : np.ndarray
        Array of WaveX indices in model.
        """
        inds = [int(p.split("_")[-1]) for p in self.params if "WXFREQ_" in p]
        return np.array(inds)

    # Initialize setup
    def setup(self):
        super().setup()
        # Get WaveX mapping and register WXSIN and WXCOS derivatives
        for prefix_par in self.get_params_of_type("prefixParameter"):
            if prefix_par.startswith("WXSIN_"):
                self.register_deriv_funcs(self.d_wavex_delay_d_WXSIN, prefix_par)
            if prefix_par.startswith("WXCOS_"):
                self.register_deriv_funcs(self.d_wavex_delay_d_WXCOS, prefix_par)
            self.wave_freqs = list(self.get_prefix_mapping_component("WXFREQ_").keys())
            self.num_wave_freqs = len(self.wave_freqs)

    def validate(self):
        # Validate all the WaveX parameters
        super().validate()
        self.setup()
        WXFREQ_mapping = self.get_prefix_mapping_component("WXFREQ_")
        WXSIN_mapping = self.get_prefix_mapping_component("WXSIN_")
        WXCOS_mapping = self.get_prefix_mapping_component("WXCOS_")
        if WXFREQ_mapping.keys() != WXSIN_mapping.keys():
            raise ValueError(
                "WXFREQ_ parameters do not match WXSIN_ parameters."
                "Please check your prefixed parameters"
            )
        if WXFREQ_mapping.keys() != WXCOS_mapping.keys():
            raise ValueError(
                "WXFREQ_ parameters do not match WXCOS_ parameters."
                "Please check your prefixed parameters"
            )
        # if len(WXFREQ_mapping.keys()) != len(WXSIN_mapping.keys()):
        #     raise ValueError(
        #         "The number of WXFREQ_ parameters do not match the number of WXSIN_ parameters."
        #         "Please check your prefixed parameters"
        #     )
        # if len(WXFREQ_mapping.keys()) != len(WXCOS_mapping.keys()):
        #     raise ValueError(
        #         "The number of WXFREQ_ parameters do not match the number of WXCOS_ parameters."
        #         "Please check your prefixed parameters"
        #     )
        if WXSIN_mapping.keys() != WXCOS_mapping.keys():
            raise ValueError(
                "WXSIN_ parameters do not match WXCOS_ parameters."
                "Please check your prefixed parameters"
            )
        if len(WXSIN_mapping.keys()) != len(WXCOS_mapping.keys()):
            raise ValueError(
                "The number of WXSIN_ and WXCOS_ parameters do not match"
                "Please check your prefixed parameters"
            )
        wfreqs = np.zeros(len(WXFREQ_mapping))
        for j, index in enumerate(WXFREQ_mapping):
            if (getattr(self, f"WXFREQ_{index:04d}").value == 0) or (
                getattr(self, f"WXFREQ_{index:04d}").quantity is None
            ):
                raise ValueError(
                    f"WXFREQ_{index:04d} is zero or None. Please check your prefixed parameters"
                )
            if getattr(self, f"WXFREQ_{index:04d}").value < 0.0:
                warn(f"Frequency WXFREQ_{index:04d} is negative")
            wfreqs[j] = getattr(self, f"WXFREQ_{index:04d}").value
        wfreqs.sort()
        # if np.any(np.diff(wfreqs) <= (1.0 / (2.0 * 364.25))):
        #     warn("Frequency resolution is greater than 1/yr")
        if self.WXEPOCH.value is None and self._parent is not None:
            if self._parent.PEPOCH.value is None:
                raise MissingParameter(
                    "WXEPOCH or PEPOCH are required if WaveX is being used"
                )
            else:
                self.WXEPOCH.quantity = self._parent.PEPOCH.quantity

    def validate_toas(self, toas):
        return super().validate_toas(toas)

    def wavex_delay(self, toas, delays):
        total_delay = np.zeros(toas.ntoas) * u.s
        wave_freqs = self.get_prefix_mapping_component("WXFREQ_")
        wave_sins = self.get_prefix_mapping_component("WXSIN_")
        wave_cos = self.get_prefix_mapping_component("WXCOS_")

        base_phase = (
            toas.table["tdbld"].data * u.d - self.WXEPOCH.value * u.d - delays.to(u.d)
        )
        for idx, param in wave_freqs.items():
            freq = getattr(self, param).quantity
            wxsin = getattr(self, wave_sins[idx]).quantity
            wxcos = getattr(self, wave_cos[idx]).quantity
            arg = 2.0 * np.pi * freq * base_phase
            total_delay += wxsin * np.sin(arg.value) + wxcos * np.cos(arg.value)
        return total_delay

    def d_wavex_delay_d_WXSIN(self, toas, param, delays, acc_delay=None):
        par = getattr(self, param)
        freq = getattr(self, f"WXFREQ_{int(par.index):04d}").quantity
        base_phase = toas.table["tdbld"].data * u.d - self.WXEPOCH.value * u.d
        arg = 2.0 * np.pi * freq * base_phase
        deriv = np.sin(arg.value)
        return deriv * u.s / par.units

    def d_wavex_delay_d_WXCOS(self, toas, param, delays, acc_delay=None):
        par = getattr(self, param)
        freq = getattr(self, f"WXFREQ_{int(par.index):04d}").quantity
        base_phase = toas.table["tdbld"].data * u.d - self.WXEPOCH.value * u.d
        arg = 2.0 * np.pi * freq * base_phase
        deriv = np.cos(arg.value)
        return deriv * u.s / par.units
