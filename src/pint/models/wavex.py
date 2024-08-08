"""Delays expressed as a sum of sinusoids."""

from copy import deepcopy
from typing import Iterable, List, Optional, Union
import warnings
import astropy.units as u
import numpy as np
from loguru import logger as log
from warnings import warn

from pint.models.parameter import MJDParameter, prefixParameter
from pint.models.timing_model import DelayComponent, MissingParameter, TimingModel


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

    To set up a WaveX model, users can use the function `wavex_setup()` with either a list of frequencies or a choice
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
                tcb2tdb_scale_factor=u.Quantity(1),
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
                tcb2tdb_scale_factor=u.Quantity(1),
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
                tcb2tdb_scale_factor=u.Quantity(1),
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
                tcb2tdb_scale_factor=u.Quantity(1),
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
                    tcb2tdb_scale_factor=u.Quantity(1),
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
                    tcb2tdb_scale_factor=u.Quantity(1),
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
                    tcb2tdb_scale_factor=u.Quantity(1),
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


def wavex_setup(
    model: TimingModel,
    T_span: Union[float, u.Quantity],
    freqs: Optional[Iterable[Union[float, u.Quantity]]] = None,
    n_freqs: Optional[int] = None,
    freeze_params: bool = False,
) -> List[int]:
    """
    Set-up a WaveX model based on either an array of user-provided frequencies or the wave number
    frequency calculation. Sine and Cosine amplitudes are initially set to zero

    User specifies T_span and either freqs or n_freqs. This function assumes that the timing model does not already
    have any WaveX components. See add_wavex_component() or add_wavex_components() to add WaveX components
    to an existing WaveX model.

    Parameters
    ----------

    model : pint.models.timing_model.TimingModel
    T_span : float, astropy.quantity.Quantity
        Time span used to calculate nyquist frequency when using freqs
        Time span used to calculate WaveX frequencies when using n_freqs
        Usually to be set as the length of the timing baseline the model is being used for
    freqs : iterable of float or astropy.quantity.Quantity, None
        User inputed base frequencies
    n_freqs : int, None
        Number of wave frequencies to calculate using the equation: freq_n = 2 * pi * n / T_span
        Where n is the wave number, and T_span is the total time span of the toas in the fitter object
    freeze_params : bool, optional
        Whether the new parameters should be frozen

    Returns
    -------

    indices : list
            Indices that have been assigned to new WaveX components
    """
    from pint.models.wavex import WaveX

    if (freqs is None) and (n_freqs is None):
        raise ValueError(
            "WaveX component base frequencies are not specified. "
            "Please input either freqs or n_freqs"
        )

    if (freqs is not None) and (n_freqs is not None):
        raise ValueError(
            "Both freqs and n_freqs are specified. Only one or the other should be used"
        )

    if n_freqs is not None and n_freqs <= 0:
        raise ValueError("Must use a non-zero number of wave frequencies")

    model.add_component(WaveX())
    if isinstance(T_span, u.quantity.Quantity):
        T_span.to(u.d)
    else:
        T_span *= u.d

    nyqist_freq = 1.0 / (2.0 * T_span)
    if freqs is not None:
        if isinstance(freqs, u.quantity.Quantity):
            freqs.to(u.d**-1)
        else:
            freqs *= u.d**-1
        if len(freqs) == 1:
            model.WXFREQ_0001.quantity = freqs
        else:
            freqs = np.array(freqs)
            freqs.sort()
            if min(np.diff(freqs)) < nyqist_freq:
                warnings.warn(
                    "Wave frequency spacing is finer than frequency resolution of data"
                )
            model.WXFREQ_0001.quantity = freqs[0]
            model.components["WaveX"].add_wavex_components(freqs[1:])

    if n_freqs is not None:
        if n_freqs == 1:
            wave_freq = 1 / T_span
            model.WXFREQ_0001.quantity = wave_freq
        else:
            wave_numbers = np.arange(1, n_freqs + 1)
            wave_freqs = wave_numbers / T_span
            model.WXFREQ_0001.quantity = wave_freqs[0]
            model.components["WaveX"].add_wavex_components(wave_freqs[1:])

    for p in model.params:
        if p.startswith("WXSIN") or p.startswith("WXCOS"):
            model[p].frozen = freeze_params

    return model.components["WaveX"].get_indices()


def get_wavex_freqs(
    model: TimingModel,
    index: Optional[Union[float, int, List, np.ndarray]] = None,
    quantity: bool = False,
) -> List[Union[float, u.Quantity]]:
    """
    Return the WaveX frequencies for a timing model.

    If index is specified, returns the frequencies corresponding to the user-provided indices.
    If index isn't specified, returns all WaveX frequencies in timing model

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
        Timing model from which to return WaveX frequencies
    index : float, int, list, np.ndarray, None
        Number or list/array of numbers corresponding to WaveX frequencies to return
    quantity : bool
        If set to True, returns a list of astropy.quanitity.Quantity rather than a list of prefixParameters

    Returns
    -------
    List of WXFREQ_ parameters
    """
    if index is None:
        freqs = model.components["WaveX"].get_prefix_mapping_component("WXFREQ_")
        if len(freqs) == 1:
            values = getattr(model.components["WaveX"], freqs.values())
        else:
            values = [
                getattr(model.components["WaveX"], param) for param in freqs.values()
            ]
    elif isinstance(index, (int, float, np.int64)):
        idx_rf = f"{int(index):04d}"
        values = getattr(model.components["WaveX"], f"WXFREQ_{idx_rf}")
    elif isinstance(index, (list, set, np.ndarray)):
        idx_rf = [f"{int(idx):04d}" for idx in index]
        values = [getattr(model.components["WaveX"], f"WXFREQ_{ind}") for ind in idx_rf]
    else:
        raise TypeError(
            f"index most be a float, int, set, list, array, or None - not {type(index)}"
        )
    if quantity:
        if len(values) == 1:
            values = [values[0].quantity]
        else:
            values = [v.quantity for v in values]
    return values


def get_wavex_amps(
    model: TimingModel,
    index: Optional[Union[float, int, List, np.ndarray]] = None,
    quantity: bool = False,
) -> List[Union[float, u.Quantity]]:
    """
    Return the WaveX amplitudes for a timing model.

    If index is specified, returns the sine/cosine amplitudes corresponding to the user-provided indices.
    If index isn't specified, returns all WaveX sine/cosine amplitudes in timing model

    Parameters
    ----------
    model : pint.models.timing_model.TimingModel
        Timing model from which to return WaveX frequencies
    index : float, int, list, np.ndarray, None
        Number or list/array of numbers corresponding to WaveX amplitudes to return
    quantity : bool
        If set to True, returns a list of tuples of astropy.quanitity.Quantity rather than a list of prefixParameters tuples

    Returns
    -------
    List of WXSIN_ and WXCOS_ parameters
    """
    if index is None:
        indices = (
            model.components["WaveX"].get_prefix_mapping_component("WXSIN_").keys()
        )
        if len(indices) == 1:
            values = getattr(
                model.components["WaveX"], f"WXSIN_{int(indices):04d}"
            ), getattr(model.components["WaveX"], f"WXCOS_{int(indices):04d}")
        else:
            values = [
                (
                    getattr(model.components["WaveX"], f"WXSIN_{int(idx):04d}"),
                    getattr(model.components["WaveX"], f"WXCOS_{int(idx):04d}"),
                )
                for idx in indices
            ]
    elif isinstance(index, (int, float, np.int64)):
        idx_rf = f"{int(index):04d}"
        values = getattr(model.components["WaveX"], f"WXSIN_{idx_rf}"), getattr(
            model.components["WaveX"], f"WXCOS_{idx_rf}"
        )
    elif isinstance(index, (list, set, np.ndarray)):
        idx_rf = [f"{int(idx):04d}" for idx in index]
        values = [
            (
                getattr(model.components["WaveX"], f"WXSIN_{ind}"),
                getattr(model.components["WaveX"], f"WXCOS_{ind}"),
            )
            for ind in idx_rf
        ]
    else:
        raise TypeError(
            f"index most be a float, int, set, list, array, or None - not {type(index)}"
        )
    if quantity:
        if isinstance(values, tuple):
            values = tuple(v.quantity for v in values)
        if isinstance(values, list):
            values = [(v[0].quantity, v[1].quantity) for v in values]
    return values


def translate_wave_to_wavex(
    model: TimingModel,
) -> TimingModel:
    """
    Go from a Wave model to a WaveX model

    WaveX frequencies get calculated based on the Wave model WAVEOM parameter and the number of WAVE parameters.
        WXFREQ_000k = [WAVEOM * (k+1)] / [2 * pi]

    WaveX amplitudes are taken from the WAVE pair parameters

    Paramters
    ---------
    model : pint.models.timing_model.TimingModel
        TimingModel containing a Wave model to be converted to a WaveX model

    Returns
    -------
    pint.models.timing_model.TimingModel
        New timing model with converted WaveX model included
    """
    from pint.models.wavex import WaveX

    new_model = deepcopy(model)
    wave_names = [
        f"WAVE{ii}" for ii in range(1, model.components["Wave"].num_wave_terms + 1)
    ]
    wave_terms = [getattr(model.components["Wave"], name) for name in wave_names]
    wave_om = model.components["Wave"].WAVE_OM.quantity
    wave_epoch = model.components["Wave"].WAVEEPOCH.quantity
    new_model.remove_component("Wave")
    new_model.add_component(WaveX())
    new_model.WXEPOCH.value = wave_epoch.value
    for k, wave_term in enumerate(wave_terms):
        wave_sin_amp, wave_cos_amp = wave_term.quantity
        wavex_freq = _translate_wave_freqs(wave_om, k)
        if k == 0:
            new_model.WXFREQ_0001.value = wavex_freq.value
            new_model.WXSIN_0001.value = -wave_sin_amp.value
            new_model.WXCOS_0001.value = -wave_cos_amp.value
        else:
            new_model.components["WaveX"].add_wavex_component(
                wavex_freq, wxsin=-wave_sin_amp, wxcos=-wave_cos_amp
            )
    return new_model


def translate_wavex_to_wave(
    model: TimingModel,
) -> TimingModel:
    """
    Go from a WaveX timing model to a Wave timing model.
    WARNING: Not every WaveX model can be appropriately translated into a Wave model. This is dependent on the user's choice of frequencies in the WaveX model.
    In order for a WaveX model to be able to be converted into a Wave model, every WaveX frequency must produce the same value of WAVEOM in the calculation:

    WAVEOM = [2 * pi * WXFREQ_000k] / (k + 1)
    Paramters
    ---------
    model : pint.models.timing_model.TimingModel
        TimingModel containing a WaveX model to be converted to a Wave model

    Returns
    -------
    pint.models.timing_model.TimingModel
        New timing model with converted Wave model included
    """
    from pint.models.wave import Wave

    new_model = deepcopy(model)
    indices = model.components["WaveX"].get_indices()
    wxfreqs = get_wavex_freqs(model, indices, quantity=True)
    wave_om = _translate_wavex_freqs(wxfreqs, (indices - 1))
    if wave_om == False:
        raise ValueError(
            "This WaveX model cannot be properly translated into a Wave model due to the WaveX frequencies not producing a consistent WAVEOM value"
        )
    wave_amps = get_wavex_amps(model, index=indices, quantity=True)
    new_model.remove_component("WaveX")
    new_model.add_component(Wave())
    new_model.WAVEEPOCH.quantity = model.WXEPOCH.quantity
    new_model.WAVE_OM.quantity = wave_om
    new_model.WAVE1.quantity = tuple(w * -1.0 for w in wave_amps[0])
    if len(indices) > 1:
        for i in range(1, len(indices)):
            print(wave_amps[i])
            wave_amps[i] = tuple(w * -1.0 for w in wave_amps[i])
            new_model.components["Wave"].add_wave_component(
                wave_amps[i], index=indices[i]
            )
    return new_model


def _translate_wavex_freqs(wxfreq: Union[float, u.Quantity], k: int) -> u.Quantity:
    """
    Use WaveX model WXFREQ_ parameters and wave number k to calculate the Wave model WAVEOM frequency parameter.

    Parameters
    ----------
    wxfreq : float or astropy.quantity.Quantity
        WaveX frequency from which the WAVEOM parameter will be calculated
        If float is given default units of 1/d assigned
    k : int
        wave number to use to calculate Wave WAVEOM parameter

    Returns
    -------
    astropy.units.Quantity
        WAVEOM quantity in units 1/d that can be used in Wave model
    """
    if isinstance(wxfreq, u.quantity.Quantity):
        wxfreq.to(u.d**-1)
    else:
        wxfreq *= u.d**-1
    if len(wxfreq) == 1:
        return (2.0 * np.pi * wxfreq) / (k + 1.0)
    wave_om = [((2.0 * np.pi * wxfreq[i]) / (k[i] + 1.0)) for i in range(len(wxfreq))]
    return (
        sum(wave_om) / len(wave_om)
        if np.allclose(wave_om, wave_om[0], atol=1e-3)
        else False
    )


def _translate_wave_freqs(om: Union[float, u.Quantity], k: int) -> u.Quantity:
    """
    Use Wave model WAVEOM parameter to calculate a WaveX WXFREQ_ frequency parameter for wave number k

    Parameters
    ----------
    om : float or astropy.quantity.Quantity
        Base frequency of Wave model solution - parameter WAVEOM
        If float is given default units of 1/d assigned
    k : int
        wave number to use to calculate WaveX WXFREQ_ frequency parameter

    Returns
    -------
    astropy.units.Quantity
        WXFREQ_ quantity in units 1/d that can be used in WaveX model
    """
    if isinstance(om, u.quantity.Quantity):
        om.to(u.d**-1)
    else:
        om *= u.d**-1
    return (om * (k + 1)) / (2.0 * np.pi)
