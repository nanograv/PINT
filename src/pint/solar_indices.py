"""Solar activity index downloading, caching and parsing.

Provides the external time series that solar-wind models need as a proxy:
the 10.7 cm radio flux (F10.7), the sunspot number, and in-situ solar wind
proton density from ACE. Additional series can be set directly with
:meth:`pint.models.solar_wind_dispersion.SolarWindProxyRegression.set_proxy`,
but not saved to file and reloaded.

Fetching, caching, expiration and failure are handled by
:func:`pint.observatory.global_clock_corrections.get_file`, which keys the
astropy download cache by URL, expires entries against an update interval, and
on a network error warns and returns the stale cached copy.

Series are resolved in two tiers, similar to how
:func:`pint.observatory.find_clock_file` is used for clock corrections:

1. an explicit *path* argument, or the directory named by ``$PINT_SOLAR_INDICES``
2. the upstream source, downloaded and cached


Examples
--------
    >>> from pint import solar_indices
    >>> f107 = solar_indices.get_series("F107")
    >>> ssn = solar_indices.get_series("SSN")

    Offline, using only what is already cached::

    >>> f107 = solar_indices.get_series("F107", download_policy="never")

    Smoothed to the conventional F10.7A window::

    >>> f107a = solar_indices.get_series("F107").smoothed(81, "BOX")

References
----------
``f107``, the 10.7 cm solar radio flux:

- Tapping, K. F. 2013, Space Weather, 11, 394, doi:10.1002/swe.20064
- Measured at the Dominion Radio Astrophysical Observatory, Penticton, and
  distributed by Space Weather Canada:
  https://www.spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-en.php

``ssn``, the international sunspot number:

- Clette, F., Svalgaard, L., Vaquero, J. M., & Cliver, E. W. 2014,
  Space Science Reviews, 186, 35, doi:10.1007/s11214-014-0074-2
- Produced by SILSO, Royal Observatory of Belgium, https://www.sidc.be/SILSO/,
  and read here from the NOAA SWPC solar-cycle indices file.

``ace_density``, solar wind proton density from ACE SWEPAM:

- McComas, D. J., Bame, S. J., Barker, P., et al. 1998, Space Science Reviews,
  86, 563, doi:10.1023/A:1005040232597
- Stone, E. C., Frandsen, A. M., Mewaldt, R. A., et al. 1998, Space Science
  Reviews, 86, 1, doi:10.1023/A:1005082526237
- Level 2 data from the ACE Science Center:
  https://izw1.caltech.edu/ACE/ASC/level2/lvl2DATA_SWEPAM.html
"""

import json
import os
import warnings
from calendar import monthrange
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
from warnings import warn

import astropy.units as u
import numpy as np
from astropy.convolution import (
    Box1DKernel,
    CustomKernel,
    Gaussian1DKernel,
    Kernel1D,
    Trapezoid1DKernel,
    convolve,
)
from astropy.table import Table
from astropy.time import Time
from erfa import ErfaWarning
from loguru import logger as log

from pint.observatory.global_clock_corrections import get_file
from pint.types import file_like, quantity_like, time_like
from pint.utils import compute_hash

__all__ = [
    # series
    "SolarIndexSeries",
    # resolution
    "get_series",
    "check_coverage",
    "SOURCES",
    "Source",
    "local_candidate",
    "pint_solar_env_var",
    # parsers
    "parse_drao_f107",
    "parse_noaa",
    "parse_noaa_ssn",
    "parse_ace_density",
    # epoch conversions
    "mjd_from_ymd",
    "mjd_from_year_month",
    "ACE_EPOCH_MJD",
    # smoothing
    "SMOOTHING_KERNELS",
    "smooth",
    "FWHM_PER_SIGMA",
    "odd",
    # units
    "sfu",
]

pint_solar_env_var = "PINT_SOLAR_INDICES"

# ACE timestamps are seconds since 1996-01-01 00:00:00 UT.
ACE_EPOCH_MJD = Time("1996-01-01 00:00:00", format="iso", scale="utc").mjd

# The solar flux unit, the conventional unit of F10.7, which astropy does not
# define. 1 sfu = 1e-22 W m^-2 Hz^-1 = 1e4 Jy.
# Tapping 2013, Space Weather, 11, 394, doi:10.1002/swe.20064
sfu = u.def_unit("sfu", 1e-22 * u.W / u.m**2 / u.Hz)

# Full width at half maximum of a Gaussian, in units of its standard deviation.
FWHM_PER_SIGMA = float(np.sqrt(8.0 * np.log(2.0)))


def odd(n: float) -> int:
    """*n* rounded up to an odd sample count, which ``CustomKernel`` requires."""
    return int(n) + 1 - int(n) % 2


SMOOTHING_KERNELS: Dict[str, Callable[[int], Kernel1D]] = {
    "box": Box1DKernel,
    "gaussian": lambda window: Gaussian1DKernel(window / FWHM_PER_SIGMA),
    "trapezoid": lambda window: Trapezoid1DKernel(0.75 * window, slope=4.0 / window),
    "triangle": lambda window: CustomKernel(np.bartlett(odd(2 * window))),
    "hann": lambda window: CustomKernel(np.hanning(odd(2 * window))),
}


def smooth(values: np.ndarray, kernel: Kernel1D) -> np.ndarray:
    """Apply *kernel* to *values*, holding the endpoints beyond either end.

    One application path for every kernel. ``boundary="extend"`` pads the series with
    its own endpoint value, so the smoothed series is defined everywhere the raw one
    is and does not taper toward zero at the ends.

    Parameters
    ----------
    values : numpy.ndarray
        Sample values, without units. ``convolve`` cannot take a Quantity in a
        unit it does not know, such as :data:`sfu`.
    kernel : astropy.convolution.Kernel1D

    Returns
    -------
    numpy.ndarray
        The same shape as *values*.
    """
    return convolve(
        values,
        kernel,
        boundary="extend",
        nan_treatment="interpolate",
        normalize_kernel=True,
        preserve_nan=True,
    )


def mjd_from_ymd(
    yyyymmdd: Union[int, str, Sequence[Union[int, str]], np.ndarray]
) -> Union[float, np.ndarray]:
    """MJD at 0h UTC from one date, or an array of dates, of the form YYYYMMDD."""
    digits = np.atleast_1d(np.asarray(yyyymmdd)).astype(np.int64).astype("U8")
    iso = [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in digits]
    with warnings.catch_warnings():
        # The sunspot record starts in 1749, which erfa calls a dubious year.
        warnings.simplefilter("ignore", ErfaWarning)
        mjd = np.atleast_1d(Time(iso, format="iso", scale="utc").mjd)
    return mjd if np.ndim(yyyymmdd) else float(mjd[0])


def mjd_from_year_month(
    tag: Union[str, Sequence[str], np.ndarray]
) -> Union[float, np.ndarray]:
    """MJD of the middle of a month, from one ``YYYY-MM`` tag or an array of them.

    A monthly index is the arithmetic mean of the daily values over the calendar
    month, so the epoch it belongs at is the middle of that month, not a fixed day
    number. Daily epochs in this module are the MJD of the day at 0h, so the mean
    epoch of an N-day month beginning at ``mjd0`` is ``mjd0 + (N - 1) / 2``: a
    31-day month lands on the 16th at 0h and a 30-day month on the 15th at 12h.
    """
    ym = [tuple(int(x) for x in str(t).split("-")[:2]) for t in np.atleast_1d(tag)]
    first = np.atleast_1d(mjd_from_ymd([y * 10000 + m * 100 + 1 for y, m in ym]))
    length = np.array([monthrange(y, m)[1] for y, m in ym], dtype=float)
    middle = first + (length - 1) / 2.0
    return middle if np.ndim(tag) else float(middle[0])


class SolarIndexSeries:
    """A solar activity time series, its units, and where it came from.

    The epochs are an :class:`astropy.time.Time` and the values an
    :class:`astropy.units.Quantity`, so the units are tied to the data
    and the metadata fields record which file an analysis actually read.
    Interpolation, bounds checking and smoothing are methods.

    This follows :class:`pint.observatory.clock_file.ClockFile`.

    Parameters
    ----------
    mjd : array_like
        Sample epochs as MJD, UTC. Must be non-decreasing.
    value : astropy.units.Quantity or array_like
        Sample values. A plain array is taken to be dimensionless, which is right
        for the sunspot number and wrong for anything else, so prefer a Quantity.
    name : str, optional
        The :data:`SOURCES` key this came from, e.g. ``"f107"``, always lower case.
    description : str, optional
        One line naming the quantity and its provider.
    url : str, optional
        Where the file was fetched from, when it was fetched.
    filename : str or Path, optional
        The file it was read from.
    retrieved : astropy.time.Time, optional
        When that file was last written, which is the age of the snapshot rather
        than the age of the data.
    sha256 : str, optional
        Hash of the file read.
    reference : str, optional
        Documentation for the series.
    smooth_days : float, optional
        Width of the smoothing already applied, 0 for the raw series. Set by
        :meth:`smoothed`.
    smooth_kernel : str, optional
        Which of :data:`SMOOTHING_KERNELS` was applied. Set by :meth:`smoothed`.

    Examples
    --------
    >>> from pint import solar_indices
    >>> f107 = solar_indices.get_series("F107")
    >>> f107.values.unit
    Unit("sfu")
    >>> f107.smoothed(81).evaluate([55000.0, 55100.0])
    <Quantity [...] sfu>
    """

    def __init__(
        self,
        mjd: time_like,
        value: quantity_like,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = None,
        filename: Optional[file_like] = None,
        retrieved: Optional[Time] = None,
        sha256: Optional[str] = None,
        reference: Optional[str] = None,
        smooth_days: float = 0.0,
        smooth_kernel: str = "box",
    ):
        mjd = mjd.mjd if isinstance(mjd, Time) else np.asarray(mjd, dtype=float)
        value = u.Quantity(value)
        if mjd.shape != value.shape:
            raise ValueError(f"mjd has {mjd.shape} entries but value has {value.shape}")
        if mjd.size > 1 and not np.all(np.diff(mjd) >= 0):
            i = int(np.where(np.diff(mjd) < 0)[0][0])
            raise ValueError(
                f"Solar index series {name or '?'} is out of order: "
                f"MJD {mjd[i]} > {mjd[i + 1]}"
            )
        self._mjd = mjd
        self._time = None  # built on demand by .time
        self._value = value
        self.name = name
        self.description = description
        self.url = url
        self.filename = filename
        self.retrieved = retrieved
        self.sha256 = sha256
        self.reference = reference
        self.smooth_days = float(smooth_days)
        self.smooth_kernel = smooth_kernel

    @property
    def time(self) -> Time:
        """An :class:`astropy.time.Time` of the sample epochs.

        Built on first access rather than in the constructor. Interpolation works
        from :attr:`mjd`, and the sunspot record reaches back to 1749, where every
        UTC conversion draws an :class:`erfa.ErfaWarning` about a dubious year.
        """
        if self._time is None:
            self._time = Time(self._mjd, format="pulsar_mjd", scale="utc")
        return self._time

    @property
    def values(self) -> u.Quantity:
        """An :class:`astropy.units.Quantity` of the sample values."""
        return self._value

    @property
    def unit(self) -> u.UnitBase:
        """The unit of :attr:`values`."""
        return self._value.unit

    @property
    def mjd(self) -> np.ndarray:
        """The sample epochs as plain MJD floats, for interpolation."""
        return self._mjd

    @property
    def cadence_days(self) -> float:
        """Median spacing of the samples, in days.

        This is what converts a smoothing width in days to a number of samples, so
        the same width means the same physical window on a daily and a monthly
        series. Returns ``nan`` for a series too short to have a spacing.
        """
        return float(np.median(np.diff(self.mjd))) if len(self) > 1 else float("nan")

    def __len__(self) -> int:
        return len(self._mjd)

    def __repr__(self) -> str:
        span = f"MJD {self.mjd[0]:.1f}-{self.mjd[-1]:.1f}" if len(self) else "empty"
        smooth = (
            f", smoothed {self.smooth_days:g} d ({self.smooth_kernel})"
            if self.smooth_days
            else ""
        )
        return (
            f"<SolarIndexSeries {self.name or '?'}: {len(self)} samples, {span}, "
            f"{self.unit if self.unit != u.dimensionless_unscaled else 'dimensionless'}"
            f"{smooth}>"
        )

    def evaluate(
        self, mjd: time_like, limits: Literal["warn", "error"] = "warn"
    ) -> u.Quantity:
        """The series interpolated to *mjd*, held flat beyond either end.

        Parameters
        ----------
        mjd : array_like or astropy.time.Time
            Epochs to evaluate at.
        limits : "warn" or "error"
            What to do about epochs the series does not cover. ``"warn"``, the
            default, holds the endpoint value and says so, which models the missing
            stretch as a constant. ``"error"`` raises instead, which is what a long
            unattended fit usually wants.

        Returns
        -------
        astropy.units.Quantity
            Same shape as *mjd*, in :attr:`unit`.
        """
        if len(self) == 0:
            raise ValueError(f"solar index series {self.name or '?'} is empty")
        epochs = mjd.mjd if isinstance(mjd, Time) else np.asarray(mjd, dtype=float)
        self.check_coverage(epochs, limits=limits)
        raw = self._value.value
        return np.interp(epochs, self.mjd, raw, left=raw[0], right=raw[-1]) * self.unit

    def check_coverage(
        self,
        mjd: time_like,
        tolerance_days: float = 0.0,
        limits: Literal["warn", "error"] = "warn",
    ) -> Tuple[int, int]:
        """How many of *mjd* fall off each end of the series. See :func:`check_coverage`."""
        epochs = mjd.mjd if isinstance(mjd, Time) else mjd
        return check_coverage(
            self.mjd,
            epochs,
            name=self.name or "solar index",
            tolerance_days=tolerance_days,
            limits=limits,
        )

    def smoothed(
        self, width_days: float, kernel: Union[str, Kernel1D] = "box"
    ) -> "SolarIndexSeries":
        """A copy of this series smoothed over *width_days*.

        Parameters
        ----------
        width_days : float
            Width of the smoothing window, in days. 0, or a width narrower than the
            sample spacing, returns the series unchanged.
        kernel : str or astropy.convolution.Kernel1D, optional
            Either a key of :data:`SMOOTHING_KERNELS`, matched without regard to
            case, or a kernel instance. ``"BOX"`` is an unweighted mean over the
            window, which is the F10.7A convention and the recommended choice for
            F10.7; ``"GAUSSIAN"``, ``"TRAPEZOID"``, ``"TRIANGLE"`` and ``"HANN"``
            taper instead. *width_days* is the full width at half maximum for all
            of them, so they are comparable at the same width.

            A kernel instance is built from a width in samples by the caller, so it
            ignores *width_days* except for deciding whether to smooth at all, and it
            cannot be written to a par file. :data:`SMOOTHING_KERNELS` is what
            ``SWPRSMOOTHKERNEL`` can name.

        Returns
        -------
        SolarIndexSeries
            A new series. :attr:`smooth_days` and :attr:`smooth_kernel` record what
            was applied.

        Notes
        -----
        The width is in days and is converted to a sample count using
        :attr:`cadence_days`, so 81 days means 81 samples of a daily series and 3 of
        a monthly one. A fractional sample count is rounded.

        Smoothing is applied here rather than stored, so the raw series is what gets
        cached and a change of width or kernel can be easily applied.
        """
        if width_days <= 0 or len(self) < 2:
            return self
        cadence = self.cadence_days
        window = int(round(width_days / cadence)) if cadence > 0 else 0
        if window > len(self):
            log.warning(
                f"Smoothing width {width_days} d spans {window} samples at the "
                f"{cadence:.3g} d cadence of {self.name or 'this series'}, which has "
                f"only {len(self)}; the result will be nearly constant."
            )
        if window <= 1:
            # astropy rejects a kernel narrower than a sample.
            return self

        if isinstance(kernel, Kernel1D):
            warn(
                f"Smoothing {self.name or 'a solar index series'} with a kernel "
                f"instance, {kernel.__class__.__name__}. A par file names a kernel "
                f"through SWPRSMOOTHKERNEL and can only hold one of "
                f"{sorted(SMOOTHING_KERNELS)}, so a model smoothed this way cannot "
                "be written to a par file or rebuilt from one."
            )
            applied, label = kernel, type(kernel).__name__
        else:
            try:
                applied = SMOOTHING_KERNELS[kernel.lower()](window)
            except KeyError:
                raise ValueError(
                    f"Unknown smoothing kernel {kernel!r} for solar index series "
                    f"{self.name or '?'}; known: {sorted(SMOOTHING_KERNELS)}"
                ) from None
            label = kernel.lower()

        return self._with_values(
            smooth(self._value.value, applied) * self.unit,
            smooth_days=width_days,
            smooth_kernel=label,
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Everything needed to say which version of the series was used."""
        return {
            "name": self.name,
            "description": self.description,
            "unit": str(self.unit),
            "url": self.url,
            "filename": None if self.filename is None else str(self.filename),
            "retrieved": None if self.retrieved is None else self.retrieved.isot,
            "sha256": self.sha256,
            "reference": self.reference,
            "n_samples": len(self),
            "mjd_start": float(self.mjd[0]) if len(self) else None,
            "mjd_end": float(self.mjd[-1]) if len(self) else None,
            "smooth_days": self.smooth_days,
            "smooth_kernel": self.smooth_kernel,
        }

    def _with_values(self, value: u.Quantity, **changes) -> "SolarIndexSeries":
        """This series with different values, carrying the metadata across."""
        fields = dict(
            name=self.name,
            description=self.description,
            url=self.url,
            filename=self.filename,
            retrieved=self.retrieved,
            sha256=self.sha256,
            reference=self.reference,
            smooth_days=self.smooth_days,
            smooth_kernel=self.smooth_kernel,
        )
        fields.update(changes)
        return SolarIndexSeries(self.mjd, value, **fields)


def parse_drao_f107(path: file_like, column: str = "fluxadjflux") -> SolarIndexSeries:
    """Daily F10.7 from the DRAO/Penticton ``fluxtable.txt``.

    The file carries roughly three measurements a day (17:00, 20:00 and 23:00 UT),
    which are averaged to one value per calendar day.

    The column choice is not a detail. ``fluxobsflux`` is the flux as received at
    Earth and carries the Sun-Earth distance annual modulation; ``fluxadjflux`` is
    corrected to 1 AU and is the default here.

    Parameters
    ----------
    path : str or pathlib.Path
    column : str, optional
        ``"fluxadjflux"`` (default), ``"fluxobsflux"`` or ``"fluxursi"``.

    Returns
    -------
    SolarIndexSeries
        One sample per calendar day, with values in :data:`sfu`.

    References
    ----------
    - Tapping, K. F. (2013), The 10.7 cm solar radio flux (F10.7), Space Weather, 11, 394-406, doi:10.1002/swe.20064.
    """
    # A two-line header, names then a row of dashes, over fixed-width columns.
    try:
        table = Table.read(path, format="ascii.fixed_width_two_line")
    except Exception as e:
        raise ValueError(f"{path} is not a DRAO flux table: {e}") from e
    if "fluxdate" not in table.colnames:
        raise ValueError(f"{path} has no 'fluxdate' column; not a DRAO flux table")
    if column not in table.colnames:
        raise ValueError(f"{column!r} not among DRAO columns {table.colnames}")

    table = table[table[column] > 0]
    if len(table) == 0:
        raise ValueError(f"no usable rows parsed from {path}")

    # Several measurements a day collapse to a daily mean, keyed by calendar date.
    table["mjd"] = mjd_from_ymd(np.asarray(table["fluxdate"]))
    daily = table["mjd", column].group_by("mjd").groups.aggregate(np.mean)
    return SolarIndexSeries(
        np.asarray(daily["mjd"], dtype=float),
        np.asarray(daily[column], dtype=float) * sfu,
        name="f107",
        description=f"10.7 cm solar flux, DRAO Penticton, {column}",
        filename=path,
        reference="https://www.spaceweather.gc.ca/forecast-prevision/solar-solaire/solarflux/sx-en.php",
    )


def parse_noaa(
    path: file_like, key: str, unit: u.UnitBase = u.dimensionless_unscaled
) -> SolarIndexSeries:
    """A single quantity from the NOAA SWPC solar-cycle indices JSON.

    Missing values appear as a nominal ``-1.0``. Don't filter by positivity since
    the sunspot number could be ``0.0`` at solar minimum.

    Only the sunspot number is exposed through :data:`SOURCES`. The same file
    carries an ``f10.7`` key, but we draw F10.7 from the DRAO source.

    Returns
    -------
    SolarIndexSeries
        One sample per month, placed at the middle of the month. See
        :func:`mjd_from_year_month`.
    """
    with open(path) as f:
        rows = json.load(f)

    tag, value = [], []
    for r in rows:
        v = r.get(key)
        if v is None or v == -1.0:
            continue
        tag.append(r["time-tag"])
        value.append(float(v))

    if not tag:
        raise ValueError(f"no rows with a usable {key!r} in {path}")

    mjd = np.atleast_1d(mjd_from_year_month(tag))
    order = np.argsort(mjd)
    return SolarIndexSeries(
        np.asarray(mjd, dtype=float)[order],
        np.asarray(value, dtype=float)[order] * unit,
        name=key,
        description=f"Monthly {key}, NOAA SWPC solar-cycle indices",
        filename=path,
        reference="https://www.sidc.be/SILSO/datafiles",
    )


def parse_noaa_ssn(path: file_like) -> SolarIndexSeries:
    """Monthly international sunspot number from the NOAA SWPC JSON.

    Unlike F10.7 in the same file, this runs the full record from 1749-01. A
    sunspot number is a count, so the series is dimensionless.
    """
    series = parse_noaa(path, "ssn")
    series.name = "ssn"
    series.description = "Monthly international sunspot number, NOAA SWPC"
    return series


def parse_ace_density(path: file_like) -> SolarIndexSeries:
    """Solar wind proton density from an ACE Science Center level-2 export.

    Reads the whitespace-delimited text the ASC data retrieval form produces,
    which carries an ``ACEepoch`` column (seconds since 1996-01-01) and a
    ``proton_density`` column in cm^-3, with ``-9999.9`` marking missing data. A
    ``BEGIN DATA`` marker is honoured when present.

    There is no automatic download for this one, which is why it has no entry in
    :data:`SOURCES` with a URL. The ACE Science Center publishes no plain-text
    bulk product carrying proton density: ``MAG_SWEPAM_merged.txt`` holds only the
    field magnitude and the proton speed, and the density lives in per-Carrington
    rotation or per-year HDF4 files that need ``pyhdf``. Export the range needed
    from

        https://izw1.caltech.edu/ACE/ASC/level2/lvl2DATA_SWEPAM.html

    and pass the file to :func:`get_series` as *path*, or put it in the directory
    named by ``$PINT_SOLAR_INDICES``.

    Returns
    -------
    SolarIndexSeries
        Values in cm^-3.
    """
    with open(path) as f:
        lines = f.readlines()

    header, start = None, 0
    for i, line in enumerate(lines):
        if line.strip() == "BEGIN DATA":
            start = i + 1
        if "proton_density" in line:
            header = line.split()
    if header is None:
        raise ValueError(f"{path} has no 'proton_density' column; not an ACE export")
    try:
        i_epoch = header.index("ACEepoch")
    except ValueError as e:
        raise ValueError(f"{path} has no 'ACEepoch' column") from e
    i_den = header.index("proton_density")

    mjd, density = [], []
    for line in lines[start:]:
        fields = line.split()
        if len(fields) <= max(i_epoch, i_den) or not fields[0][0].isdigit():
            continue
        try:
            d = float(fields[i_den])
            e = float(fields[i_epoch])
        except ValueError:
            continue
        if d <= -9999.0:
            continue
        mjd.append(ACE_EPOCH_MJD + e / 86400.0)
        density.append(d)

    if not mjd:
        raise ValueError(f"no usable rows parsed from {path}")

    order = np.argsort(mjd)
    return SolarIndexSeries(
        np.asarray(mjd, dtype=float)[order],
        np.asarray(density, dtype=float)[order] * u.cm**-3,
        name="ace_density",
        description="Solar wind proton density, ACE SWEPAM level 2",
        filename=path,
        reference="https://izw1.caltech.edu/ACE/ASC/level2/lvl2DATA_SWEPAM.html",
    )


class Source(NamedTuple):
    """Where one series comes from and how to read it.

    Attributes
    ----------
    filename : str
        Name of the upstream file, also the name looked for under
        ``$PINT_SOLAR_INDICES``.
    url_base : str or None
        Directory URL the file is fetched from, or None when there is no
        automatic download, as for ``ace_density``.
    update_interval_days : float or None
        How long a cached copy stays fresh, passed to
        :func:`pint.observatory.global_clock_corrections.get_file`.
    parser : callable
        Reads the file and returns a :class:`SolarIndexSeries`.
    unit : astropy.units.UnitBase
        Unit of the series values. :data:`astropy.units.dimensionless_unscaled`
        for a count such as the sunspot number. The parser attaches this to the
        data, so it is carried by the series.
    description : str
        One line naming the quantity and its provider.
    """

    filename: str
    url_base: Optional[str]
    update_interval_days: Optional[float]
    parser: Callable[..., "SolarIndexSeries"]
    unit: u.UnitBase
    description: str


SOURCES = {
    "f107": Source(
        filename="fluxtable.txt",
        url_base="https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/",
        update_interval_days=1,
        parser=parse_drao_f107,
        unit=sfu,
        description="Daily 10.7 cm solar flux adjusted to 1 AU, DRAO Penticton",
    ),
    "ssn": Source(
        filename="observed-solar-cycle-indices.json",
        url_base="https://services.swpc.noaa.gov/json/solar-cycle/",
        update_interval_days=30,
        parser=parse_noaa_ssn,
        unit=u.dimensionless_unscaled,
        description="Monthly international sunspot number, NOAA SWPC",
    ),
    "ace_density": Source(
        filename="ACE_SWEPAM_Data.txt",
        url_base=None,  # no bulk text product upstream; see parse_ace_density
        update_interval_days=None,
        parser=parse_ace_density,
        unit=u.cm**-3,
        description="Solar wind proton density, ACE SWEPAM level 2 (manual export)",
    ),
}


def local_candidate(filename: str) -> Optional[Path]:
    """The copy of *filename* under ``$PINT_SOLAR_INDICES``, if there is one."""
    if pint_solar_env_var not in os.environ:
        return None
    p = Path(os.environ[pint_solar_env_var]) / filename
    return p if p.exists() else None


def _finish(
    series: SolarIndexSeries,
    source: Source,
    path: file_like,
    name: str,
    url: Optional[str] = None,
) -> SolarIndexSeries:
    """Denote a freshly parsed series with where it was read from."""
    series.name = name
    series.url = url
    series.filename = str(path)
    if series.description is None:
        series.description = source.description
    try:
        series.retrieved = Time(os.path.getmtime(path), format="unix", scale="utc")
        series.sha256 = compute_hash(path).hex()
    except OSError as e:  # a file-like object, or one that has since gone away
        log.debug(f"Could not record metadata for solar index {name}: {e}")
    if series.unit != source.unit:
        warn(
            f"Solar index {name} parsed to {series.unit!s} but SOURCES declares "
            f"{source.unit!s}. A unit change would rescale any fitted "
            f"coefficient, so check which column or file was read."
        )
    return series


def get_series(
    name: str = "f107",
    download_policy: Literal[
        "always", "never", "if_expired", "if_missing"
    ] = "if_expired",
    path: Optional[file_like] = None,
    **parser_kwargs,
) -> SolarIndexSeries:
    """Return a solar activity time series as ``(mjd, value)``.

    Parameters
    ----------
    name : str
        One of :data:`SOURCES`: ``"f107"`` (DRAO daily), ``"ssn"`` (NOAA monthly
        sunspot number), or ``"ace_density"``.
    download_policy : str
        Passed to :func:`pint.observatory.global_clock_corrections.get_file`:
        ``"always"``, ``"never"``, ``"if_expired"`` (default) or ``"if_missing"``.
        ``"never"`` makes this offline and deterministic, using only what is
        already cached or supplied locally.
    path : str or pathlib.Path, optional
        Read this file directly and skip resolution entirely. Use it to pin an
        analysis to a fixed snapshot.
    **parser_kwargs
        Passed through to the parser, e.g. ``column="fluxobsflux"``.

    Returns
    -------
    SolarIndexSeries
        The series, with its unit attached to the values and its metadata
        recorded. Use :meth:`SolarIndexSeries.evaluate` to interpolate it and
        :meth:`SolarIndexSeries.get_metadata` to record what was read.
    """
    # SWPRSOURCE is upper case in a par file; SOURCES is keyed in lower case.
    name = name.lower()
    if name not in SOURCES:
        raise KeyError(f"unknown series {name!r}; known: {sorted(SOURCES)}")
    source = SOURCES[name]

    if path is not None:
        log.info(f"Solar index {name} read from the supplied path {path}")
        return _finish(source.parser(path, **parser_kwargs), source, path, name)

    local = local_candidate(source.filename)
    if local is not None:
        log.info(
            f"Solar index {name} read from {local} because of {pint_solar_env_var}"
        )
        return _finish(source.parser(local, **parser_kwargs), source, local, name)

    if source.url_base is not None:
        try:
            resolved = get_file(
                source.filename,
                update_interval_days=source.update_interval_days,
                download_policy=download_policy,
                url_base=source.url_base,
            )
            log.info(f"Solar index {name} read from {resolved}")
            return _finish(
                source.parser(resolved, **parser_kwargs),
                source,
                resolved,
                name,
                url=source.url_base + source.filename,
            )
        except (OSError, KeyError) as e:
            log.warning(
                f"Could not obtain solar index {name} from {source.url_base}: {e}"
            )

    hint = (
        f" There is no automatic download for {name}; export it and pass it as "
        f"path=, put it in ${pint_solar_env_var}, or hand the arrays to "
        f"SolarWindProxyRegression.set_proxy(), which is not serialisable."
        if source.url_base is None
        else ""
    )
    raise FileNotFoundError(
        f"No source available for solar index {name!r} under download_policy="
        f"{download_policy!r}.{hint}"
    )


def check_coverage(
    mjd: time_like,
    toas_mjd: time_like,
    name: str = "solar index",
    tolerance_days: float = 0.0,
    limits: Literal["warn", "error"] = "warn",
) -> Tuple[int, int]:
    """Warn when a series does not span the epochs it will be evaluated at.

    A proxy series that stops short is held at its endpoint value,
    which is handled by elsewhere by the interpolator, which models the
    missing stretch as constant.

    Parameters
    ----------
    mjd : array_like
        Epochs the series covers.
    toas_mjd : array_like
        Epochs it will be evaluated at.
    name : str
        Used in the message.
    tolerance_days : float
        Gaps no larger than this are not reported.
    limits : "warn" or "error"
        Whether an uncovered epoch is a warning, the default, or a
        :exc:`ValueError`. Holding a covariate flat outside its range is a
        modelling choice, so an unattended fit may prefer to refuse.

    Returns
    -------
    (n_before, n_after) : tuple of int
        How many epochs fall off each end. ``(0, 0)`` when fully covered.
    """
    mjd = np.asarray(mjd, dtype=float)
    toas_mjd = np.asarray(toas_mjd, dtype=float)
    if mjd.size == 0 or toas_mjd.size == 0:
        return 0, 0

    lo, hi = float(mjd.min()), float(mjd.max())
    before = toas_mjd < lo - tolerance_days
    after = toas_mjd > hi + tolerance_days
    n_before, n_after = int(before.sum()), int(after.sum())

    if n_before or n_after:
        parts = []
        if n_before:
            parts.append(
                f"{n_before} epoch(s) up to {lo - float(toas_mjd.min()):.0f} d before "
                f"it starts (MJD {lo:.1f})"
            )
        if n_after:
            parts.append(
                f"{n_after} epoch(s) up to {float(toas_mjd.max()) - hi:.0f} d after "
                f"it ends (MJD {hi:.1f})"
            )
        message = (
            f"The {name} series does not cover every epoch it is evaluated at: "
            + ", and ".join(parts)
            + ". Those epochs are modelled with the value held flat at the nearest "
            "endpoint. Extend the series to cover them."
        )
        if limits == "error":
            raise ValueError(message)
        warn(message)
    return n_before, n_after
