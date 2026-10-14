"""Solar activity index downloading, caching and parsing.

Provides the external time series that solar-wind models need as covariates:
the 10.7 cm radio flux (F10.7), the sunspot number, and in-situ solar wind
proton density from ACE.

Fetching, caching, expiry and failure
handling are all delegated to
:func:`pint.observatory.global_clock_corrections.get_file`, which keys the
astropy download cache by URL, expires entries against an update interval, and
on a network error warns and returns the stale cached copy rather than raising.
That last behaviour is what makes this safe to call from a long analysis run.

Series are resolved in three tiers, similar to how
:func:`pint.observatory.find_clock_file` is used for clock corrections:

1. an explicit *path* argument, or the directory named by ``$PINT_SOLAR_INDICES``
2. the upstream source, downloaded and cached
3. a derived fallback bundled in ``pint/data/runtime/``

Tier 1 sets an analysis to a fixed snapshot so the
content at a fixed upstream URL changes as new data arrive.

Examples
--------
    >>> from pint import solar_indices
    >>> mjd, flux = solar_indices.get_series("f107")
    >>> mjd, ssn = solar_indices.get_series("ssn")

    Offline, using only what is already cached::

    >>> mjd, flux = solar_indices.get_series("f107", download_policy="never")
"""

import json
import os
from collections import namedtuple
from datetime import date
from pathlib import Path
from warnings import warn

import numpy as np
from loguru import logger as log

from pint.config import runtimefile
from pint.observatory.global_clock_corrections import get_file

__all__ = [
    # resolution and the series themselves
    "get_series",
    "check_coverage",
    "SOURCES",
    "Source",
    "local_candidate",
    "pint_solar_env_var",
    # parsers, usable directly on a file you already have
    "parse_drao_f107",
    "parse_noaa",
    "parse_noaa_ssn",
    "parse_ace_density",
    # epoch conversions
    "mjd_from_ymd",
    "mjd_from_year_month",
    "MJD_ORDINAL_OFFSET",
    "ACE_EPOCH_MJD",
]

pint_solar_env_var = "PINT_SOLAR_INDICES"

# MJD 0 is 1858-11-17, so MJD = date.toordinal() - MJD_ORDINAL_OFFSET.
MJD_ORDINAL_OFFSET = date(1858, 11, 17).toordinal()

# ACE timestamps are seconds since 1996-01-01 00:00:00 UT.
ACE_EPOCH_MJD = date(1996, 1, 1).toordinal() - MJD_ORDINAL_OFFSET


def mjd_from_ymd(yyyymmdd):
    """Integer MJD from an integer or string date of the form YYYYMMDD."""
    s = str(int(yyyymmdd))
    return (
        date(int(s[:4]), int(s[4:6]), int(s[6:8])).toordinal() - MJD_ORDINAL_OFFSET
    )


def mjd_from_year_month(tag):
    """MJD of the middle of a month, from a ``YYYY-MM`` tag.

    Day 15 by convention, matching how monthly solar indices are normally plotted
    against time.
    """
    year, month = (int(x) for x in tag.split("-")[:2])
    return date(year, month, 15).toordinal() - MJD_ORDINAL_OFFSET


# ── parsers ───────────────────────────────────────────────────────────────────
#
# Each returns (mjd, value), sorted by mjd, with missing data removed.


def parse_drao_f107(path, column="fluxadjflux"):
    """Daily F10.7 from the DRAO/Penticton ``fluxtable.txt``.

    The file carries roughly three measurements a day (17:00, 20:00 and 23:00 UT),
    which are averaged to one value per calendar day.

    The column choice is not a detail. ``fluxobsflux`` is the flux as received at
    Earth and carries the Sun-Earth distance annual modulation; ``fluxadjflux`` is
    corrected to 1 AU and is the default here. Using the observed column would put
    a spurious annual term into the covariate at precisely the frequency the solar
    wind signal occupies, which is the one place it could not be told apart.

    This is the only F10.7 source offered. NOAA SWPC publishes a monthly F10.7 in
    its solar-cycle indices JSON, but it is not an independent measurement: it is
    the monthly mean of this same Penticton series, and of the **observed** column
    rather than the adjusted one. Measured over 263 complete months, NOAA divided
    by the monthly mean of ``fluxobsflux`` has ratio 0.99938 with scatter 0.0100,
    against 0.99963 with scatter 0.0256 for ``fluxadjflux``; the extra scatter is
    exactly the annual term, since sqrt(0.0236^2 + 0.0100^2) = 0.0256 and 0.0236 is
    that term's standard deviation. Offering it would mean offering the same data,
    coarser, in the convention this function exists to avoid.

    Parameters
    ----------
    path : str or pathlib.Path
    column : str, optional
        ``"fluxadjflux"`` (default), ``"fluxobsflux"`` or ``"fluxursi"``.

    Returns
    -------
    mjd : ndarray of float, one entry per day
    flux : ndarray of float, solar flux units
    """
    with open(path) as f:
        lines = f.readlines()

    header = None
    for line in lines:
        if "fluxdate" in line:
            header = line.split()
            break
    if header is None:
        raise ValueError(f"{path} has no 'fluxdate' header line; not a DRAO flux table")
    if column not in header:
        raise ValueError(f"{column!r} not among DRAO columns {header}")
    icol = header.index(column)

    by_day = {}
    for line in lines:
        fields = line.split()
        if len(fields) <= icol or not fields[0].isdigit():
            continue
        try:
            value = float(fields[icol])
        except ValueError:
            continue
        if value <= 0:
            continue
        by_day.setdefault(mjd_from_ymd(fields[0]), []).append(value)

    if not by_day:
        raise ValueError(f"no usable rows parsed from {path}")

    mjd = np.array(sorted(by_day), dtype=float)
    flux = np.array([np.mean(by_day[m]) for m in mjd], dtype=float)
    return mjd, flux


def parse_noaa(path, key):
    """A single quantity from the NOAA SWPC solar-cycle indices JSON.

    Missing values appear as a nominal ``-1.0``. Note this is not the same as
    filtering on positivity: the sunspot number is legitimately ``0.0`` at solar
    minimum.

    Only the sunspot number is exposed through :data:`SOURCES`. The same file
    carries an ``f10.7`` key, but we draw f10.7 from the DRAO source.
    """
    with open(path) as f:
        rows = json.load(f)

    mjd, value = [], []
    for r in rows:
        v = r.get(key)
        if v is None or v == -1.0:
            continue
        mjd.append(mjd_from_year_month(r["time-tag"]))
        value.append(float(v))

    if not mjd:
        raise ValueError(f"no rows with a usable {key!r} in {path}")

    order = np.argsort(mjd)
    return np.asarray(mjd, dtype=float)[order], np.asarray(value, dtype=float)[order]


def parse_noaa_ssn(path):
    """Monthly international sunspot number from the NOAA SWPC JSON.

    Unlike F10.7 in the same file, this runs the full record from 1749-01.
    """
    return parse_noaa(path, "ssn")


def parse_ace_density(path):
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
    mjd : ndarray of float
    density : ndarray of float, cm^-3
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
    return np.asarray(mjd, dtype=float)[order], np.asarray(density, dtype=float)[order]


# Sources

Source = namedtuple(
    "Source",
    ["filename", "url_base", "update_interval_days", "parser", "units", "description"],
)

SOURCES = {
    "f107": Source(
        filename="fluxtable.txt",
        url_base="https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/",
        update_interval_days=1,
        parser=parse_drao_f107,
        units="sfu",
        description="Daily 10.7 cm solar flux adjusted to 1 AU, DRAO Penticton",
    ),
    "ssn": Source(
        filename="observed-solar-cycle-indices.json",
        url_base="https://services.swpc.noaa.gov/json/solar-cycle/",
        update_interval_days=30,
        parser=parse_noaa_ssn,
        units="",
        description="Monthly international sunspot number, NOAA SWPC",
    ),
    "ace_density": Source(
        filename="ACE_SWEPAM_Data.txt",
        url_base=None,  # no bulk text product upstream; see parse_ace_density
        update_interval_days=None,
        parser=parse_ace_density,
        units="cm^-3",
        description="Solar wind proton density, ACE SWEPAM level 2 (manual export)",
    ),
}


def local_candidate(filename):
    """The copy of *filename* under ``$PINT_SOLAR_INDICES``, if there is one."""
    if pint_solar_env_var not in os.environ:
        return None
    p = Path(os.environ[pint_solar_env_var]) / filename
    return p if p.exists() else None


def get_series(name="f107", download_policy="if_expired", path=None, **parser_kwargs):
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
    mjd : ndarray of float
    value : ndarray of float, in the source's units
    """
    if name not in SOURCES:
        raise KeyError(f"unknown series {name!r}; known: {sorted(SOURCES)}")
    source = SOURCES[name]

    if path is not None:
        log.info(f"Solar index {name} read from the supplied path {path}")
        return source.parser(path, **parser_kwargs)

    local = local_candidate(source.filename)
    if local is not None:
        log.info(f"Solar index {name} read from {local} because of {pint_solar_env_var}")
        return source.parser(local, **parser_kwargs)

    if source.url_base is not None:
        try:
            resolved = get_file(
                source.filename,
                update_interval_days=source.update_interval_days,
                download_policy=download_policy,
                url_base=source.url_base,
            )
            log.info(f"Solar index {name} read from {resolved}")
            return source.parser(resolved, **parser_kwargs)
        except (OSError, KeyError) as e:
            log.warning(f"Could not obtain solar index {name} from {source.url_base}: {e}")

    fallback = Path(runtimefile(source.filename))
    if fallback.exists():
        warn(
            f"Solar index {name} is being read from the bundled fallback "
            f"{fallback}, which is a snapshot and will be out of date. Set "
            f"${pint_solar_env_var} or allow a download to use current data."
        )
        return source.parser(fallback, **parser_kwargs)

    hint = (
        f" There is no automatic download for {name}; export it and pass it as "
        f"path=, or put it in ${pint_solar_env_var}."
        if source.url_base is None
        else ""
    )
    raise FileNotFoundError(
        f"No source available for solar index {name!r} under download_policy="
        f"{download_policy!r}.{hint}"
    )


def check_coverage(mjd, toas_mjd, name="solar index", tolerance_days=0.0):
    """Warn when a series does not span the epochs it will be evaluated at.

    A covariate that stops short is held at its endpoint value, which models the
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
        warn(
            f"The {name} series does not cover every epoch it is evaluated at: "
            + ", and ".join(parts)
            + ". Those epochs are modelled with the value held flat at the nearest "
            "endpoint. Extend the series to cover them."
        )
    return n_before, n_after
