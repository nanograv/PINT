"""Work with Fermi TOAs."""

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astropy.io import fits
import astropy.units as u
import numpy as np
from loguru import logger as log

import pint.toa as toa
from pint.fits_utils import read_fits_event_mjds_tuples
from pint.observatory import get_observatory

# default TOA (event) uncertainty depending on facility
_default_uncertainty = 1 * u.us

__all__ = ["load_Fermi_TOAs", "get_Fermi_TOAs"]


def calc_lat_weights(energies, angseps, logeref=4.1, logesig=0.5):
    """Compute photon weights based on PSF.

    This function computes photon weights based on the energy-dependent
    PSF, as defined in Philippe Bruel's SearchPulsation code.
    It was built by David Smith, based on some code from Lucas Guillemot.
    This computation uses only the PSF as a function of energy, not a full
    spectral model of the region, so is less exact than gtsrcprob.

    Parameters
    ----------
    energies : np.array
        Array of photon energies in MeV
    angseps : array of astropy Angles
        Angular separations between photon direction and target
        This should be astropy Angle array, such as returned from
        SkyCoord_photons.separation(SkyCoord_target)
    logeref : Parameter
        from SearchPulsation optimization
    logesig : Parameter
        from SearchPulsation optimization

    Returns
    -------
    np.array
        weights (probabilities that the photons came from the target, based on the PSF).
    """
    # A few parameters that define the PSF shape
    psfpar0 = 5.445
    psfpar1 = 0.848
    psfpar2 = 0.084
    norm = 1.0
    gam = 2.0
    scalepsf = 3.0

    logE = np.log10(energies)

    sigma = (
        np.sqrt(
            (psfpar0**2 * np.power(100.0 / energies, 2.0 * psfpar1) + psfpar2**2)
        )
        / scalepsf
    )

    fgeom = norm * np.power(
        1 + angseps.degree * angseps.degree / 2.0 / gam / sigma / sigma, -gam
    )

    return fgeom * np.exp(-np.power((logE - logeref) / np.sqrt(2.0) / logesig, 2.0))


def load_Fermi_TOAs(
    ft1name,
    weightcolumn=None,
    targetcoord=None,
    logeref=4.1,
    logesig=0.5,
    minweight=0.0,
    minmjd=-np.inf,
    maxmjd=np.inf,
    fermiobs="Fermi",
    errors=_default_uncertainty,
):
    """
    Read photon event times out of a Fermi FT1 file and return a list of PINT :class:`~pint.toa.TOA` objects.

    Correctly handles raw FT1 files, or ones processed with gtbary
    to have barycentered or geocentered TOAs.


    Parameters
    ----------
    weightcolumn : str
        Specifies the FITS column name to read the photon weights from.
        The special value 'CALC' causes the weights to be computed
        empirically as in Philippe Bruel's SearchPulsation code.
    targetcoord : astropy.coordinates.SkyCoord
        Source coordinate for weight computation if weightcolumn=='CALC'
    logeref : float
        Parameter for the weight computation if weightcolumn=='CALC'
    logesig : float
        Parameter for the weight computation if weightcolumn=='CALC'
    minweight : float
        If weights are loaded or computed, exclude events with smaller weights.
    minmjd : float
        Events with earlier MJDs are excluded.
    maxmjd : float
        Events with later MJDs are excluded.
    fermiobs: str
      The default observatory name is Fermi, and must have already been
      registered.  The user can specify another name
    errors : astropy.units.Quantity or float, optional
        The uncertainty on the TOA; if it's a float it is assumed to be
        in microseconds

    Returns
    -------
    toalist : list
        A list of :class:`~pint.toa.TOA` objects corresponding to the Fermi events.

    Note
    ----
    This list should be converted into a :class:`~pint.toa.TOAs` object with
    :func:`pint.toa.get_TOAs_list` for most operations

    See Also
    --------
    :func:`get_Fermi_TOAs`

    """
    t = get_Fermi_TOAs(
        ft1name,
        weightcolumn=weightcolumn,
        targetcoord=targetcoord,
        logeref=logeref,
        logesig=logesig,
        minweight=minweight,
        minmjd=minmjd,
        maxmjd=maxmjd,
        fermiobs=fermiobs,
        errors=errors,
    )
    return t.to_TOA_list()


def get_Fermi_TOAs(
    ft1name,
    weightcolumn=None,
    targetcoord=None,
    logeref=4.1,
    logesig=0.5,
    minweight=0.0,
    minmjd=-np.inf,
    maxmjd=np.inf,
    fermiobs="Fermi",
    ephem=None,
    planets=False,
    include_bipm=False,
    include_gps=False,
    errors=_default_uncertainty,
):
    """
      Read photon event times out of a Fermi FT1 file and return a :class:`pint.toa.TOAs` object

      Correctly handles raw FT1 files, or ones processed with gtbary
      to have barycentered or geocentered TOAs.

    Parameters
    ----------
    weightcolumn : str
        Specifies the FITS column name to read the photon weights from.
        The special value ``CALC`` causes the weights to be computed
        empirically as in Philippe Bruel's SearchPulsation code.
    targetcoord : astropy.coordinates.SkyCoord
        Source coordinate for weight computation if weightcolumn=='CALC'
    logeref : float
        Parameter for the weight computation if weightcolumn=='CALC'
    logesig : float
        Parameter for the weight computation if weightcolumn=='CALC'
    minweight : float
        If weights are loaded or computed, exclude events with smaller weights.
    minmjd : float
        Events with earlier MJDs are excluded.
    maxmjd : float
        Events with later MJDs are excluded.
    fermiobs: str
      The default observatory name is Fermi, and must have already been
      registered.  The user can specify another name
    ephem : str, optional
        The name of the solar system ephemeris to use; defaults to "DE421".
    planets : bool, optional
        Whether to apply Shapiro delays based on planet positions. Note that a
        long-standing TEMPO2 bug in this feature went unnoticed for years.
        Defaults to False.
    include_bipm : bool, optional
        Use TT(BIPM) instead of TT(TAI)
    include_gps : bool, optional
        Apply GPS to UTC clock corrections
    errors : astropy.units.Quantity or float, optional
        The uncertainty on the TOA; if it's a float it is assumed to be
        in microseconds

    Returns
    -------
    pint.toa.TOAs

    Examples
    -------
    To create a :class:`pint.toa.TOAs` object, you need an event file and a spacecraft
    orbit file (called ``ft2file``)::

        >>> import pint.toa as toa
        >>> from pint.fermi_toas import get_Fermi_TOAs
        >>> from pint.observatory.satellite_obs import get_satellite_observatory
        >>> get_satellite_observatory("Fermi", ft2file, overwrite=True)
        >>> toas = get_Fermi_TOAs(eventfile, weightcolumn="PSRJ0030+0451", ephem="DE405")

    """

    # Load photon times from FT1 file
    hdulist = fits.open(ft1name)
    ft1hdr = hdulist[1].header
    ft1dat = hdulist[1].data

    # TIMESYS will be 'TT' for unmodified Fermi LAT events (or geocentered), and
    #                 'TDB' for events barycentered with gtbary
    # TIMEREF will be 'GEOCENTER' for geocentered events,
    #                 'SOLARSYSTEM' for barycentered,
    #             and 'LOCAL' for unmodified events

    timesys = ft1hdr["TIMESYS"]
    log.info("TIMESYS {0}".format(timesys))
    timeref = ft1hdr["TIMEREF"]
    log.info("TIMEREF {0}".format(timeref))

    # Read time column from FITS file
    mjds = read_fits_event_mjds_tuples(hdulist[1])
    if len(mjds) == 0:
        log.error("No MJDs read from file!")
        raise

    energies = ft1dat.field("ENERGY") * u.MeV
    if weightcolumn is not None:
        if weightcolumn == "CALC":
            photoncoords = SkyCoord(
                ft1dat.field("RA") * u.degree,
                ft1dat.field("DEC") * u.degree,
                frame="icrs",
            )
            weights = calc_lat_weights(
                ft1dat.field("ENERGY"),
                photoncoords.separation(targetcoord),
                logeref=logeref,
                logesig=logesig,
            )
        else:
            weights = ft1dat.field(weightcolumn)
        if minweight > 0.0:
            idx = np.where(weights > minweight)[0]
            mjds = mjds[idx]
            energies = energies[idx]
            weights = weights[idx]

    if not isinstance(errors, u.Quantity):
        errors = errors * u.microsecond

    # limit the TOAs to ones in selected MJD range
    mjds_float = np.asarray([r[0] + r[1] for r in mjds])
    idx = (minmjd < mjds_float) & (mjds_float < maxmjd)
    mjds = mjds[idx]
    energies = energies[idx]
    if weightcolumn is not None:
        weights = weights[idx]

    if timesys == "TDB":
        log.info("Building barycentered TOAs")
        obs = "Barycenter"
        scale = "tdb"
        msg = "barycentric"
        location = None
    elif (timesys == "TT") and (timeref == "LOCAL"):
        assert timesys == "TT"
        try:
            get_observatory(fermiobs)
        except KeyError:
            log.error(
                f"{fermiobs} observatory not defined. Make sure you have specified an FT2 file!"
            )
            raise
        obs = fermiobs
        scale = "tt"
        msg = "spacecraft local"
        location = None
    elif (timesys == "TT") and (timeref == "GEOCENTRIC"):
        obs = "Geocenter"
        scale = "tt"
        msg = "geocentric"
        location = EarthLocation(0, 0, 0)
    else:
        raise ValueError("Unrecognized TIMEREF/TIMESYS.")

    log.info(
        f"Building {msg} TOAs, with MJDs in range {mjds[0, 0] + mjds[0, 1]} to {mjds[-1, 0] + mjds[-1, 1]}"
    )
    if len(mjds.shape) == 2:
        t = Time(
            val=mjds[:, 0],
            val2=mjds[:, 1],
            format="mjd",
            scale=scale,
            location=location,
        )
    else:
        t = Time(mjds, format="mjd", scale=scale, location=location)
    if weightcolumn is None:
        return toa.get_TOAs_array(
            t,
            obs,
            errors=errors,
            include_gps=include_gps,
            include_bipm=include_bipm,
            planets=planets,
            ephem=ephem,
            flags=[{"energy": str(e)} for e in energies.to_value(u.MeV)],
        )
    else:
        return toa.get_TOAs_array(
            t,
            obs,
            errors=errors,
            include_gps=False,
            include_bipm=False,
            planets=planets,
            ephem=ephem,
            flags=[
                {"energy": str(e), "weight": str(w)}
                for e, w in zip(energies.to_value(u.MeV), weights)
            ],
        )
