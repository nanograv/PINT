"""Work with Fermi TOAs."""
from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np
from astropy import log
from astropy.coordinates import SkyCoord

import pint.toa as toa
from pint.fits_utils import read_fits_event_mjds_tuples
from pint.observatory import get_observatory

__all__ = ["load_Fermi_TOAs"]


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
            psfpar0 * psfpar0 * np.power(100.0 / energies, 2.0 * psfpar1)
            + psfpar2 * psfpar2
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
    minmjd=0.0,
    maxmjd=np.inf,
):
    """
    TOAlist = load_Fermi_TOAs(ft1name)
      Read photon event times out of a Fermi FT1 file and return
      a list of PINT TOA objects.
      Correctly handles raw FT1 files, or ones processed with gtbary
      to have barycentered or geocentered TOAs.

      weightcolumn specifies the FITS column name to read the photon weights
      from.  The special value 'CALC' causes the weights to be computed empirically
      as in Philippe Bruel's SearchPulsation code.
      logeref and logesig are parameters for the weight computation and are only
      used when weightcolumn='CALC'.

      When weights are loaded, or computed, events are filtered by weight >= minweight
    """
    import astropy.io.fits as pyfits

    # Load photon times from FT1 file
    hdulist = pyfits.open(ft1name)
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

    # limit the TOAs to ones in selected MJD range
    mjds_float = np.array([r[0] + r[1] for r in mjds])
    idx = np.logical_and((mjds_float > minmjd), (mjds_float < maxmjd))
    mjds = mjds[idx]
    energies = energies[idx]
    if weightcolumn is not None:
        weights = weights[idx]

    if timesys == "TDB":
        log.info("Building barycentered TOAs")
        if weightcolumn is None:
            toalist = [
                toa.TOA(m, obs="Barycenter", scale="tdb", energy=e, error=1.0 * u.us)
                for m, e in zip(mjds, energies)
            ]
        else:
            toalist = [
                toa.TOA(
                    m,
                    obs="Barycenter",
                    scale="tdb",
                    energy=e,
                    weight=w,
                    error=1.0 * u.us,
                )
                for m, e, w in zip(mjds, energies, weights)
            ]
    else:
        if timeref == "LOCAL":
            log.info(
                "Building spacecraft local TOAs, with MJDs in range {0} to {1}".format(
                    mjds[0], mjds[-1]
                )
            )
            assert timesys == "TT"
            try:
                fermiobs = get_observatory("Fermi")
            except KeyError:
                log.error(
                    "Fermi observatory not defined. Make sure you have specified an FT2 file!"
                )
                raise

            try:
                if weightcolumn is None:
                    toalist = [
                        toa.TOA(m, obs="Fermi", scale="tt", energy=e, error=1.0 * u.us)
                        for m, e in zip(mjds, energies)
                    ]
                else:
                    toalist = [
                        toa.TOA(
                            m,
                            obs="Fermi",
                            scale="tt",
                            energy=e,
                            weight=w,
                            error=1.0 * u.us,
                        )
                        for m, e, w in zip(mjds, energies, weights)
                    ]
            except KeyError:
                log.error(
                    "Error processing Fermi TOAs. You may have forgotten to specify an FT2 file with --ft2"
                )
                raise
        else:
            log.info("Building geocentered TOAs")
            if weightcolumn is None:
                toalist = [
                    toa.TOA(m, obs="Geocenter", scale="tt", energy=e, error=1.0 * u.us)
                    for m, e in zip(mjds, energies)
                ]
            else:
                toalist = [
                    toa.TOA(
                        m,
                        obs="Geocenter",
                        scale="tt",
                        energy=e,
                        weight=w,
                        error=1.0 * u.us,
                    )
                    for m, e, w in zip(mjds, energies, weights)
                ]

    return toalist
