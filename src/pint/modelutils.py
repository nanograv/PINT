from loguru import logger as log
from pint.models.astrometry import AstrometryEquatorial, AstrometryEcliptic


# FIXME: shouldn't this be in the AstrometryEquatorial and AstrometryEcliptic classes?


def model_ecliptic_to_equatorial(model, force=False):
    r"""Converts Astrometry model component, Ecliptic to Equatorial

    Parameters
    ----------
    model: `pint.models.TimingModel` object
        current model with AstrometryEcliptic component
    force: boolean, optional
        will force conversion even if an equatorial component is already present

    Returns
    -------
    model
        new model with AstrometryEquatorial component
    """

    if "AstrometryEquatorial" not in model.components or force:
        if "AstrometryEquatorial" in model.components:
            log.warning(
                "Equatorial coordinates already present but re-calculating anyway"
            )

        if "AstrometryEcliptic" not in model.components:
            raise AttributeError(
                "Requested conversion to equatorial coordinates, but no alternate coordinates found"
            )

        c = model.coords_as_ICRS()
        a = AstrometryEquatorial()

        a.POSEPOCH = model.POSEPOCH
        a.PX = model.PX

        a.RAJ.quantity = c.ra
        a.DECJ.quantity = c.dec
        a.PMRA.quantity = c.pm_ra_cosdec
        a.PMDEC.quantity = c.pm_dec

        model.remove_component("AstrometryEcliptic")
        model.add_component(a)

        model.setup()
        model.validate()

    else:
        log.warning("Equatorial coordinates already present; not re-calculating")

    return model


def model_equatorial_to_ecliptic(model, force=False):
    """Converts Astrometry model component, Equatorial to Ecliptic

    Parameters
    ----------
    model: `pint.models.TimingModel` object
        current model with AstrometryEquatorial component
    force: boolean, optional
        will force conversion even if an ecliptic component is already present

    Returns
    -------
    model
        new model with AstrometryEcliptic component
    """

    if "AstrometryEcliptic" not in model.components or force:
        if "AstrometryEcliptic" in model.components:
            log.warning(
                "Ecliptic coordinates already present but re-calculating anyway"
            )
        if "AstrometryEquatorial" not in model.components:
            raise AttributeError(
                "Requested conversion to ecliptic coordinates, but no alternate coordinates found"
            )

        c = model.coords_as_ECL()
        a = AstrometryEcliptic()

        a.POSEPOCH = model.POSEPOCH
        a.PX = model.PX

        a.ELONG.quantity = c.lon
        a.ELAT.quantity = c.lat
        a.PMELONG.quantity = c.pm_lon_coslat
        a.PMELAT.quantity = c.pm_lat

        model.remove_component("AstrometryEquatorial")
        model.add_component(a)

        model.setup()
        model.validate()

    else:
        log.warning("Ecliptic coordinates already present; not re-calculating")

    return model
