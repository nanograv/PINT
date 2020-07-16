from pint.models.astrometry import AstrometryEquatorial, AstrometryEcliptic


def convert_to_equatorial(model):
    """Converts pulsar's PulsarEcliptic coordinates to ICRS,
    adds the AstrometryEquatorial model component with the
    correct values, and removes AstrometryEcliptic component.
    """

    if "AstrometryEcliptic" in model.components:

        c = model.coords_as_ICRS()
        a = AstrometryEquatorial()

        a.POSEPOCH = model.POSEPOCH
        a.PX = model.PX

        a.RAJ.quantity = c.ra
        a.DECJ.quantity = c.dec
        a.PMRA.quantity = c.pm_ra_cosdec
        a.PMDEC.quantity = c.pm_dec

        model.add_component(a)
        model.remove_component("AstrometryEcliptic")

    else:
        pass

    return model


def convert_to_ecliptic(model):
    """Converts pulsar's ICRS coordinates to PulsarEcliptic,
    adds the AstrometryEcliptic model component with the
    correct values, and removes AstrometryEquatorial component.
    """

    # Initial validation to make sure model is not in a bad state?

    if "AstrometryEquatorial" in model.components:

        c = model.coords_as_ECL()
        a = AstrometryEcliptic()

        a.POSEPOCH = model.POSEPOCH
        a.PX = model.PX

        a.ELONG.quantity = c.lon
        a.ELAT.quantity = c.lat
        a.PMELONG.quantity = c.pm_lon_coslat
        a.PMELAT.quantity = c.pm_lat

        model.add_component(a)
        model.remove_component("AstrometryEquatorial")

    else:
        pass

    return model
