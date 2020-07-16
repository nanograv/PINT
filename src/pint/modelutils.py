from pint.models.astrometry import AstrometryEquatorial, AstrometryEcliptic


def convert_to_equatorial(model):
    """Converts Astrometry model component, Ecliptic to Equatorial

      Parameters
      ----------
      model
          current model with AstrometryEcliptic component

      Returns
      -------
      model
          new model with AstrometryEquatorial component
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
    """Converts Astrometry model component, Equatorial to Ecliptic

      Parameters
      ----------
      model
          current model with AstrometryEquatorial component

      Returns
      -------
      model
          new model with AstrometryEcliptic component
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
