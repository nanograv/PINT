"""Special locations that are not really observatories.

Special "site" locations (eg, barycenter) which do not need clock
corrections or much else done.

Can be loaded using :func:`pint.observatory.special_locations.load_special_locations`, which is run on import.
Otherwise it only needs to be run if :func:`pint.observatory.Observatory.clear_registry` is run.

See Also
--------
:mod:`pint.observatory.topo_obs`
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from loguru import logger as log

from pint.solar_system_ephemerides import objPosVel_wrt_SSB
from pint.utils import PosVel

from . import Observatory

__all__ = [
    "SpecialLocation",
    "BarycenterObs",
    "GeocenterObs",
    "T2SpacecraftObs",
    "load_special_locations",
]


class SpecialLocation(Observatory):
    """Special locations that are not really observatories.

    Observatory-derived class for special sites that are not really
    observatories but sometimes are used as TOA locations (eg, solar
    system barycenter).  Currently the only feature of this class is
    that clock corrections are zero.

    Parameters
    ----------
    name : string
        The name of the observatory
    aliases : str, optional
        List of other aliases for the observatory name.
    apply_gps2utc : bool or None
        Set False to disable UTC(GPS)->UTC clock correction
    overwrite : bool, optional
        If True, allow redefinition of an existing observatory; if False,
        raise an exception.
    """

    def __init__(
        self,
        name,
        aliases=None,
        apply_gps2utc=None,
        overwrite=False,
    ):
        super().__init__(
            name,
            aliases=aliases,
            apply_gps2utc=apply_gps2utc,
            overwrite=overwrite,
        )

        self.origin = "Built-in special location."


class BarycenterObs(SpecialLocation):
    """Observatory-derived class for the solar system barycenter.

    Time scale is assumed to be TDB, so no GPS or BIPM corrections should be applied."""

    def __init__(
        self,
        name,
        aliases=None,
        overwrite=False,
    ):
        super().__init__(
            name,
            aliases=aliases,
            apply_gps2utc=False,
            overwrite=overwrite,
        )

    @property
    def timescale(self):
        return "tdb"

    @property
    def tempo_code(self):
        return "@"

    @property
    def tempo2_code(self):
        return "bat"

    def get_gcrs(self, t, ephem=None):
        if ephem is None:
            raise ValueError("Ephemeris needed for BarycenterObs get_gcrs")
        ssb_pv = objPosVel_wrt_SSB("earth", t, ephem)
        return -1 * ssb_pv.pos

    def posvel(self, t, ephem, group=None):
        vdim = (3,) + t.shape
        return PosVel(
            np.zeros(vdim) * u.m,
            np.zeros(vdim) * u.m / u.s,
            obj=self.name,
            origin="ssb",
        )


class GeocenterObs(SpecialLocation):
    """Observatory-derived class for the Earth geocenter."""

    def __init__(
        self,
        name,
        aliases=None,
        apply_gps2utc=False,
        overwrite=False,
        tempo_code_str="0",
        tempo2_code_str="coe",
    ):
        self.tempo_code_str = tempo_code_str
        self.tempo2_code_str = tempo2_code_str
        super().__init__(
            name,
            aliases=aliases,
            apply_gps2utc=apply_gps2utc,
            overwrite=overwrite,
        )

    @property
    def timescale(self):
        return "utc"

    def earth_location_itrf(self, time=None):
        return EarthLocation.from_geocentric(0.0, 0.0, 0.0, unit=u.m)

    @property
    def tempo_code(self):
        return self.tempo_code_str

    @property
    def tempo2_code(self):
        return self.tempo2_code_str

    def get_gcrs(self, t, ephem=None):
        vdim = (3,) + t.shape
        return np.zeros(vdim) * u.m

    def posvel(self, t, ephem, group=None):
        return objPosVel_wrt_SSB("earth", t, ephem)


class T2SpacecraftObs(SpecialLocation):
    """An observatory with position tabulated following Tempo2 convention.

    In tempo2, it is possible to specify the GCRS position of the
    observatory via the -telx, -tely, and -telz flags in a TOA file.  This
    class is able to obtain its position in this way, i.e. by examining the
    flags in a TOA table.

    Since the source of timing for spacecraft TOAs is unknown, by default the GPS to UTC correction is not applied.
    """

    def __init__(
        self,
        name,
        aliases=None,
        apply_gps2utc=False,
        overwrite=False,
    ):
        super().__init__(
            name,
            aliases=aliases,
            apply_gps2utc=apply_gps2utc,
            overwrite=overwrite,
        )

    @property
    def timescale(self):
        return "utc"

    @property
    def tempo_code(self):
        return None

    def get_gcrs(self, t, group, ephem=None):
        """Return spacecraft GCRS position; this assumes position flags in tim file are in km"""

        if group is None:
            raise ValueError("TOA group table needed for SpacecraftObs get_gcrs")

        try:
            x = np.array([float(flags["telx"]) for flags in group["flags"]])
            y = np.array([float(flags["tely"]) for flags in group["flags"]])
            z = np.array([float(flags["telz"]) for flags in group["flags"]])
        except:
            log.error(
                "Missing flag. TOA line should have telx,tely,telz flags for GCRS position in km."
            )
            raise ValueError(
                "Missing flag. TOA line should have telx,tely,telz flags for GCRS position in km."
            )

        pos = np.vstack((x, y, z))
        vdim = (3,) + t.shape
        if pos.shape != vdim:
            raise ValueError(
                "GCRS position vector has wrong shape: ",
                pos.shape,
                " instead of ",
                vdim.shape,
            )

        return pos * u.km

    def posvel_gcrs(self, t, group, ephem=None):
        """Return spacecraft GCRS position and velocity; this assumes position flags in tim file are in km and velocity flags are in km/s"""

        if group is None:
            raise ValueError("TOA group table needed for SpacecraftObs posvel_gcrs")

        try:
            vx = np.array([float(flags["vx"]) for flags in group["flags"]])
            vy = np.array([float(flags["vy"]) for flags in group["flags"]])
            vz = np.array([float(flags["vz"]) for flags in group["flags"]])
        except:
            log.error(
                "Missing flag. TOA line should have vx,vy,vz flags for GCRS velocity in km/s."
            )
            raise ValueError(
                "Missing flag. TOA line should have vx,vy,vz flags for GCRS velocity in km/s."
            )

        vel_geo = np.vstack((vx, vy, vz)) * (u.km / u.s)
        vdim = (3,) + t.shape
        if vel_geo.shape != vdim:
            raise ValueError(
                "GCRS velocity vector has wrong shape: ",
                vel_geo.shape,
                " instead of ",
                vdim.shape,
            )

        pos_geo = self.get_gcrs(t, group, ephem=None)

        return PosVel(pos_geo, vel_geo, origin="earth", obj="spacecraft")

    def posvel(self, t, ephem, group=None):
        if group is None:
            raise ValueError("TOA group table needed for SpacecraftObs posvel")

        # Compute vector from SSB to Earth
        geo_posvel = objPosVel_wrt_SSB("earth", t, ephem)

        # Spacecraft posvel w.r.t. Earth
        stl_posvel = self.posvel_gcrs(t, group)

        # Vector add to geo_posvel to get full posvel vector w.r.t. SSB.
        return geo_posvel + stl_posvel


def load_special_locations():
    """Load Barycenter, Geocenter, and other special locations into observatory registry.

    Loads :class:`~pint.observatory.special_locations.BarycenterObs`, :class:`~pint.observatory.special_locations.GeocenterObs`,
    and :class:`~pint.observatory.special_locations.T2SpacecraftObs` into observatory registry.
    """
    # Need to initialize one of each so that it gets added to the list
    BarycenterObs(
        "barycenter",
        aliases=["@", "ssb", "bary", "bat"],
        overwrite=True,
    )
    # Geocentric observatory where time is assume to be UTC with no corrections needed
    GeocenterObs(
        "geocenter",
        aliases=["0", "o", "coe", "geo", "geo_nogps"],
        apply_gps2utc=False,
        overwrite=True,
    )
    # Geocentric observatory where GPS->UTC corrections are applied
    GeocenterObs(
        "geocenter_gps",
        aliases=["geo_gps", "coe_gps"],
        apply_gps2utc=True,
        overwrite=True,
        tempo_code_str=None,
        tempo2_code_str=None,
    )
    T2SpacecraftObs(
        "stl_geo",
        aliases=["STL_GEO", "spacecraft"],
        apply_gps2utc=False,
        overwrite=True,
    )
    # TODO -- How to handle user changing bipm_version?


# run this on import
load_special_locations()
