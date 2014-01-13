"""This module loads basic ephemeris data when imported."""
import spice
import os
from astropy import log
from pint import pintdir

spice.furnsh(os.path.join(pintdir, "datafiles/pck00010.tpc"))
log.info("SPICE loaded planetary constants.")
spice.furnsh(os.path.join(pintdir, "datafiles/naif0010.tls"))
log.info("SPICE loaded leap seconds.")
spice.furnsh(os.path.join(pintdir, "datafiles/earth_latest_high_prec.bpc"))
log.info("SPICE loaded Earth rotation parameters.")
spice.furnsh(os.path.join(pintdir, "datafiles/de405.bsp"))
log.info("SPICE loaded DE405 Planetary Ephemeris.")
