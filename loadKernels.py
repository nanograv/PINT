import spice
import os

# Load the PINT environment variable to get the top level directory
pintdir = os.getenv("PINT")
if pintdir is None:
    print "Could not load the PINT environment variable pointing to the PINT directory!"
    os.sys.exit()
    
try:
    spice.furnsh(os.path.join(pintdir, "datafiles/pck00010.tpc"))
    print "SPICE loaded planetary constants."
    spice.furnsh(os.path.join(pintdir, "datafiles/naif0010.tls"))
    print "SPICE loaded leap seconds."
    spice.furnsh(os.path.join(pintdir, "datafiles/earth_latest_high_prec.bpc"))
    print "SPICE loaded Earth rotation parameters."
    spice.furnsh(os.path.join(pintdir, "datafiles/de405.bsp"))
    print "SPICE loaded DE405 Planetary Ephemeris."
except:
    print "Could not load SPICE kernel!"
