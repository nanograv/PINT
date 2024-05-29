import pint.solar_system_ephemerides
import pint.observatory

for e in [
    "de440",
    "de405",
    "de421",
    "de430",
    "de430t",
    "de434",
    "de436t",
    "de436",
    "de436t",
]:
    pint.solar_system_ephemerides.load_kernel(e)


pint.observatory.find_latest_bipm()
