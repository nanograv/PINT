import urllib.request
import pint.solar_system_ephemerides


urllib.request.urlopen("https://data.nanograv.org/static/data/ephem/de440.bsp").read()

pint.solar_system_ephemerides.load_kernel("de440")
