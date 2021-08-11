import urllib.request
import pint.solar_system_ephemerides

#pint.solar_system_ephemerides.load_kernel("de440")

urllib.request.urlopen("https://data.nanograv.org/static/data/ephem/de440.bsp").read()
