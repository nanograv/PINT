import spice
import observatories as obs
import sys
import astropy as ap
from spiceTookit import *
import pyximport
import toa
pyximport.install()

obss,obscode1s,obscode2s = obs.read_observatories()

rec = obss['ARECIBO']['xyz']

lon,lonD,lat,latD,alt = ITRF2GEO(rec)

print lonD,latD  

t = toa.TOAs("tempo2Test/J0000+0000.tim")

t.summary()


 
