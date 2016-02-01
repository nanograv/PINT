from pint.models import model_builder as mb
import pint.toa as toa
import matplotlib.pyplot as plt
from pint import residuals
import astropy.units as u

parf = 'J1923+2515_NANOGrav_9yv1.gls.par'
timf = 'J1923+2515_NANOGrav_9yv1.tim'
DMXmodel = mb.get_model(parf)
t = toa.get_TOAs(timf)

dispDelay = DMXmodel.dispersion_delay(t.table)
xt = t.get_mjds()
plt.figure(1)
plt.plot(xt, dispDelay,'x')
plt.title("%s DMX dispersion delay" % DMXmodel.PSR.value)
plt.xlabel('MJD')
plt.ylabel("Dispersion (s)")
plt.show()
plt.figure(2)
rs = residuals.resids(t, DMXmodel).time_resids.to(u.us).value
plt.plot(xt, rs, 'x')
plt.title("%s Pre-Fit Timing Residuals" % DMXmodel.PSR.value)
plt.xlabel('MJD')
plt.ylabel('Residual (us)')
plt.grid()
plt.show()
