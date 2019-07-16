











import numpy as np
import pint.models
import pint.toa
import pint.residuals
import matplotlib.pyplot as plt



m = pint.models.model_builder.get_model('Ter5al_start.par')
t = pint.toa.get_TOAs('Ter5al_start.tim')
rs = pint.residuals.resids(t, m).phase_resids
plt.plot(t.get_mjds(), rs, '.', label = 'prefit')
plt.legend()
plt.grid()
plt.show()

RAs = np.arange(17.400000000, 17.700000000, 0.00000005)
for RA in RAs:
    print(m.as_parfile())
    getattr(m, 'RAJ').value = RA
    print(m.as_parfile())
    rs = pint.residuals.resids(t, m).phase_resids
    plt.plot(t.get_mjds(), rs, '.', label=RA)
    plt.legend()
    plt.grid()
    plt.show()
