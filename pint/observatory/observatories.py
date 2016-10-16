# PINT observatories.py

# This file contains the basic definitions of observatory sites for 
# PINT.

from pint.observatory.topo_obs import TopoObs

TopoObs('gbt', tempo_code='1', itoa_code='GB',
        itrf_xyz=[882589.65, -4924872.32, 3943729.348])
TopoObs('arecibo', tempo_code='3', itoa_code='ao',
        itrf_xyz=[2390490.0, -5564764.0,  1994727.0])
TopoObs('parkes', tempo_code='7', itoa_code='pk', aliases=['pks',],
        itrf_xyz=[-4554231.5, 2816759.1, -3454036.3])
