# PINT observatories.py

# This file contains the basic definitions of observatory sites for 
# PINT.

from pint.observatory.topo_obs import TopoObs

TopoObs('gbt', tempo_code='1', itoa_code='GB',
        itrf_xyz=[882589.65, -4924872.32, 3943729.348])
