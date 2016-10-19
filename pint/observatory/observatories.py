# PINT observatories.py

# This file contains the basic definitions of observatory sites for 
# PINT.

from pint.observatory.topo_obs import TopoObs

TopoObs('gbt',          tempo_code='1', itoa_code='GB',
        itrf_xyz=[882589.65, -4924872.32, 3943729.348])
TopoObs('arecibo',      tempo_code='3', itoa_code='AO',
        itrf_xyz=[2390490.0, -5564764.0, 1994727.0])
TopoObs('vla',          tempo_code='6', itoa_code='VL', aliases=['jvla'],
        itrf_xyz=[-1601192.0, -5041981.4, 3554871.4])
TopoObs('parkes',       tempo_code='7', itoa_code='PK', aliases=['pks'],
        itrf_xyz=[-4554231.5, 2816759.1, -3454036.3])
TopoObs('nancay',       tempo_code='f', itoa_code='NC', aliases=['ncy'],
        itrf_xyz=[4324165.81, 165927.11, 4670132.83])
TopoObs('effelsberg',   tempo_code='g', itoa_code='EF', aliases=['eff'],
        itrf_xyz=[4033949.5, 486989.4, 4900430.8])
TopoObs('wsrt',         tempo_code='i', itoa_code='WB',
        itrf_xyz=[3828445.659, 445223.600, 5064921.5677])
TopoObs('mwa',          tempo_code='u', itoa_code='MW',
        itrf_xyz=[-2559454.08, 5095372.14, -2849057.18])
TopoObs('lwa1',         tempo_code='x', itoa_code='LW',
        itrf_xyz=[-1602196.60, -5042313.47, 3553971.51])
TopoObs('ps1',          tempo_code='p', itoa_code='PS',
        itrf_xyz=[-5461997.8, -2412559.0, 2243024.0])
