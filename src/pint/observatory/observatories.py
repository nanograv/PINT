"""Definitions for standard observatories.

These observatories are registered when this file is imported. As a result it
cannot be imported until TopoObs has successfully been imported.

"""
from __future__ import absolute_import, division, print_function

from pint.observatory.topo_obs import TopoObs

TopoObs(
    "gbt",
    tempo_code="1",
    itoa_code="GB",
    itrf_xyz=[882589.65, -4924872.32, 3943729.348],
)
TopoObs(
    "arecibo",
    tempo_code="3",
    itoa_code="AO",
    aliases=["aoutc"],
    itrf_xyz=[2390490.0, -5564764.0, 1994727.0],
)
TopoObs(
    "vla",
    tempo_code="6",
    itoa_code="VL",
    aliases=["jvla"],
    itrf_xyz=[-1601192.0, -5041981.4, 3554871.4],
)
TopoObs(
    "parkes",
    tempo_code="7",
    itoa_code="PK",
    aliases=["pks"],
    itrf_xyz=[-4554231.5, 2816759.1, -3454036.3],
)
TopoObs(
    "jodrell",
    tempo_code="8",
    itoa_code="JB",
    aliases=["jbdfb", "jbroach", "jbafb"],
    itrf_xyz=[3822626.04, -154105.65, 5086486.04],
)
TopoObs(
    "nancay",
    tempo_code="f",
    itoa_code="NC",
    aliases=["ncy"],
    itrf_xyz=[4324165.81, 165927.11, 4670132.83],
)
TopoObs(
    "ncyobs",
    aliases=["ncyobs"],
    itrf_xyz=[4324165.81, 165927.11, 4670132.83],
    clock_fmt="tempo2",
    clock_file=["ncyobs2obspm.clk", "obspm2gps.clk"],
    clock_dir="TEMPO2",
)
TopoObs(
    "effelsberg",
    tempo_code="g",
    itoa_code="EF",
    aliases=["eff"],
    itrf_xyz=[4033949.5, 486989.4, 4900430.8],
)
TopoObs(
    "gmrt",
    tempo_code="r",
    itoa_code="GM",
    clock_fmt="tempo2",
    clock_file="gmrt2gps.clk",
    clock_dir="TEMPO2",
    itrf_xyz=[1656342.30, 5797947.77, 2073243.16],
)
TopoObs(
    "wsrt",
    tempo_code="i",
    itoa_code="WB",
    clock_fmt="tempo2",
    clock_dir="TEMPO2",
    clock_file="wsrt2gps.clk",
    itrf_xyz=[3828445.659, 445223.600, 5064921.5677],
)
TopoObs(
    "fast", tempo_code="k", itoa_code="FA", itrf_xyz=[-1668557.0, 5506838.0, 2744934.0]
)
TopoObs(
    "mwa",
    tempo_code="u",
    itoa_code="MW",
    itrf_xyz=[-2559454.08, 5095372.14, -2849057.18],
)
TopoObs(
    "lwa1",
    tempo_code="x",
    itoa_code="LW",
    itrf_xyz=[-1602196.60, -5042313.47, 3553971.51],
)
TopoObs(
    "ps1", tempo_code="p", itoa_code="PS", itrf_xyz=[-5461997.8, -2412559.0, 2243024.0]
)
TopoObs(
    "hobart",
    tempo_code="4",
    itoa_code="HO",
    itrf_xyz=[-3950077.96, 2522377.31, -4311667.52],
)
TopoObs(
    "most",
    tempo_code="e",
    itoa_code="MO",
    itrf_xyz=[-4483311.64, 2648815.92, -3671909.31],
    clock_fmt="tempo2",
    clock_dir="TEMPO2",
    clock_file="mo2gps.clk",
)
TopoObs(
    "chime",
    tempo_code="y",
    itoa_code="CH",
    itrf_xyz=[-2058795.0, -3621559.0, 4814280.0],
)

# ground-based gravitational-wave observatories
TopoObs(
    "virgo",
    aliases=["v1"],
    include_bipm=False,
    itrf_xyz=[4546374.0990, 842989.6976, 4378576.9624],
)
TopoObs(
    "lho",
    aliases=["h1", "hanford"],
    include_bipm=False,
    itrf_xyz=[-2161414.9264, -3834695.1789, 4600350.2266],
)
TopoObs(
    "llo",
    aliases=["l1", "livingston"],
    include_bipm=False,
    itrf_xyz=[-74276.0447, -5496283.7197, 3224257.0174],
)
TopoObs(
    "geo600",
    aliases=["geohf", "g1"],
    include_bipm=False,
    itrf_xyz=[3856309.9493, 666598.9563, 5019641.4172],
)
TopoObs(
    "kagra",
    aliases=["k1", "lcgt"],
    include_bipm=False,
    itrf_xyz=[-3777336.0240, 3484898.411, 3765313.6970],
)
TopoObs(
    "algonquin",
    itoa_code="AR",
    aliases=["aro", "ARO"],
    itrf_xyz=[918091.6472072796, -4346129.702203057, 4562012.861165226],
)
TopoObs(
    "drao",
    itoa_code="DR",
    aliases=["drao", "DRAO"],
    itrf_xyz=[-2058897.5725006417, -3621371.264826613, 4814353.577678314],
)
TopoObs(
    "acre",
    aliases=["acreroad", "a", "AR"],
    itrf_xyz=[3573741.1, -269156.74, 5258407.3],
)
