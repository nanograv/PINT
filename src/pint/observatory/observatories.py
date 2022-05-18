"""Definitions for standard observatories.

These observatories are registered when this file is imported. As a result it
cannot be imported until TopoObs has successfully been imported.
"""
from pint.observatory.topo_obs import TopoObs

TopoObs(
    "gbt",
    tempo_code="1",
    itoa_code="GB",
    clock_file="time_gbt.dat",
    itrf_xyz=[882589.289, -4924872.368, 3943729.418],
    origin="""The Robert C. Byrd Green Bank Telescope.

    This data was obtained by Joe Swiggum from Ryan Lynch in 2021 September.
    """,
)
TopoObs(
    "gbt_pre_2021",
    clock_file="time_gbt.dat",
    itrf_xyz=[882589.65, -4924872.32, 3943729.348],
    origin="""The Robert C. Byrd Green Bank Telescope.

    The origin of this data is unknown but as of 2021 June 8 it agrees exactly with
    the values used by TEMPO and TEMPO2.
    """,
)
TopoObs(
    "arecibo",
    tempo_code="3",
    itoa_code="AO",
    clock_file="time_ao.dat",
    aliases=["aoutc"],
    itrf_xyz=[2390487.080, -5564731.357, 1994720.633],
    origin="""The Arecibo telescope.

    These are the coordinates used for VLBI as of March 2020 (MJD 58919). They are based on
    a fiducial position at MJD 52275 plus a (continental) drift velocity of
    [0.0099, 0.0045, 0.0101] m/yr. This data was obtained from Ben Perera in September 2021.
    """,
)
TopoObs(
    "arecibo_pre_2021",
    itrf_xyz=[2390490.0, -5564764.0, 1994727.0],
    clock_file="time_ao.dat",
    origin="""The Arecibo telescope.

    The origin of this data is unknown but as of 2021 June 8 it agrees exactly with
    the values used by TEMPO and TEMPO2. It is preserved to facilitate comparisons with
    the more modern position measurement.
    """,
)
TopoObs(
    "vla",
    tempo_code="6",
    itoa_code="VL",
    clock_file="time_vla.dat",
    aliases=["jvla"],
    itrf_xyz=[-1601192.0, -5041981.4, 3554871.4],
    origin="""The Jansky Very Large Array.

    The origin of this data is unknown but as of 2021 June 8 it agrees exactly with
    the values used by TEMPO and TEMPO2.
    """,
)
TopoObs(
    "meerkat",
    tempo_code="m",
    itoa_code="MK",
    clock_fmt="tempo2",
    clock_file="mk2utc.clk",
    clock_dir="TEMPO2",
    itrf_xyz=[5109360.133, 2006852.586, -3238948.127],
    origin="""MEERKAT, used in timing mode.

    The origin of this data is unknown but as of 2021 June 8 it agrees exactly with
    the values used by TEMPO and TEMPO2.
    """,
)
TopoObs(
    "parkes",
    tempo_code="7",
    itoa_code="PK",
    aliases=["pks"],
    clock_fmt="tempo2",
    clock_file="pks2gps.clk",
    clock_dir="TEMPO2",
    itrf_xyz=[-4554231.5, 2816759.1, -3454036.3],
    origin="""The Parkes radio telescope.

    The origin of this data is unknown but as of 2021 June 8 it agrees exactly with
    the values used by TEMPO and TEMPO2.
    """,
)
TopoObs(
    "jodrell",
    tempo_code="8",
    itoa_code="JB",
    clock_file="time_jb.dat",
    aliases=["jbdfb", "jbroach", "jbafb", "jbodfb", "jboafb", "jboroach"],
    bogus_last_correction=True,
    itrf_xyz=[3822625.769, -154105.255, 5086486.256],
    origin="""The Lovell telescope at Jodrell Bank.

    These are the coordinates used for VLBI as of March 2020 (MJD 58919). They are based on
    a fiducial position at MJD 50449 plus a (continental) drift velocity of
    [-0.0117, 0.0170, 0.0093] m/yr. This data was obtained from Ben Perera in September 2021.
    """,
)
TopoObs(
    "jodrell_pre_2021",
    clock_file="time_jb.dat",
    itrf_xyz=[3822626.04, -154105.65, 5086486.04],
    bogus_last_correction=True,
    origin="""The Lovell telescope at Jodrell Bank.

    The origin of this data is unknown but as of 2021 June 8 it agrees exactly with
    the values used by TEMPO and TEMPO2.
    """,
)
TopoObs(
    "nancay",
    tempo_code="f",
    itoa_code="NC",
    aliases=["ncy"],
    clock_fmt="tempo2",
    clock_file=[],
    itrf_xyz=[4324165.81, 165927.11, 4670132.83],
    origin="""The Nançay radio telescope.

    The origin of this data is unknown but as of 2021 June 8 it agrees exactly with
    the values used by TEMPO and TEMPO2.

    This telescope appears to require zero clock corrections to GPS.
    """,
)
TopoObs(
    "ncyobs",
    aliases=["nuppi", "w"],
    itrf_xyz=[4324165.81, 165927.11, 4670132.83],
    clock_fmt="tempo2",
    clock_file=["ncyobs2obspm.clk", "obspm2gps.clk"],
    clock_dir="TEMPO2",
    origin="""The Nançay radio telescope with the NUPPI back-end.

    The origin of this data is unknown but as of 2021 June 8 it agrees exactly with
    the values used by TEMPO and TEMPO2.
    """,
)
TopoObs(
    "effelsberg",
    tempo_code="g",
    itoa_code="EF",
    aliases=["eff"],
    clock_fmt="tempo2",
    clock_file="eff2gps.clk",
    clock_dir="TEMPO2",
    itrf_xyz=[4033947.146, 486990.898, 4900431.067],
    bogus_last_correction=True,
    origin="""The Effelsberg radio telescope.

    These are the coordinates used for VLBI as of March 2020 (MJD 58919). They are based on
    a fiducial position at MJD 56658 plus a (continental) drift velocity of
    [-0.0144, 0.0167, 0.0106] m/yr. This data was obtained from Ben Perera in September 2021.
    """,
)
TopoObs(
    "effelsberg_pre_2021",
    clock_fmt="tempo2",
    clock_file="eff2gps.clk",
    itrf_xyz=[4033949.5, 486989.4, 4900430.8],
    bogus_last_correction=True,
    origin="""The Effelsberg radio telescope.

    The origin of this data is unknown but as of 2021 June 8 it agrees exactly with
    the values used by TEMPO and TEMPO2.
    """,
)
TopoObs(
    "gmrt",
    tempo_code="r",
    itoa_code="GM",
    clock_fmt="tempo2",
    clock_file="gmrt2gps.clk",
    itrf_xyz=[1656342.30, 5797947.77, 2073243.16],
    origin="""The Giant Metrewave Radio Telescope.

    The origin of this data is unknown but as of 2021 June 8 it agrees exactly with
    the values used by TEMPO and TEMPO2.
    """,
)
TopoObs(
    "wsrt",
    aliases=["we"],
    tempo_code="i",
    itoa_code="WS",
    clock_fmt="tempo2",
    clock_file="wsrt2gps.clk",
    clock_dir="TEMPO2",
    itrf_xyz=[3828445.659, 445223.600, 5064921.5677],
    origin="""The Westerbork Synthesis Radio Telescope.

    Note that different letters have been used in the past to indicate this telescope.

    The origin of this data is unknown but as of 2021 June 8 it agrees exactly with
    the values used by TEMPO and TEMPO2.
    """,
)
TopoObs(
    "fast",
    tempo_code="k",
    itoa_code="FA",
    clock_file="time_fast.dat",
    itrf_xyz=[-1668557.0, 5506838.0, 2744934.0],
    origin="""The FAST radio telescope in China.

    Origin of this data is unknown but as of 2021 June 8 it agrees exactly with the
    TEMPO value and disagrees by about 17 km with the TEMPO2 value.
    """,
)
TopoObs(
    "mwa",
    tempo_code="u",
    itoa_code="MW",
    itrf_xyz=[-2559454.08, 5095372.14, -2849057.18],
    origin="""The Murchison Widefield Array.

    Origin of this data is unknown but as of 2021 June 8 this value agrees exactly with
    the value used by TEMPO2 and TEMPO.
    """,
)
TopoObs(
    "lwa1",
    tempo_code="x",
    itoa_code="LW",
    itrf_xyz=[-1602196.60, -5042313.47, 3553971.51],
    origin="""The LWA (long wavelength array, in New Mexico).

    Origin of this data is unknown but as of 2021 June 8 this value agrees exactly with
    the value used by TEMPO2 but disagrees with the value used by TEMPO by about 125 m.
    """,
)
TopoObs(
    "ps1",
    tempo_code="p",
    itoa_code="PS",
    itrf_xyz=[-5461997.8, -2412559.0, 2243024.0],
    origin="""Pan-STARRS.

    Origin of this data is unknown.
    """,
)
TopoObs(
    "hobart",
    tempo_code="4",
    itoa_code="HO",
    itrf_xyz=[-3950077.96, 2522377.31, -4311667.52],
    origin="""A telescope in Hobart, Tasmania.

    Origin of this data is unknown but as of 2021 June 8 this value agrees exactly with
    the value used by TEMPO2 and TEMPO.
    """,
)
TopoObs(
    "most",
    tempo_code="e",
    itoa_code="MO",
    itrf_xyz=[-4483311.64, 2648815.92, -3671909.31],
    clock_fmt="tempo2",
    clock_file="mo2gps.clk",
    clock_dir="TEMPO2",
    bogus_last_correction=True,
    origin="""The Molonglo Observatory Synthesis Telescope.

    Origin of this data is unknown but as of 2021 June 8 this value agrees exactly with
    the value used by TEMPO2.
    """,
)
TopoObs(
    "chime",
    tempo_code="y",
    itoa_code="CH",
    clock_file=[],
    itrf_xyz=[-2059166.313, -3621302.972, 4814304.113],
    origin="""The Canadian Hydrogen Intensity Mapping Experiment.

    Origin of these coordinates are from surveyor reports of the CHIME site
    (circa 2019 & 2020) and technical documents on the dimensions of the telescope
    structure (circa 2015). Results were compiled in January 2021. The coordinates
    are relative to the GRS80 ellipsoid.
    """,
)
TopoObs(
    "magic",
    aliases=["magic"],
    include_bipm=False,
    itrf_xyz=[5326878.7967, -1719509.5201, 3051884.5175],
    origin="""MAGIC (a ground-based gamma-ray telescope).

    Origin of this data is unknown.
    """,
)

TopoObs(
    "lst",
    aliases=["lst"],
    include_bipm=False,
    itrf_xyz=[5326832.7629, -1719636.1588, 3051795.1913],
)

# ground-based gravitational-wave observatories
TopoObs(
    "virgo",
    aliases=["v1"],
    include_bipm=False,
    itrf_xyz=[4546374.0990, 842989.6976, 4378576.9624],
    origin="""The VIRGO gravitational-wave observatory.

    Origin of this data is unknown but as of 2021 June 8 this value agrees exactly with
    the value used by TEMPO2.
    """,
)
TopoObs(
    "lho",
    aliases=["h1", "hanford"],
    include_bipm=False,
    itrf_xyz=[-2161414.9264, -3834695.1789, 4600350.2266],
    origin="""The LIGO Hanford gravitational-wave observatory.

    Origin of this data is unknown but as of 2021 June 8 this value agrees exactly with
    the value used by TEMPO2.
    """,
)
TopoObs(
    "llo",
    aliases=["l1", "livingston"],
    include_bipm=False,
    itrf_xyz=[-74276.0447, -5496283.7197, 3224257.0174],
    origin="""The LIGO Livingston gravitational-wave observatory.

    Origin of this data is unknown but as of 2021 June 8 this value agrees exactly with
    the value used by TEMPO2.
    """,
)
TopoObs(
    "geo600",
    aliases=["geohf"],  # is g1 used? It was here but TEMPO uses it for the GB 140ft
    include_bipm=False,
    itrf_xyz=[3856309.9493, 666598.9563, 5019641.4172],
    origin="""The GEO600 gravitational-wave observatory.

    Note that PINT used to list 'G1' as an alias for this telescope, but TEMPO accepts
    'G1' as an alias for the Green Bank 140-foot telescope, so it was removed here.

    Origin of this data is unknown but as of 2021 June 8 this value agrees exactly with
    the value used by TEMPO2.
    """,
)
TopoObs(
    "kagra",
    aliases=["k1", "lcgt"],
    include_bipm=False,
    itrf_xyz=[-3777336.0240, 3484898.411, 3765313.6970],
    origin="""The KAGRA gravitational-wave observatory.

    Origin of this data is unknown but as of 2021 June 8 this value agrees exactly with
    the value used by TEMPO2.
    """,
)


TopoObs(
    "algonquin",
    itoa_code="AR",
    aliases=["aro", "ARO"],
    itrf_xyz=[918091.6472072796, -4346129.702203057, 4562012.861165226],
    origin="""The Algonquin Radio Observatory.

    The origin of this data is unknown.
    """,
)
TopoObs(
    "drao",
    itoa_code="DR",
    aliases=["drao", "DRAO"],
    itrf_xyz=[-2058897.5725006417, -3621371.264826613, 4814353.577678314],
    origin="""The Dominion Radio Astronomical Observatory.

    The origin of this data is unknown.
    """,
)
TopoObs(
    "acre",
    aliases=["acreroad", "a", "AR"],
    itrf_xyz=[3573741.1, -269156.74, 5258407.3],
    origin="""The origin of this data is unknown.""",
)
TopoObs(
    "ata",
    aliases=["hcro"],
    itrf_xyz=[-2524263.18, -4123529.78, 4147966.36],
    origin="""The Allan telescope array.

    Origin of this data is unknown but as of 2021 June 8 this value agrees exactly with
    the value used by TEMPO2.
    """,
)
TopoObs(
    "ccera",
    itrf_xyz=[1093406.840, -4391945.819, 4479103.550],
    origin="""The origin of this data is unknown.""",
)

# Fake telescope for IPTA data challenge
TopoObs(
    "AXIS",
    aliases=["axi"],
    itrf_xyz=[6378138.00, 0.0, 0.0],
    origin="""Fake telescope for IPTA data challenge.

    Imported from TEMPO2 observatories.dat 2021 June 7.
    """,
)

# imported from tempo2 2021 June 7
TopoObs(
    name="narrabri",
    aliases=["atca"],
    itrf_xyz=[-4752329.7, 2790505.934, -3200483.747],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="nanshan",
    aliases=["ns"],
    itrf_xyz=[228310.702, 4631922.905, 4367064.059],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="uao",
    aliases=["ns"],
    itrf_xyz=[228310.702, 4631922.905, 4367064.059],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="dss_43",
    aliases=["tid43"],
    itrf_xyz=[-4460892.6, 2682358.9, -3674756.0],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="op",
    aliases=["obspm"],
    itrf_xyz=[4324165.81, 165927.11, 4670132.83],
    origin="""The Nançay radio telescope.

    Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="effelsberg_asterix",
    aliases=["effix"],
    itrf_xyz=[4033949.5, 486989.4, 4900430.8],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="leap",
    aliases=["leap"],
    itrf_xyz=[4033949.5, 486989.4, 4900430.8],
    origin="""The Large European Array for Pulsars.

    This is the same as the position of the Effelsberg radio telescope.

    Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="jodrellm4",
    aliases=["jbm4"],
    itrf_xyz=[3822252.643, -153995.683, 5086051.443],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="gb300",
    aliases=["gb300"],
    tempo_code="9",
    itoa_code="G3",
    itrf_xyz=[881856.58, -4925311.86, 3943459.7],
    origin="""The Green Bank 300-foot telescope.

    Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="gb140",
    aliases=["gb140"],
    itoa_code="G1",
    tempo_code="a",
    clock_file="time_gb140.dat",
    itrf_xyz=[882872.57, -4924552.73, 3944154.92],
    origin="""The Green Bank 140-foot telescope.

    Note that PINT used to accept 'G1' as an alias for the GEO600 gravitational-wave
    observatory but that conflicted with what TEMPO accepted for this telescope
    so that has been removed.

    Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="gb853",
    aliases=["gb853"],
    tempo_code="b",
    itoa_code="G8",
    clock_file="time_gb853.dat",
    itrf_xyz=[882315.33, -4925191.41, 3943414.05],
    bogus_last_correction=True,
    origin="""The Green Bank 85-3 telescope.

    Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="la_palma",
    aliases=["lap", "lapalma"],
    itrf_xyz=[5327021.651, -1719555.576, 3051967.932],
    origin="""La Palma observatory in the Canary Islands.

    Note that as of 2021 June 8 TEMPO2's position for this observatory lists
    it as somewhere in central Pakistan, exactly 90 degrees to the east of
    this position.
    """,
)
TopoObs(
    name="hartebeesthoek",
    aliases=["hart"],
    itrf_xyz=[5085442.78, 2668263.483, -2768697.034],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="warkworth_30m",
    aliases=["wark30m"],
    itrf_xyz=[-5115425.6, 477880.31, -3767042.81],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="warkworth_12m",
    aliases=["wark12m"],
    itrf_xyz=[-5115324.399, 477843.305, -3767192.886],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="lofar",
    aliases=["lofar"],
    tempo_code="t",
    itoa_code="LF",
    itrf_xyz=[3826577.462, 461022.624, 5064892.526],
    origin="""The Dutch low-frequency array LOFAR.

    Note that other TEMPO codes have been used for this telescope.

    Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de601lba",
    aliases=["eflfrlba"],
    itrf_xyz=[4034038.635, 487026.223, 4900280.057],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de601lbh",
    aliases=["eflfrlbh"],
    itrf_xyz=[4034038.635, 487026.223, 4900280.057],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de601hba",
    aliases=["eflfrhba"],
    itrf_xyz=[4034101.901, 487012.401, 4900230.21],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de601",
    aliases=["eflfr"],
    itrf_xyz=[4034101.901, 487012.401, 4900230.21],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de602lba",
    aliases=["uwlfrlba"],
    itrf_xyz=[4152561.068, 828868.725, 4754356.878],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de602lbh",
    aliases=["uwlfrlbh"],
    itrf_xyz=[4152561.068, 828868.725, 4754356.878],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de602hba",
    aliases=["uwlfrhba"],
    itrf_xyz=[4152568.416, 828788.802, 4754361.926],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de602",
    aliases=["uwlfr"],
    itrf_xyz=[4152568.416, 828788.802, 4754361.926],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de603lba",
    aliases=["tblfrlba"],
    itrf_xyz=[3940285.328, 816802.001, 4932392.757],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de603lbh",
    aliases=["tblfrlbh"],
    itrf_xyz=[3940285.328, 816802.001, 4932392.757],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de603hba",
    aliases=["tblfrhba"],
    itrf_xyz=[3940296.126, 816722.532, 4932394.152],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de603",
    aliases=["tblfr"],
    itrf_xyz=[3940296.126, 816722.532, 4932394.152],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de604lba",
    aliases=["polfrlba"],
    itrf_xyz=[3796327.609, 877591.315, 5032757.252],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de604lbh",
    aliases=["polfrlbh"],
    itrf_xyz=[3796327.609, 877591.315, 5032757.252],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de604hba",
    aliases=["polfrhba"],
    itrf_xyz=[3796380.254, 877613.809, 5032712.272],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de604",
    aliases=["polfr"],
    itrf_xyz=[3796380.254, 877613.809, 5032712.272],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de605lba",
    aliases=["julfrlba"],
    itrf_xyz=[4005681.742, 450968.282, 4926457.67],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de605lbh",
    aliases=["julfrlbh"],
    itrf_xyz=[4005681.742, 450968.282, 4926457.67],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de605hba",
    aliases=["julfrhba"],
    itrf_xyz=[4005681.407, 450968.304, 4926457.94],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de605",
    aliases=["julfr"],
    itrf_xyz=[4005681.407, 450968.304, 4926457.94],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="fr606lba",
    aliases=["frlfrlba"],
    itrf_xyz=[4323980.155, 165608.408, 4670302.803],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="fr606lbh",
    aliases=["frlfrlbh"],
    itrf_xyz=[4323980.155, 165608.408, 4670302.803],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="fr606hba",
    aliases=["frlfrhba"],
    itrf_xyz=[4324017.054, 165545.16, 4670271.072],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="fr606",
    aliases=["frlfr"],
    itrf_xyz=[4324017.054, 165545.16, 4670271.072],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="se607lba",
    aliases=["onlfrlba"],
    itrf_xyz=[3370287.366, 712053.586, 5349991.228],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="se607lbh",
    aliases=["onlfrlbh"],
    itrf_xyz=[3370287.366, 712053.586, 5349991.228],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="se607hba",
    aliases=["onlfrhba"],
    itrf_xyz=[3370272.092, 712125.596, 5349990.934],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="se607",
    aliases=["onlfr"],
    itrf_xyz=[3370272.092, 712125.596, 5349990.934],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="uk608lba",
    aliases=["uklfrlba"],
    itrf_xyz=[4008438.796, -100310.064, 4943735.554],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="uk608lbh",
    aliases=["uklfrlbh"],
    itrf_xyz=[4008438.796, -100310.064, 4943735.554],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="uk608hba",
    aliases=["uklfrhba"],
    itrf_xyz=[4008462.28, -100376.948, 4943716.6],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="uk608",
    aliases=["uklfr"],
    itrf_xyz=[4008462.28, -100376.948, 4943716.6],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de609lba",
    aliases=["ndlfrlba"],
    itrf_xyz=[3727207.778, 655184.9, 5117000.625],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de609lbh",
    aliases=["ndlfrlbh"],
    itrf_xyz=[3727207.778, 655184.9, 5117000.625],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de609hba",
    aliases=["ndlfrhba"],
    itrf_xyz=[3727218.128, 655108.821, 5117002.847],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="de609",
    aliases=["ndlfr"],
    itrf_xyz=[3727218.128, 655108.821, 5117002.847],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="fi609lba",
    aliases=["filfrlba"],
    itrf_xyz=[2136833.225, 810088.74, 5935285.279],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="fi609lbh",
    aliases=["filfrlbh"],
    itrf_xyz=[2136833.225, 810088.74, 5935285.279],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="fi609hba",
    aliases=["filfrhba"],
    itrf_xyz=[2136819.194, 810039.5757, 5935299.0536],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="fi609",
    aliases=["filfr"],
    itrf_xyz=[2136819.194, 810039.5757, 5935299.0536],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="utr-2",
    aliases=["utr2"],
    itrf_xyz=[3307865.236, 2487350.541, 4836939.784],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="goldstone",
    aliases=["gs"],
    itrf_xyz=[-2353621.22, -4641341.52, 3677052.352],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="shao",
    aliases=["shao"],
    tempo_code="s",
    itoa_code="SH",
    itrf_xyz=[-2826711.951, 4679231.627, 3274665.675],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="pico_veleta",
    aliases=["pv"],
    tempo_code="v",
    itoa_code="PV",
    itrf_xyz=[5088964.0, 301689.8, 3825017.0],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="iar1",
    aliases=["iar1"],
    itrf_xyz=[2765357.08, -4449628.98, -3625726.47],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="iar2",
    aliases=["iar2"],
    itrf_xyz=[2765322.49, -4449569.52, -3625825.14],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="kat-7",
    aliases=["k7"],
    itrf_xyz=[5109943.105, 2003650.7359, -3239908.3195],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="mkiii",
    aliases=["jbmk3"],
    itrf_xyz=[383395.727, -173759.585, 5077751.313],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="tabley",
    aliases=["tabley"],
    itrf_xyz=[3817176.557, -162921.17, 5089462.046],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="darnhall",
    aliases=["darnhall"],
    itrf_xyz=[3828714.504, -169458.987, 5080647.749],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="knockin",
    aliases=["knockin"],
    itrf_xyz=[3859711.492, -201995.082, 5056134.285],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="defford",
    aliases=["defford"],
    itrf_xyz=[3923069.135, -146804.404, 5009320.57],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="cambridge",
    aliases=["cam"],
    itrf_xyz=[3919982.752, 2651.982, 5013849.826],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="princeton",
    aliases=["princeton"],
    tempo_code="5",
    itoa_code="PR",
    itrf_xyz=[1288748.38, -4694221.77, 4107418.8],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="hamburg",
    aliases=["hamburg"],
    itrf_xyz=[3788815.62, 1131748.336, 5035101.19],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="jb_42ft",
    aliases=["jb42"],
    itrf_xyz=[3822294.825, -153862.275, 5085987.071],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="jb_mkii",
    aliases=["jbmk2"],
    tempo_code="h",
    itoa_code="J2",
    itrf_xyz=[3822846.76, -153802.28, 5086285.9],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="jb_mkii_rch",
    aliases=["jbmk2roach"],
    itrf_xyz=[3822846.76, -153802.28, 5086285.9],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="jb_mkii_dfb",
    aliases=["jbmk2dfb"],
    itrf_xyz=[3822846.76, -153802.28, 5086285.9],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="lwa_sv",
    aliases=["lwasv"],
    itoa_code="LS",
    itrf_xyz=[-1531155.54418, -5045324.30517, 3579583.8945],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="grao",
    aliases=["grao"],
    itrf_xyz=[6346273.531, -33779.7127, 634844.9454],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)
TopoObs(
    name="srt",
    aliases=["srt"],
    tempo_code="z",
    itoa_code="SR",
    itrf_xyz=[4865182.766, 791922.689, 4035137.174],
    origin="""Imported from TEMPO2 observatories.dat 2021 June 7.""",
)

# From Tempo 2021 June 8
TopoObs(
    name="quabbin",
    tempo_code="2",
    itoa_code="QU",
    itrf_xyz=[1430913.3496148302, -4495711.383965823, 4278113.974517222],
    origin="""Imported from TEMPO obsys.dat 2021 June 8.""",
)
TopoObs(
    name="vla_site",
    tempo_code="c",
    itoa_code="V2",
    itrf_xyz=[-1601135.5133304405, -5042005.480977412, 3554875.076856462],
    origin="""Imported from TEMPO obsys.dat 2021 June 8.""",
)
TopoObs(
    name="gb_20m_xyz",
    itoa_code="G2",
    itrf_xyz=[883772.7974, -4924385.5975, 3944042.4991],
    origin="""Imported from TEMPO obsys.dat 2021 June 8.""",
)
TopoObs(
    name="northern_cross",
    tempo_code="d",
    itoa_code="BO",
    itrf_xyz=[4461242.882451464, 919559.8351226494, 4449633.220012489],
    origin="""Imported from TEMPO obsys.dat 2021 June 8.""",
)
