import os
import unittest
import pytest
from io import StringIO

from pint.models import get_model, get_model_and_toas
from pint import fitter
from pinttestdata import datadir


def test_SWM():
    """Should be present in PINT, not in TEMPO/TEMPO2"""

    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    assert (
        ("SWM" in m.as_parfile())
        and not ("SWM" in m.as_parfile(format="tempo"))
        and not ("SWM" in m.as_parfile(format="tempo2"))
    )


def test_CHI2():
    """Should be present after fit in PINT, not in TEMPO/TEMPO2"""
    m, t = get_model_and_toas(
        os.path.join(datadir, "NGC6440E.par"), os.path.join(datadir, "NGC6440E.tim")
    )

    f = fitter.WLSFitter(toas=t, model=m)
    assert "CHI2" in f.model.as_parfile()
    assert not ("CHI2" in f.model.as_parfile(format="tempo2"))
    assert not ("CHI2" in f.model.as_parfile(format="tempo"))


def test_T2CMETHOD():
    """Should be commented out in TEMPO2"""
    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_TAI.par"))
    for l in m.as_parfile().split("\n"):
        if "T2CMETHOD" in l:
            assert not (l.startswith("#"))
    for l in m.as_parfile(format="tempo").split("\n"):
        if "T2CMETHOD" in l:
            assert not (l.startswith("#"))
    for l in m.as_parfile(format="tempo2").split("\n"):
        if "T2CMETHOD" in l:
            assert l.startswith("#")


def test_MODE1():
    """Should start TEMPO2"""
    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_dfg+12_TAI.par"))
    assert (
        not (m.as_parfile(include_info=False).startswith("MODE 1"))
        and not (m.as_parfile(include_info=False, format="tempo").startswith("MODE 1"))
        and (m.as_parfile(include_info=False, format="tempo2").startswith("MODE 1"))
    )


def test_STIGMA():
    """Should get changed to VARSIGMA for TEMPO/TEMPO2"""
    m = get_model(os.path.join(datadir, "J0613-0200_NANOGrav_9yv1_ELL1H_STIG.gls.par"))
    assert (
        ("STIGMA" in m.as_parfile())
        and not ("VARSIGMA" in m.as_parfile())
        and not ("STIGMA" in m.as_parfile(format="tempo"))
        and ("VARSIGMA" in m.as_parfile(format="tempo"))
        and not ("STIGMA" in m.as_parfile(format="tempo2"))
        and ("VARSIGMA" in m.as_parfile(format="tempo2"))
    )


def test_A1DOT():
    """Should get changed to XDOT for TEMPO/TEMPO2"""
    m = get_model(os.path.join(datadir, "J1600-3053_test.par"))
    assert (
        ("A1DOT" in m.as_parfile())
        and not ("XDOT" in m.as_parfile())
        and not ("A1DOT" in m.as_parfile(format="tempo"))
        and ("XDOT" in m.as_parfile(format="tempo"))
        and not ("A1DOT" in m.as_parfile(format="tempo2"))
        and ("XDOT" in m.as_parfile(format="tempo2"))
    )


def test_ECL():
    """Should be only IERS2003 for TEMPO2"""
    m = get_model(os.path.join(datadir, "J0613-0200_NANOGrav_9yv1.gls.par"))
    for l in m.as_parfile().split("\n"):
        if "ECL" in l:
            assert l.split()[-1] == "IERS2010"
    for l in m.as_parfile(format="tempo").split("\n"):
        if "ECL" in l:
            assert l.split()[-1] == "IERS2010"
    for l in m.as_parfile(format="tempo2").split("\n"):
        if "ECL" in l:
            assert l.split()[-1] == "IERS2003"


def test_formats():
    m = get_model(os.path.join(datadir, "B1855+09_NANOGrav_9yv1.gls.par"))
    with pytest.raises(AssertionError):
        s = m.as_parfile(format="nottempo")


def test_EFAC():
    """Should become T2EFAC in TEMPO/TEMPO2"""
    model = get_model(
        StringIO(
            """
        PSRJ J1234+5678
        ELAT 0
        ELONG 0
        DM 10
        F0 1
        PEPOCH 58000
        EFAC mjd 57000 58000 2
        """
        )
    )
    assert any(
        l.startswith("EFAC") for l in model.as_parfile(format="pint").split("\n")
    )
    assert not any(
        l.startswith("T2EFAC") for l in model.as_parfile(format="pint").split("\n")
    )
    assert not any(
        l.startswith("EFAC") for l in model.as_parfile(format="tempo").split("\n")
    )
    assert any(
        l.startswith("T2EFAC") for l in model.as_parfile(format="tempo").split("\n")
    )
    assert not any(
        l.startswith("EFAC") for l in model.as_parfile(format="tempo2").split("\n")
    )
    assert any(
        l.startswith("T2EFAC") for l in model.as_parfile(format="tempo2").split("\n")
    )


def test_EQUAD():
    """Should become T2EQUAD in TEMPO/TEMPO2"""
    model = get_model(
        StringIO(
            """
        PSRJ J1234+5678
        ELAT 0
        ELONG 0
        DM 10
        F0 1
        PEPOCH 58000
        EQUAD mjd 57000 58000 2
        """
        )
    )
    assert any(
        l.startswith("EQUAD") for l in model.as_parfile(format="pint").split("\n")
    )
    assert not any(
        l.startswith("T2EQUAD") for l in model.as_parfile(format="pint").split("\n")
    )
    assert not any(
        l.startswith("EQUAD") for l in model.as_parfile(format="tempo").split("\n")
    )
    assert any(
        l.startswith("T2EQUAD") for l in model.as_parfile(format="tempo").split("\n")
    )
    assert not any(
        l.startswith("EQUAD") for l in model.as_parfile(format="tempo2").split("\n")
    )
    assert any(
        l.startswith("T2EQUAD") for l in model.as_parfile(format="tempo2").split("\n")
    )


def test_NHARMS():
    model = get_model(
        StringIO(
            """
        # Created: 2021-09-28T21:53:53.494561
        # PINT_version: 0.8.3
        # User: jpg00017
        # Host: tmmcm107.hpc.wvu.edu
        # OS: Linux-3.10.0-1160.24.1.el7.x86_64-x86_64-with-glibc2.31
        PSR                            J1802-2124
        EPHEM                               DE440
        CLOCK                        TT(BIPM2019)
        UNITS                                 TDB
        START              57682.9164254267206481
        FINISH             58945.5403619002087731
        DILATEFREQ                              N
        DMDATA                                  N
        NTOA                                 7486
        CHI2                    7423.861574964089
        ELONG                 270.486526147922007 1 0.00000012082526951556
        ELAT                    2.037373400788321 1 0.00000314425302204301
        PMELONG                -1.826189745727688 1 0.4777024218759009
        PMELAT                -2.9704295220972257 1 12.55798968405999
        PX                     0.5286751048529988 1 1.2501655070063251
        ECL                              IERS2010
        POSEPOCH           58314.0000000000000000
        F0                    79.0664240384491762 1 2.7971394286742147137e-12
        F1               -4.55647355155766584e-16 1 2.5152536793477603086e-19
        PEPOCH             58314.0000000000000000
        CORRECT_TROPOSPHERE                         Y
        PLANET_SHAPIRO                          Y
        NE_SW                                 0.0
        SWM                                   0.0
        DM                  149.60232712555309087
        DM1                                   0.0
        DMEPOCH            58314.0000000000000000
        DMX                                   6.5
        DMX_0001             0.004696399283540413 1 0.0006591603056674324
        DMXR1_0001         57682.8664254263858333
        DMXR2_0001         57682.9749333329746180
        DMX_0002            0.0064007440081210705 1 0.0006774981437564753
        DMXR1_0002         57711.7663691889974885
        DMXR2_0002         57712.8898653412397801
        DMX_0003             0.007089398087860698 1 0.0010017525706993418
        DMXR1_0003         57744.6970277523578241
        DMXR2_0003         57744.8062178231921295
        DMX_0004             0.006737039848836672 1 0.0006018872883701722
        DMXR1_0004         57745.6864473574096296
        DMXR2_0004         57745.7953691773727777
        DMX_0005              0.00515948561993118 1 0.0005307043359928936
        DMXR1_0005         57893.3399973677296643
        DMXR2_0005         57894.4603220452627315
        DMX_0006             0.004929972501758111 1 0.0005185515318964849
        DMXR1_0006         57921.2746599964448264
        DMXR2_0006         57922.3871902091268519
        DMX_0007            0.0041443696348809475 1 0.0005008840006981711
        DMXR1_0007         57948.2015643181002778
        DMXR2_0007         57949.3236326954429399
        DMX_0008             0.003315938198192618 1 0.0005434555806901042
        DMXR1_0008         57977.1229173716346875
        DMXR2_0008         57979.2569280433090162
        DMX_0009            0.0024685322594886353 1 0.0004902481688528785
        DMXR1_0009         58014.8917796535170024
        DMXR2_0009         58016.0100260919061806
        DMX_0010            0.0013893032271176504 1 0.000448931995429193
        DMXR1_0010         58046.8211808780397454
        DMXR2_0010         58047.8760652077253356
        DMX_0011            0.0005031529016224686 1 0.00046633178861378626
        DMXR1_0011         58059.7959430718838079
        DMXR2_0011         58061.0152292506318171
        DMX_0012          -0.00015713521598254693 1 0.0006802398142039652
        DMXR1_0012         58094.6402435449126620
        DMXR2_0012         58094.7510816852664004
        DMX_0013          -2.0063367167663543e-05 1 0.0005059270088450605
        DMXR1_0013         58096.7428006146001274
        DMXR2_0013         58096.8539294806978472
        DMX_0014            0.0002352100740037611 1 0.000553028154123409
        DMXR1_0014         58133.7123506960197107
        DMXR2_0014         58133.8221691385697337
        DMX_0015           -8.530311088855539e-06 1 0.00043672449390729477
        DMXR1_0015         58141.4160234026931019
        DMXR2_0015         58142.6046038065251158
        DMX_0016           0.00037543353072028824 1 0.00046086063577416015
        DMXR1_0016         58160.6393713456054629
        DMXR2_0016         58161.7518798974633680
        DMX_0017           0.00023262662430101254 1 0.0004715580596162597
        DMXR1_0017         58187.5657186571916087
        DMXR2_0017         58188.6788645828055438
        DMX_0018            0.0005625523180597451 1 0.0005392881749381467
        DMXR1_0018         58216.4827094133215741
        DMXR2_0018         58217.6027056441234839
        DMX_0019           0.00027422227123482493 1 0.000527820301337838
        DMXR1_0019         58243.3866794593777778
        DMXR2_0019         58244.5225383601573380
        DMX_0020            0.0007208923707761619 1 0.0005076928853046824
        DMXR1_0020         58270.1813525921679630
        DMXR2_0020         58271.2547486546172917
        DMX_0021             0.001318573597378471 1 0.0005081047578277821
        DMXR1_0021         58301.2550606145814583
        DMXR2_0021         58302.3649924346973379
        DMX_0022           -0.0005135016207960321 1 0.0004770577842430441
        DMXR1_0022         58335.1102705057055555
        DMXR2_0022         58336.2224940446831134
        DMX_0023            -0.000664557714349201 1 0.0004684596139782111
        DMXR1_0023         58365.0694685719063310
        DMXR2_0023         58366.1483667601496528
        DMX_0024           -0.0014006832537535323 1 0.0005061637896460964
        DMXR1_0024         58396.9601085353642708
        DMXR2_0024         58398.0679047488229167
        DMX_0025           -0.0002989858956384618 1 0.000532445514182861
        DMXR1_0025         58424.8908417172206133
        DMXR2_0025         58425.9839191379214700
        DMX_0026            0.0011712349190143515 1 0.000868687250487956
        DMXR1_0026         58460.7832550711609607
        DMXR2_0026         58460.8921725698310186
        DMX_0027            -0.000425269869132141 1 0.0006044372646145639
        DMXR1_0027         58461.7821153065815509
        DMXR2_0027         58461.8906924871262152
        DMX_0028             0.000999644989592473 1 0.0009619071881871589
        DMXR1_0028         58489.7495506076084029
        DMXR2_0028         58489.8581179093671413
        DMX_0029             0.002334649423130252 1 0.0007862466031956895
        DMXR1_0029         58490.6979050840018286
        DMXR2_0029         58490.7979052257360301
        DMX_0030            0.0004401583296719997 1 0.0006316184434413346
        DMXR1_0030         58516.6036853384139237
        DMXR2_0030         58517.7348568968154745
        DMX_0031            0.0007520690512245063 1 0.0007087802847992461
        DMXR1_0031         58547.5834219135431365
        DMXR2_0031         58548.6162168267546760
        DMX_0032            0.0017189131841064576 1 0.0006451881436054719
        DMXR1_0032         58583.4493006690172454
        DMXR2_0032         58583.5577925015022684
        DMX_0033            0.0002573929073544109 1 0.0005914967106594508
        DMXR1_0033         58608.3745397348684491
        DMXR2_0033         58609.4841592242737500
        DMX_0034           -0.0005124964110303959 1 0.000568361009259633
        DMXR1_0034         58635.3438664973728009
        DMXR2_0034         58636.4104592450821644
        DMX_0035          -0.00019868661272143246 1 0.0005688840275327434
        DMXR1_0035         58671.2081490472046412
        DMXR2_0035         58672.3172362898767477
        DMX_0036            -0.001361606479341988 1 0.0005872479276087107
        DMXR1_0036         58699.1278183178583333
        DMXR2_0036         58700.2389117932398264
        DMX_0037           -0.0021327560876241527 1 0.0005624435854948168
        DMXR1_0037         58729.0438607796168056
        DMXR2_0037         58730.1506955060758680
        DMX_0038            -0.002206848553168709 1 0.0005616838193877897
        DMXR1_0038         58758.9681207974062963
        DMXR2_0038         58760.0784131403197569
        DMX_0039           -0.0029356917519516254 1 0.0006093606697260566
        DMXR1_0039         58786.8969848349297569
        DMXR2_0039         58787.9943882955267361
        DMX_0040            -0.004809793196400854 1 0.0005524546142495452
        DMXR1_0040         58826.7926456005879977
        DMXR2_0040         58826.9012350118500810
        DMX_0041           -0.0058649823357842835 1 0.0007601992714649454
        DMXR1_0041         58827.7925842139375116
        DMXR2_0041         58827.9010326579737037
        DMX_0042            -0.006238520702200575 1 0.0006792748630395879
        DMXR1_0042         58848.7289558475458797
        DMXR2_0042         58848.8378161906221065
        DMX_0043            -0.005089644219158067 1 0.0005701061096263564
        DMXR1_0043         58849.7297562774547223
        DMXR2_0043         58849.8382628233084722
        DMX_0044            -0.006614500482843444 1 0.0004957270662099692
        DMXR1_0044         58881.6691483454196759
        DMXR2_0044         58882.7762844075041320
        DMX_0045            -0.007164958991610447 1 0.0005595610978859809
        DMXR1_0045         58910.5608714618210764
        DMXR2_0045         58911.6581970805564236
        DMX_0046            -0.006363336774121782 1 0.0006003505840151168
        DMXR1_0046         58945.4818026794673611
        DMXR2_0046         58945.5903618999000115
        BINARY ELL1H
        PB                  0.6988892433285496519 1 2.306950233730408576e-11
        A1                      3.718865580010016 1 5.10696441177697e-07
        TASC               58314.1068677136653058 1 1.02392354114256444625e-08
        EPS1             4.117234830314865184e-06 1 2.8610710701285065444e-07
        EPS2            1.9600104585898805536e-06 1 2.9783944064651076693e-07
        H3              2.4037642640660087724e-06 1 3.31127323135470378e-07
        H4              1.7027113002634860964e-06 1 2.88248121675800648e-07
        NHARMS                                  7.0
        FD1                2.2042642599772845e-05 1 1.2254082126867467e-06
        TZRMJD             58133.7623761140993056
        TZRSITE                                GB
        TZRFRQ                           1455.314
        JUMP            -fe Rcvr_800    -8.756145657132816e-05 1 8.10943477015656e-07
        EFAC            -f Rcvr1_2_GUPPI          1.11383940850175
        EQUAD           -f Rcvr1_2_GUPPI       0.48300872464422495
        EFAC            -f Rcvr_800_GUPPI         1.078261304920897
        EQUAD           -f Rcvr_800_GUPPI      0.005978124184979209
        ECORR           -f Rcvr1_2_GUPPI        0.6424434213096708
        ECORR           -f Rcvr_800_GUPPI         4.740888311524382
        RNAMP                  1.5164580721961418
        RNIDX                 -1.3814382392008375
        """
        )
    )
    for l in model.as_parfile(format="tempo").split("\n"):
        if l.startswith("NHARMS"):
            d = l.split()
            # it should be an integer
            assert not "." in d[-1]
