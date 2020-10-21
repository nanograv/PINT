from io import StringIO

import astropy.units as u
import numpy as np
import pytest

from pint.toa import TOAs, get_TOAs

basic_tim_header = "FORMAT 1\n"

basic_tim = """
53358.000056.3.000.000.9y.x.ff 424.000000 53358.767912764015642   1.277  ao  -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 903.18 -tmplt B1855+09.430.PUPPI.9y.x.sum.sm -gof 1.09 -nbin 2048 -nch 1 -chan 1 -subint 0 -snr 142.71 -wt 15 -proc 9y -pta NANOGrav -to -0.789e-6
53532.000025.3.000.000.9y.x.ff 424.000000 53532.239347988112972   1.061  ao  -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 903.18 -tmplt B1855+09.430.PUPPI.9y.x.sum.sm -gof 1.09 -nbin 2048 -nch 1 -chan 1 -subint 0 -snr 166.51 -wt 15 -proc 9y -pta NANOGrav -to -0.789e-6
54862.000023.3.000.000.9y.x.ff 428.000000 54862.593840761061363   1.337  ao  -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 903.18 -tmplt B1855+09.430.PUPPI.9y.x.sum.sm -gof 0.921 -nbin 2048 -nch 1 -chan 2 -subint 1 -snr 131.03 -wt 15 -proc 9y -pta NANOGrav -to -0.789e-6
puppi_56411_1855+09_0030.9y.x.ff 422.187012 56411.396988092582945   3.027  ao  -fe 430 -be PUPPI -f 430_PUPPI -bw 1.562 -tobs 1534.1 -tmplt B1855+09.430.PUPPI.9y.x.sum.sm -gof 1.19 -nbin 2048 -nch 1 -chan 27 -subint 0 -snr 60.659 -wt 2397 -proc 9y -pta NANOGrav
53453.000000.3.000.000.9y.x.ff 1434.000000 53453.471481198627050   0.433  ao  -fe L-wide -be ASP -f L-wide_ASP -bw 4 -tobs 903.79 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.08 -nbin 2048 -nch 1 -chan 2 -subint 0 -snr 414.47 -wt 15 -proc 9y -pta NANOGrav -to -0.839e-6
53851.000031.3.000.000.9y.x.ff 1422.000000 53851.404233713620999   2.745  ao  -fe L-wide -be ASP -f L-wide_ASP -bw 4 -tobs 903.79 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1 -nbin 2048 -nch 1 -chan 5 -subint 1 -snr 67.191 -wt 15 -proc 9y -pta NANOGrav -to -0.839e-6
55298.000029.3.000.000.9y.x.ff 1406.000000 55298.405393153214034   0.701  ao  -fe L-wide -be ASP -f L-wide_ASP -bw 4 -tobs 903.79 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.18 -nbin 2048 -nch 1 -chan 9 -subint 0 -snr 257.6 -wt 15 -proc 9y -pta NANOGrav -to -0.839e-6
puppi_56122_1855+09_0527.9y.x.ff 1349.480957 56122.129856965925030   0.766  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1393.9 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.25 -nbin 2048 -nch 8 -chan 34 -subint 0 -snr 240.17 -wt 14221 -proc 9y -pta NANOGrav
puppi_56212_1855+09_1056.9y.x.ff 1511.510010 56212.923999620852331   0.679  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1527.2 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.04 -nbin 2048 -nch 8 -chan 21 -subint 2 -snr 266.29 -wt 14317 -proc 9y -pta NANOGrav
puppi_56299_1855+09_1682.9y.x.ff 1550.911011 56299.687959131351917   2.100  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1209.4 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.09 -nbin 2048 -nch 8 -chan 18 -subint 0 -snr 84.112 -wt 11114 -proc 9y -pta NANOGrav
puppi_56374_1855+09_0482.9y.x.ff 1224.500000 56374.458848469919109   1.865  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1229.9 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 0.966 -nbin 2048 -nch 8 -chan 44 -subint 1 -snr 96.158 -wt 15166 -proc 9y -pta NANOGrav
puppi_56498_1855+09_0477.9y.x.ff 1599.865967 56498.124360796719963   0.415  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1424.7 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.34 -nbin 2048 -nch 8 -chan 14 -subint 0 -snr 426.8 -wt 8311.7 -proc 9y -pta NANOGrav
puppi_56577_B1855+09_0547.9y.x.ff 1687.030029 56577.922801352174762   0.284  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1200.8 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 2.92 -nbin 2048 -nch 8 -chan 7 -subint 0 -snr 619.21 -wt 15011 -proc 9y -pta NANOGrav
puppi_56598_B1855+09_0089.9y.x.ff 1712.032959 56598.871995329746393   0.927  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1200.8 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.21 -nbin 2048 -nch 8 -chan 5 -subint 0 -snr 187.84 -wt 14990 -proc 9y -pta NANOGrav
puppi_56598_B1855+09_0089.9y.x.ff 1174.531006 56598.871995326598351   1.110  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1200.8 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.13 -nbin 2048 -nch 8 -chan 48 -subint 0 -snr 167.54 -wt 15011 -proc 9y -pta NANOGrav
puppi_56598_B1855+09_0089.9y.x.ff 1162.031006 56598.871995336593657   0.657  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1200.8 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.68 -nbin 2048 -nch 8 -chan 49 -subint 0 -snr 285.13 -wt 14746 -proc 9y -pta NANOGrav
puppi_56598_B1855+09_0089.9y.x.ff 1153.437012 56598.871995343722109   2.613  ao  -fe L-wide -be PUPPI -f L-wide_PUPPI -bw 12.5 -tobs 1200.8 -tmplt B1855+09.L-wide.PUPPI.9y.x.sum.sm -gof 1.22 -nbin 2048 -nch 8 -chan 50 -subint 0 -snr 72.635 -wt 5629 -proc 9y -pta NANOGrav
"""


def do_roundtrip(toas, format="tempo2"):
    f = StringIO()
    toas.write_TOA_file(f, format=format)
    toas_2 = get_TOAs(StringIO(f.getvalue()))
    assert toas.commands == toas_2.commands
    assert toas.ntoas == toas_2.ntoas
    assert np.all(
        toas.get_mjds(high_precision=True) == toas_2.get_mjds(high_precision=True)
    )
    assert np.all(toas.get_freqs() == toas_2.get_freqs())
    assert np.all(toas.get_errors() == toas_2.get_errors())
    assert np.all(toas.get_obss() == toas_2.get_obss())
    assert np.all(toas.get_pulse_numbers() == toas_2.get_pulse_numbers())
    assert toas.get_flags() == toas_2.get_flags()


def test_basic():
    f = StringIO(basic_tim_header + basic_tim)
    do_roundtrip(get_TOAs(f))
