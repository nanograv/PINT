import pytest
from pint import toa


def test_parse_toa_line_exceptions():
    # This should work.
    goodline = "55731.000019.3.000.000.9y.x.ff 428.000000 55731.193719931024413   2.959  ao  -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 1200.7 -tmplt J1853+1303.430.PUPPI.9y.x.sum.sm -gof 0.98 -nbin 2048 -nch 1 -chan 2 -subint 0 -snr 36.653 -wt 20 -proc 9y -pta NANOGrav -to -0.789e-6"
    toa._parse_TOA_line(goodline)

    # obs is given as a flag.
    badline = "55731.000019.3.000.000.9y.x.ff 428.000000 55731.193719931024413   2.959  ao -obs ao -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 1200.7 -tmplt J1853+1303.430.PUPPI.9y.x.sum.sm -gof 0.98 -nbin 2048 -nch 1 -chan 2 -subint 0 -snr 36.653 -wt 20 -proc 9y -pta NANOGrav -to -0.789e-6"
    with pytest.raises(ValueError):
        toa._parse_TOA_line(badline)

    # Flag without corresponding value (-a)
    badline = "55731.000019.3.000.000.9y.x.ff 428.000000 55731.193719931024413   2.959  ao -a -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 1200.7 -tmplt J1853+1303.430.PUPPI.9y.x.sum.sm -gof 0.98 -nbin 2048 -nch 1 -chan 2 -subint 0 -snr 36.653 -wt 20 -proc 9y -pta NANOGrav -to -0.789e-6"
    with pytest.raises(ValueError):
        toa._parse_TOA_line(badline)

    # Empty flag label (-)
    badline = "55731.000019.3.000.000.9y.x.ff 428.000000 55731.193719931024413   2.959  ao - a -fe 430 -be ASP -f 430_ASP -bw 4 -tobs 1200.7 -tmplt J1853+1303.430.PUPPI.9y.x.sum.sm -gof 0.98 -nbin 2048 -nch 1 -chan 2 -subint 0 -snr 36.653 -wt 20 -proc 9y -pta NANOGrav -to -0.789e-6"
    with pytest.raises(ValueError):
        toa._parse_TOA_line(badline)
