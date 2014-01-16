import astropy.utils
from astropy.utils.iers import IERS_A, IERS_A_URL
from astropy.utils.data import download_file
import astropy.time

class TestAstroPyTime:
    """This class contains a sequence of time conversion tests.

    From the SOFA manual, these times are all equivalent:
    UTC 2006/01/15 21:24:37.500000
    UT1 2006/01/15 21:24:37.834100
    TAI 2006/01/15 21:25:10.500000
    TT 2006/01/15 21:25:42.684000
    TCG 2006/01/15 21:25:43.322690
    TDB 2006/01/15 21:25:42.684373
    TCB 2006/01/15 21:25:56.893952
    """

    def setUp(self):
        self.lat = 19.48125
        self.lon = -155.933222
        self.t = astropy.time.Time('2006-01-15 21:24:37.5', \
                                   format='iso', scale='utc', \
                                   lat=self.lat, lon=self.lon, precision=6)
    def test_utc(self):
        x = self.t.utc.iso
        y = '2006-01-15 21:24:37.500000'
        assert x == y
    def test_utc_ut1(self):
        x = self.t.ut1.iso
        y = '2006-01-15 21:24:37.834078'
        assert x == y
    def test_ut1_tai(self):
        x = self.t.tai.iso
        y = '2006-01-15 21:25:10.500000'
        assert x == y
    def test_tai_tt(self):
        x = self.t.tt.iso
        y = '2006-01-15 21:25:42.684000'
        assert x == y
    def test_tt_tcg(self):
        x = self.t.tcg.iso
        y = '2006-01-15 21:25:43.322690'
        assert x == y
    def test_tt_tdb(self):
        x = self.t.tdb.iso
        y = '2006-01-15 21:25:42.684373'
        assert x == y
    def test_tt_tcb(self):
        x = self.t.tcb.iso
        y = '2006-01-15 21:25:56.893952'
        assert x == y


def test_iers_a_now():
    iers_a_file = download_file(IERS_A_URL, cache=True)
    iers_a = IERS_A.open(iers_a_file)
    t2 = astropy.time.Time.now()
    t2.delta_ut1_utc = t2.get_delta_ut1_utc(iers_a)
    print t2.tdb.iso

