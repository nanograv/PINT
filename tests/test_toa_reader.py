from pint import toa

class TestTOAReader:
    def setUp(self):
        self.x = toa.TOAs("tests/test1.tim")
        self.x.apply_clock_corrections()
        self.x.compute_TDBs()

    def test_commands(self):
        assert len(self.x.commands) == 18
    def test_count(self):
        assert self.x.ntoas == 9
    def test_info(self):
        assert self.x.toas[0].flags["info"] == "test1"
    def test_jump(self):
        assert self.x.toas[0].flags["jump"] == 0
    def test_info_2(self):
        assert self.x.toas[3].flags["info"] == "test2"
    def test_time(self):
        assert self.x.toas[3].flags["time"] == 1.0
    def test_jump_2(self):
        assert "jump" not in self.x.toas[4].flags
    def test_time_2(self):
        assert "time" not in self.x.toas[4].flags
    def test_jump_3(self):
        assert self.x.toas[-1].flags["jump"] == 1
    def test_obs(self):
        assert self.x.toas[1].obs=="GBT" 
