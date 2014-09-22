from pint import toa, toaTable

class TestTOAReader:
    def setUp(self):
        self.x = toa.TOAs("tests/test1.tim")
        self.y = toaTable.TOAs("tests/test1.tim")

    def test_commands(self):
        assert len(self.x.commands) == 16
        assert len(self.y.commands) == 16
    def test_count(self):
        assert len(self.x.toas) == 9
        assert len(self.y.toas) == 9
    def test_info(self):
        assert self.x.toas[0].flags["info"] == "test1"
        assert self.y.toas[0].flags["info"] == "test1"
    def test_jump(self):
        assert self.x.toas[0].flags["jump"] == 0
        assert self.y.toas[0].flags["jump"] == 0
    def test_info_2(self):
        assert self.x.toas[3].flags["info"] == "test2"
        assert self.y.toas[3].flags["info"] == "test2"
    def test_time(self):
        assert self.x.toas[3].flags["time"] == 1.0
        assert self.y.toas[3].flags["time"] == 1.0
    def test_jump_2(self):
        assert "jump" not in self.x.toas[4].flags
        assert "jump" not in self.y.toas[4].flags
    def test_time_2(self):
        assert "time" not in self.x.toas[4].flags
        assert "time" not in self.y.toas[4].flags
    def test_jump_3(self):
        assert self.x.toas[-1].flags["jump"] == 1
        assert self.y.toas[-1].flags["jump"] == 1
    def test_obs(self):
        assert self.x.toas[1].obs=="GBT" 
        assert self.y.toas[1].obs=="GBT" 
