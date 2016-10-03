from pint import toa
import os

from pinttestdata import testdir, datadir
os.chdir(datadir)

class TestTOAReader:
    def setUp(self):
        self.x = toa.TOAs("test1.tim", usepickle=False)
        self.x.apply_clock_corrections()
        self.x.compute_TDBs()
        self.x.table.sort('index')
    def test_commands(self):
        assert len(self.x.commands) == 18
    def test_count(self):
        assert self.x.ntoas == 9
    def test_info(self):
        assert self.x.table[0]['flags']["info"] == "test1"
    def test_jump(self):
        assert self.x.table[0]['flags']["jump"] == 0
    def test_info_2(self):
        assert self.x.table[3]['flags']["info"] == "test2"
    def test_time(self):
        assert self.x.table[3]['flags']["to"] == 1.0
    def test_jump_2(self):
        assert "jump" not in self.x.table[4]['flags']
    def test_time_2(self):
        assert "time" not in self.x.table[4]['flags']
    def test_jump_3(self):
        assert self.x.table[-1]['flags']["jump"] == 1
    def test_obs(self):
        assert self.x.table[1]["obs"]=="GBT"

if __name__ == '__main__':
    t = TestTOAReader()
    t.setUp()
    print 'Tests are set up.'

    t.test_commands()
    t.test_count()
    t.test_info()
    t.test_jump()
    t.test_info_2()
    t.test_time()
    t.test_jump_2()
    t.test_time_2()
    t.test_jump_3()
    t.test_obs()
