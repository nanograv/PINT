import toa

x = toa.TOAs("tests/test1.tim")

assert(len(x.commands)==14)
assert(len(x.toas)==9)
assert(x.toas[0].flags["info"]=="test1")
assert(x.toas[0].flags["jump"]==0)
assert(x.toas[3].flags["info"]=="test2")
assert(x.toas[3].flags["time"]==1.0)
assert("jump" not in x.toas[4].flags)
assert("time" not in x.toas[4].flags)
