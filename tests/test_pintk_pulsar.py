import re

import numpy as np

import pint.pintk.pulsar

tim = """
FORMAT 1
unk 999999.000000 57000.0000000078830324 1.000 gbt  -pn -2273593021.0
unk 999999.000000 57052.6315789538116088 1.000 gbt  -pn -124285.0
unk 999999.000000 57105.2631578955917593 1.000 gbt  -pn 2273470219.0
unk 999999.000000 57157.8947368420341204 1.000 gbt  -pn 4547256435.0
JUMP
unk 999999.000000 57210.5263157897339815 1.000 gbt  -pn 6821152750.0
unk 999999.000000 57263.1578947318551852 1.000 gbt  -pn 9095000767.0
unk 999999.000000 57315.7894736865498264 1.000 gbt  -pn 11368677703.0
unk 999999.000000 57368.4210526391189699 1.000 gbt  -pn 13642183645.0
JUMP
unk 999999.000000 57421.0526315687085764 1.000 gbt  -pn 15915655534.0
unk 999999.000000 57473.6842105144226621 1.000 gbt  -pn 18189261252.0
unk 999999.000000 57526.3157894708454977 1.000 gbt  -pn 20463057581.0
unk 999999.000000 57578.9473684237653588 1.000 gbt  -pn 22736955090.0
JUMP
unk 999999.000000 57631.5789473769360416 1.000 gbt  -pn 25010795383.0
unk 999999.000000 57684.2105263208505671 1.000 gbt  -pn 27284460486.0
unk 999999.000000 57736.8421052672976158 1.000 gbt  -pn 29557959408.0
unk 999999.000000 57789.4736842041808449 1.000 gbt  -pn 31831435551.0
JUMP
JUMP
unk 999999.000000 57842.1052631637698032 1.000 gbt  -pn 34105052652.0
unk 999999.000000 57894.7368420936182176 1.000 gbt  -pn 36378858766.0
unk 999999.000000 57947.3684210606924768 1.000 gbt  -pn 38652757406.0
unk 999999.000000 57999.9999999883338542 1.000 gbt  -pn 40926589498.0
JUMP
"""

par = """
PSR J1234+5678
ELAT 0
ELONG 0
PEPOCH 57000
POSEPOCH 57000
F0 500
JUMP mjd 58000 60000 0
JUMP mjd 59000 60000 0
"""


def test_fit_summary(tmp_path, capsys):
    # Pulsar can't cope with file-like objects in place of filenames - I think?
    par_file = tmp_path / "file.par"
    with open(par_file, "wt") as f:
        f.write(par)
    tim_file = tmp_path / "file.tim"
    with open(tim_file, "wt") as f:
        f.write(tim)
    p = pint.pintk.pulsar.Pulsar(parfile=str(par_file), timfile=str(tim_file))
    p.fit(np.ones(len(p.all_toas), dtype=bool))

    capsys.readouterr()
    p.write_fit_summary()
    captured_fit_summary = capsys.readouterr()

    assert re.search(r"Post-Fit Chi2:\s*[0-9.]+", captured_fit_summary.out)
    assert not re.search(r"Post-Fit Chi2:\s*[0-9.]+ +us", captured_fit_summary.out)
    assert re.search(r"Post-Fit Weighted RMS:\s*[0-9.]+ +us", captured_fit_summary.out)
    assert re.search(r"\s*JUMP", captured_fit_summary.out)

    p.reset_model()
    p.reset_TOAs()
    p.resetAll()
    p.update_resids()
    p.add_model_params()

    assert all(np.isfinite(p.orbitalphase()))
    assert all(np.isfinite(p.dayofyear()))
    assert all(np.isfinite(p.year()))
