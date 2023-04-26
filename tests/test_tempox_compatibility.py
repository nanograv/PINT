from io import StringIO
import re

import pytest
from pint.models import get_model
import pint.toa

par_basic = """
PSR J1234+5678
ELAT 0
ELONG 0
PEPOCH 57000
F0 1
JUMP -tel ao 1
"""


@pytest.mark.parametrize(
    "name,want",
    [
        ("EFAC", "EFAC"),
        ("EQUAD", "EQUAD"),
        ("ECORR", "ECORR"),
        ("T2EFAC", "EFAC"),
        ("T2EQUAD", "EQUAD"),
        ("TNECORR", "ECORR"),
        # pytest.param("T2EFAC", "EFAC", marks=pytest.mark.xfail(reason="Bug #1019")),
        # pytest.param("T2EQUAD", "EQUAD", marks=pytest.mark.xfail(reason="Bug #1019")),
        # pytest.param("TNECORR", "ECORR", marks=pytest.mark.xfail(reason="Bug #1019")),
    ],
)
def test_noise_parameter_aliases(name, want):
    parfile = "\n".join([par_basic, f"{name} -f bogus 1.234"])
    m = get_model(StringIO(parfile))
    assert getattr(m, f"{want}1").value == 1.234
    assert re.search(
        f"^{want}" + r"\s+-f\s+bogus\s+1.23\d+\s*$", m.as_parfile(), re.MULTILINE
    )


def test_tim_aliases():
    t = pint.toa.get_TOAs(
        StringIO(
            """
            FORMAT 1
            a_file.extension 1430.000000 53393.561383615118386   0.178  ao  -fe L-wide -be ASP -f L-wide_ASP
            """
        )
    )
    f = StringIO()
    t.write_TOA_file(f)
    raw = f.getvalue()

    f = StringIO()
    t.alias_translation = pint.toa.tempo_aliases
    t.write_TOA_file(f)
    translated = f.getvalue()

    assert "arecibo" in raw
    assert "arecibo" not in translated
    assert "AO" in translated


@pytest.mark.parametrize(
    "p", [ln.split()[0] for ln in par_basic.split("\n") if ln.split()]
)
def test_par_aliases(p):
    m = get_model(StringIO(par_basic))
    m.use_aliases(alias_translation={p: "CAPYBARA"})
    pf = m.as_parfile()
    assert not any(ln.startswith(p) for ln in pf.split("\n"))
    assert any(ln.startswith("CAPYBARA") for ln in pf.split("\n"))
