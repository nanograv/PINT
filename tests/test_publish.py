from pinttestdata import datadir
import pytest
from pint.models import get_model_and_toas
from pint.output.publish import publish
from pint.scripts import pintpublish
import os

data_NGC6440E = get_model_and_toas(datadir / "NGC6440E.par", datadir / "NGC6440E.tim")


def test_NGC6440E():
    m, t = data_NGC6440E
    output = publish(m, t)
    assert "1748-2021E" in output
    assert "DE421" in output


data_J0613m0200_NANOGrav_9yv1 = get_model_and_toas(
    datadir / "J0613-0200_NANOGrav_9yv1.gls.par",
    datadir / "J0613-0200_NANOGrav_9yv1.tim",
)


@pytest.mark.parametrize("full", [True, False])
def test_J0613m0200_NANOGrav_9yv1(full):
    m, t = data_J0613m0200_NANOGrav_9yv1
    output = publish(
        m, t, include_dmx=full, include_fd=full, include_noise=full, include_jumps=full
    )

    assert "J0613-0200" in output
    assert "ELL1" in output
    assert "Narrowband" in output
    assert "freedom" not in output
    assert "Reduced" in output
    assert not full or "DMX" in output
    assert not full or "JUMP" in output
    assert not full or "FD1" in output
    assert not full or "RNAMP" in output


data_J1614m2230_NANOGrav_12yv3_wb = get_model_and_toas(
    datadir / "J1614-2230_NANOGrav_12yv3.wb.gls.par",
    datadir / "J1614-2230_NANOGrav_12yv3.wb.tim",
)


@pytest.mark.parametrize("full", [True, False])
def test_J1614m2230_NANOGrav_12yv3_wb(full):
    m, t = data_J1614m2230_NANOGrav_12yv3_wb
    output = publish(
        m, t, include_dmx=full, include_fd=full, include_noise=full, include_jumps=full
    )

    assert "DE436" in output
    assert "ELL1" in output
    assert "Wideband" in output
    assert "freedom" in output
    assert not full or "DMX" in output
    assert not full or "DMJUMP" in output
    assert not full or "JUMP" in output
    assert "TT(BIPM2017)" in output


@pytest.mark.parametrize("file", [True, False])
def test_script(file):
    par, tim = str(datadir / "NGC6440E.par"), str(datadir / "NGC6440E.tim")
    outfile = "--outfile=pintpublish_test.tex" if file else ""
    args = f"{par} {tim} {outfile}"
    pintpublish.main(args.split())

    assert not file or os.path.isfile("pintpublish_test.tex")
