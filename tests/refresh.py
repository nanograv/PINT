from pint.models import get_model_and_toas
from pinttestdata import datadir

get_model_and_toas(datadir / "NGC6440E.par", datadir / "NGC6440E.tim")
