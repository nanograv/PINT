from io import StringIO
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform

def test_fn_removal():
    m = get_model(StringIO("""
        PSR J1235+5678
        F0 1
        F1 0
        F2 1e-30
        ELAT 0
        ELONG 0
        PEPOCH 57000
        """))
    m.remove_param("F2")
    make_fake_toas_uniform(57000, 58000, 2, m)
