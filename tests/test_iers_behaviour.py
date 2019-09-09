import pytest

from astropy.table import Table
from astropy.utils.iers import IERS_A, IERS_A_URL, IERS_A_FILE, IERS_A_README
from astropy.utils.data import download_file

masks_needed = ["UT1_UTC_A", "PolPMFlag_A"]

@pytest.mark.parametrize("c", masks_needed)
@pytest.mark.remote_data
def test_table_masked(c):
    A = Table.read(download_file(IERS_A_URL, cache=True), format="cds", readme=IERS_A_README)
    A[c].mask
