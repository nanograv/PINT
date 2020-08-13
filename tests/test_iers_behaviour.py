import pytest
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.utils.iers import IERS_A_README, IERS_A_URL

# from urllib.error import HTTPError

masks_needed = ["UT1_UTC_A", "PolPMFlag_A"]


@pytest.mark.parametrize("c", masks_needed)
def test_table_masked(c):
    try:
        A = Table.read(
            download_file(IERS_A_URL, cache=True), format="cds", readme=IERS_A_README
        )
    except IOError:
        pytest.skip("Unable to obtain IERS A data")
    A[c].mask
