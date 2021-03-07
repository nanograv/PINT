from hypothesis import given
from hypothesis.strategies import integers
import time


@given(integers())
def test_deadline(x):
    time.sleep(3)
