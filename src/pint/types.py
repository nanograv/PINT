from pathlib import Path
import numpy as np
from typing import Union, IO
from astropy import units as u
import astropy.time

# custom types
# Something that is a Quantity or can behave like one (with units assumed)
quantity_like = Union[float, np.ndarray, u.Quantity]
# Something that is a Time or can behave like one
time_like = Union[float, np.ndarray, u.Quantity, astropy.time.Time]
file_like = Union[str, Path, IO]
dir_like = Union[str, Path]
toas_index_like = Union[str, tuple, np.ndarray, slice, int]
