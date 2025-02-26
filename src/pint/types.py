from pathlib import Path
from typing import IO, Union

import astropy.time
import numpy as np
from astropy import units as u

# custom types
# Something that is a Quantity or can behave like one (with units assumed)
quantity_like = Union[float, np.ndarray, u.Quantity]
# Something that is a Time or can behave like one
time_like = Union[float, np.ndarray, u.Quantity, astropy.time.Time]
file_like = Union[str, Path, IO]
dir_like = Union[str, Path]
toas_index_like = Union[str, tuple, np.ndarray, slice, int]
