import numpy as np

# These constants are taken from Irwin & Fukushima 1999.
# These are the same as the constants used in tempo2 as of 10 Feb 2023.
IFTE_MJD0 = np.longdouble("43144.0003725")
IFTE_KM1 = np.longdouble("1.55051979176e-8")
IFTE_K = 1 + IFTE_KM1
IFTE_F = 1/(1 + IFTE_KM1)