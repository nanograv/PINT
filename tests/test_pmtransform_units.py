import io
import numpy as np
from astropy import units as u, constants as c
from astropy.time import Time
from pint.models import get_model
import pint.logging

pint.logging.setup("WARNING")
modelstring_ECL = """
PSR              B1855+09
LAMBDA   286.8634893301156  1     0.0000000165859
BETA      32.3214877555037  1     0.0000000273526
PMLAMBDA           -3.2701  1              0.0141
PMBETA             -5.0982  1              0.0291
F0    186.4940812707752116  1  0.0000000000328468
F1     -6.205147513395D-16  1  1.379566413719D-19
PEPOCH        54978.000000
POSEPOCH        54978.000000
START            53358.726
FINISH           56598.873
DM               13.299393
"""
modelstring_ICRS = """
PSRJ           1855+09
RAJ             18:57:36.3932884         1  0.00002602730280675029
DECJ           +09:43:17.29196           1  0.00078789485676919773
F0             186.49408156698235146     1  0.00000000000698911818
F1             -6.2049547277487420583e-16 1  1.7380934373573401505e-20
PEPOCH        54978.000000
POSEPOCH        54978.000000
START            53358.726
FINISH           56598.873
DM             13.29709
PMRA           -2.5054345161030380639    1  0.03104958261053317181
PMDEC          -5.4974558631993817232    1  0.06348008663748286318
"""

mECL = get_model(io.StringIO(modelstring_ECL))
mICRS = get_model(io.StringIO(modelstring_ICRS))
models = {"ECL": mECL, "ICRS": mICRS}

t_float = np.linspace(55000, 56000)
for modelframe in ["ECL", "ICRS"]:
    for coordframe in ["ECL", "ICRS"]:
        for ttype in ["float", "quantity", "time"]:
            if ttype == "float":
                t = t_float
            elif ttype == "quantity":
                t = t_float * u.d
            elif ttype == "time":
                t = Time(t_float, format="mjd")
            print(f"Running m{modelframe}.ssb_to_psb_xyz_{coordframe}({ttype})")
            getattr(models[modelframe], f"ssb_to_psb_xyz_{coordframe}")(t)
            print(f"Running m{modelframe}.coords_as_{coordframe}({ttype})")
            getattr(models[modelframe], f"coords_as_{coordframe}")(t)
            print(f"Running m{modelframe}.as_{coordframe}({ttype})")
            getattr(models[modelframe], f"as_{coordframe}")(t)
