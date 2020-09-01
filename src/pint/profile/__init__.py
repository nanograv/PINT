
import pint.profile.fftfit_aarchiba
import pint.profile.fftfit_nustar
import pint.profile.fftfit_presto

def fftfit_full(template, profile, code="aarchiba"):
    if code=="aarchiba":
        return pint.profile.fftfit_aarchiba.fftfit_full(template, profile)
    elif code=="nustar":
        return pint.profile.fftfit_nustar.fftfit_full(template, profile)
    elif code=="presto":
        if pint.profile.fftfit_presto.presto is None:
            raise ValueError("The PRESTO compiled code is not available")
        return pint.profile.fftfit_presto.fftfit_full(template, profile)
    else:
        raise ValueError("Unrecognized FFTFIT implementation {}".format(code))

def fftfit_basic(template, profile, code="aarchiba"):
    if code=="aarchiba":
        return pint.profile.fftfit_aarchiba.fftfit_basic(template, profile)
    else:
        return fftfit_full(template, profile, code=code).shift

