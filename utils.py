# utils.py
# Miscellaneous potentially-helpful functions
import string
import astropy.time

def fortran_float(x):
    """
    fortran_float(x)

    returns a copy of the input string with all 'D' or 'd' turned
    into 'e' characters.  Intended for dealing with exponential 
    notation in tempo1-generated parfiles.
    """
    return float(x.translate(string.maketrans('Dd','ee')))

def time_from_mjd_string(s):
    """
    timse_from_mjd_string(s)

    Returns an astropy Time object generated from a MJD string input.
    """
    ss = s.lower()
    if ("e" in ss or "d" in ss):
        ss = ss.translate(string.maketrans("d", "e"))
        num, expon = ss.split("e")
        expon = int(expon)
        if (expon < 0):
            print "Warning:  likely bogus sci notation input in time_from_mjd_string ('%s')!" % s
            # This could cause a loss of precision...
            # maybe throw an exception instead?
            imjd, fmjd = 0, float(ss)
        else:
            imjd_s, fmjd_s = num.split('.')
            imjd = int(imjd_s + fmjd_s[:expon])
            fmjd = float("0."+fmjd_s[expon:])
    else:
        imjd_s, fmjd_s = ss.split('.')
        imjd = int(imjd_s)
        fmjd = float("0." + fmjd_s)
    # TODO: what to do about scale?
    return astropy.time.Time(imjd, fmjd, scale='utc', format='mjd',
                             precision=9)

def time_to_mjd_string(t, prec=15):
    """
    time_to_mjd_string(t, prec=15)

    Print an MJD time with lots of digits (number is 'prec').  astropy
    does not seem to provide this capability (yet?).
    """
    jd1 = t.jd1 - astropy.time.core.MJD_ZERO
    imjd = int(jd1)
    fjd1 = jd1 - imjd
    fmjd = t.jd2 + fjd1
    if fmjd >= 1.0:
        imjd += 1
        fmjd -= 1.0
    fmt = "%."+"%sf"%prec
    return str(imjd) + (fmt%fmjd)[1:]
