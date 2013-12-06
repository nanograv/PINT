import astropy.time as t
import re, sys
import numpy
from observatories import read_observatories

toa_commands = ("DITHER", "EFAC", "EMAX", "EMAP", "EMIN", "EQUAD", "FMAX",
                "FMIN", "INCLUDE", "INFO", "JUMP", "MODE", "NOSKIP", "PHA1",
                "PHA2", "PHASE", "SEARCH", "SIGMA", "SIM", "SKIP", "TIME",
                "TRACK", "ZAWGT", "FORMAT")

obss, obscode1s, obscode2s = read_observatories()

def toa_format(line):
    """Identifies a TOA line as one of the following types:  Comment, Command,
    Blank, Tempo2, Princeton, ITOA, Parkes, Unknown."""
    try:
        if line[0]=='C' or line[0]=='#':
            return "Comment"
        elif line.startswith(toa_commands):
            return "Command"
        elif re.match("^\s+$", line):
            return "Blank"
        elif len(line) > 80: 
            return "Tempo2"
        elif re.match("[0-9a-z@] ",line):
            return "Princeton"
        elif re.match("\S\S",line) and line[14]=='.':
            return "ITOA"
        elif re.match("  ",line) and line[41]=='.':
            return "Parkes"
        else:
            return "Unknown"
    except:
        return "Unknown"

def parse_TOA_line(line):
    MJD = None
    d = {}
    fmt = toa_format(line)
    d["format"] = fmt
    if fmt=="Command":
        MJD = "Command"
        d["command"] = line.split()
    elif fmt=="Princeton":
        fields = line.split()
        d["obs"] = obscode1s[line[0]]
        d["freq"] = float(fields[1])
        d["error"] = float(fields[3])
        ii, ff = fields[2].split('.')
        MJD = (int(ii), float("0."+ff))
        try:
            d["ddm"] = float(fields[4])
        except:
            d["ddm"] = 0.0
    elif fmt=="Tempo2":
        # This could use more error catching...
        fields = line.split()
        d["name"] = fields[0]
        d["freq"] = float(fields[1])
        ii, ff = fields[2].split('.')
        MJD = (int(ii), float("0."+ff))
        d["error"] = float(fields[3])
        ocode = fields[4].upper()
        if ocode in obscode2s.keys():
            d["obs"] = obscode2s[ocode]
        elif ocode in obss.keys():
            d["obs"] = ocode
        # All the rest should be flags
        flags = fields[5:]
        for i in range(0, len(flags), 2):
            d[flags[i].lstrip('-')] = flags[i+1]
    elif fmt=="Parkes" or fmt=="ITOA":
        raise RuntimeError, \
            "TOA format '%s' not implemented yet" % self.format
    return MJD, d

class toa(object):
    """
    toa(MJD, error=0.0, obs='bary', freq=float("inf"), scale='utc',...)
        A time of arrival class.

        MJD will be stored in astropy.time.Time format, and can be
            passed as a double (not recommended), a string, a
            tuple of component parts (day and fraction of day).
        error is the TOA uncertainty in microseconds
        obs is the observatory name as defined in XXX
        freq is the observatory-centric frequency in MHz
        freq
        other keyword/value pairs can be specified as needed

    Example:
        >>> a = toa("54567.876876876876876", 4.5, freq=1400.0,
                    obs="GBT", backend="GUPPI")
        >>> print a
        54567.876876876876876: 4.500 us at 'GBT' at 1400.0000 MHz
        {'backend': 'GUPPI'}

    """
    __std_params = ['mjd', 'freq', 'error', 'obs', 'scale']

    def __init__(self, MJD, # required
                 error=0.0, obs='bary', freq=float("inf"), scale='utc', # with defaults
                 **kwargs):  # keyword args that are completely optional
        if type(MJD) is tuple:
            self.mjd = t.Time(*MJD, scale=scale, format='mjd', precision=9)
            self.info = None
            self.error = error
            self.obs = obs
            self.freq = freq
        else:
            if MJD is "Command":
                self.format = "Command"
            elif MJD is None: # Blank line, comment, unknown
                pass
        # set any other optional params
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __str__(self):
        if self.format in ("Tempo2", "Princeton", "Parkes", "ITOA"):
            s = str(self.mjd) + ": %.3f us at '%s' at %.4f MHz" % \
                (self.error, self.obs, self.freq)
        else:
            s = self.format
        other_keys = [k for k in self.__dict__.keys() if k not in self.__std_params]
        if len(other_keys):
            s += "\n" + str(dict([(k,self.__dict__[k]) for k in other_keys]))
        return s

    def is_toa(self):
        if self.format in ("Tempo2", "Princeton", "Parkes", "ITOA"):
            return True
        else:
            return False


class TOAs(object):

    def __init__(self, toafile):
        if type(toafile) in [tuple, list]:
            for infile in toafile:
                self.read_toa_file(infile)
        else:
            self.read_toa_file(toafile)

    def get_freqs(self):
        return numpy.array([t.freq for t in self if t.is_toa()])

    def get_mjds(self):
        return numpy.array([t.mjd for t in self if t.is_toa()])

    def get_errs(self):
        return numpy.array([t.error for t in self if t.is_toa()])

    def get_flags(self,flag,f=lambda x: x):
        return numpy.array([f(t.flags[flag]) for t in self if t.is_toa()])

    def read_toa_file(self, filename, process_includes=True,
                      ignore_blanks=True, top=True):
        """
        read_toa_file(filename, process_includes=True, ignore_blanks=True)
            Read the given filename and return a list of toa objects 
            parsed from it.  Will recurse to process INCLUDE-d files unless
            process_includes is set to False.
        """
        if top:
            self.info = None
            self.toas = []
        with open(filename, "r") as f:
            skip = False
            for l in f.readlines():
                MJD, d = parse_TOA_line(l)
                newtoa = toa(MJD, **d)
                if newtoa.format=="Command":
                    if newtoa.command[0]=="SKIP":
                        skip = True
                    elif newtoa.command[0]=="NOSKIP":
                        skip = False
                        continue
                    if skip: continue
                    if newtoa.command[0]=="INFO":
                        self.info = newtoa.command[1]
                    if newtoa.command[0]=="INCLUDE" and process_includes:
                        self.toas += self.read_toa_file(newtoa.command[1],
                                                        ignore_blanks=ignore_blanks,
                                                        top=False)
                    else:
                        self.toas.append(newtoa)
                elif skip:
                    continue
                elif newtoa.format in ("Blank", "Unknown", "Comment"):
                    if not ignore_blanks:
                        self.toas.append(newtoa)
                else:
                    newtoa.info = self.info
                    self.toas.append(newtoa)
                # print newtoa
            f.close()
            if not top:
                return self.toas
