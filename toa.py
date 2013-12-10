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
        d[fmt] = line.split()
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
            k, v = flags[i].lstrip('-'), flags[i+1]
            try:  # Convert what we can to floats and ints
                d[k] = int(v)
            except ValueError:
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
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
        >>> a = toa((54567, 0.876876876876876), 4.5, freq=1400.0,
                    obs="GBT", backend="GUPPI")
        >>> print a
        54567.876876876876876: 4.500 us at 'GBT' at 1400.0000 MHz
        {'backend': 'GUPPI'}

    """
    def __init__(self, MJD, # required
                 error=0.0, obs='bary', freq=float("inf"), scale='utc', # with defaults
                 **kwargs):  # keyword args that are completely optional
        try:
            self.mjd = t.Time(*MJD, scale=scale, format='mjd', precision=9)
        except:
            "Error processing MJD for TOA:", MJD
        self.error = error
        self.obs = obs
        self.freq = freq
        self.flags = kwargs

    def __str__(self):
        s = str(self.mjd) + ": %6.3f us error from '%s' at %.4f MHz " % \
            (self.error, self.obs, self.freq)
        if len(self.flags):
            s += str(self.flags)
        return s


class TOAs(object):

    def __init__(self, toafile=None):
        if toafile:
            if type(toafile) in [tuple, list]:
                for infile in toafile:
                    self.read_toa_file(infile)
            else:
                self.read_toa_file(toafile)
        else:
            self.toas = []
            self.commands = []

    def get_freqs(self):
        return numpy.array([t.freq for t in self.toas])

    def get_mjds(self):
        return numpy.array([t.mjd for t in self.toas])

    def get_errs(self):
        return numpy.array([t.error for t in self.toas])

    def summary(self):
        print "There are %d observatories:" % len(self.observatories), \
              list(self.observatories)
        print "There are %d TOAs" % len([x for x in self.toas])
        print "There are %d commands" % \
              len([x for x in self.commands])
        errs = self.get_errs()
        print "Min / Max TOA errors:", min(errs), max(errs)
        print "Mean / Median / StDev TOA error:", errs.mean(), \
              numpy.median(errs), errs.std()

    def read_toa_file(self, filename, process_includes=True, top=True):
        """
        read_toa_file(filename, process_includes=True)
            Read the given filename and return a list of toa objects 
            parsed from it.  Will recurse to process INCLUDE-d files unless
            process_includes is set to False.
        """
        if top:
            self.toas = []
            self.commands = []
            self.cdict = {"EFAC": 1.0, "EQUAD":1.0,
                          "EMIN": 0.0, "EMAX": 1e100,
                          "FMIN": 0.0, "FMAX": 1e100,
                          "INFO": None, "SKIP": False,
                          "TIME": 0.0, "PHASE": 0,
                          "PHA1": None, "PHA2": None,
                          "MODE": 1, "JUMP": [False, 0]}
            self.observatories = set()
        with open(filename, "r") as f:
            skip = False
            for l in f.readlines():
                MJD, d = parse_TOA_line(l)
                # print MJD, d
                if d["format"]=="Command":
                    cmd = d["Command"][0]
                    self.commands.append((d["Command"], len(self.toas)))
                    if cmd=="SKIP":
                        self.cdict[cmd] = True
                    elif cmd=="NOSKIP":
                        self.cdict[cmd] = False
                    if self.cdict["SKIP"]:
                        continue
                    if cmd=="END":
                        break
                    elif cmd in ("EMIN", "EMAX", "EFAC", "EQUAD",\
                                 "PHA1", "PHA2", "TIME", "PHASE",\
                                 "FMIN", "FMAX"):
                        self.cdict[cmd] = float(d["Command"][1])
                        if cmd in ("PHA1", "PHA2", "TIME", "PHASE"):
                            d[cmd] = d["Command"][1]
                    elif cmd=="INFO":
                        self.cdict[cmd] = d["Command"][1]
                        d[cmd] = d["Command"][1]
                    elif cmd=="JUMP":
                        if (self.cdict[cmd][0]):
                            self.cdict[cmd][0] = False
                            self.cdict[cmd][1] += 1
                        else:
                            self.cdict[cmd][0] = True
                    elif cmd=="INCLUDE" and process_includes:
                        self.read_toa_file(d["Command"][1], top=False)
                    else:
                        continue
                if d["format"] in ("Blank", "Unknown", "Comment"):
                    continue
                else:
                    newtoa = toa(MJD, **d)
                    if ((self.cdict["EMIN"] > newtoa.error) or \
                        (self.cdict["EMAX"] < newtoa.error) or \
                        (self.cdict["FMIN"] > newtoa.freq) or \
                        (self.cdict["FMAX"] < newtoa.freq)):
                        continue
                    else:
                        newtoa.error *= self.cdict["EFAC"]
                        newtoa.error = numpy.hypot(newtoa.error, self.cdict["EQUAD"])
                        if self.cdict["INFO"]:
                            newtoa.flags["INFO"] = self.cdict["INFO"]
                        if self.cdict["JUMP"][0]:
                            newtoa.flags["JUMP"] = self.cdict["JUMP"][1]
                        if self.cdict["PHASE"] != 0:
                            newtoa.flags["PHASE"] = self.cdict["PHASE"]
                        if self.cdict["TIME"] != 0.0:
                            newtoa.flags["TIME"] = self.cdict["TIME"]
                        self.observatories.add(newtoa.obs)
                        self.toas.append(newtoa)
            if top:
                # Clean up our temporaries used when reading TOAs
                del(self.cdict)
