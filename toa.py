import astropy.constants as c
import astropy.time as t
import re
import struct
import numpy

toa_commands = ("DITHER", "EFAC", "EMAX", "EMAP", "EMIN", "EQUAD", "FMAX",
                "FMIN", "INCLUDE", "INFO", "JUMP", "MODE", "NOSKIP", "PHA1",
                "PHA2", "PHASE", "SEARCH", "SIGMA", "SIM", "SKIP", "TIME",
                "TRACK", "ZAWGT", "FORMAT")


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
        else:
            if type(MJD) is not str:
                sys.stderr.write("Warning:  possibly loss of precision with %s" %
                                 str(type(MJD)))
            self.mjd = t.Time(MJD, scale=scale, format='mjd', precision=9)
        # set the other three required values
        self.obs = obs
        self.freq = freq
        self.error = error
        # set any other optional params
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __str__(self):
        s = str(self.mjd) + ": %.3f us at '%s' at %.4f MHz" % \
            (self.error, self.obs, self.freq)
        other_keys = [k for k in self.__dict__.keys() if k not in self.__std_params]
        if len(other_keys):
            s += "\n" + str(dict([(k,self.__dict__[k]) for k in other_keys]))
        return s

    def parse_line(self):
        self.command = None
        self.arg = None
        self.site = None
        self.freq = None
        self.mjd = None
        self.imjd = None
        self.fmjd = None
        self.error = None
        self.ddm = None
        self.res = None
        self.fname = None
        self.info = None
        self.flags = {}
        self.format = toa_format(self.line)
        if self.format=="Command":
            tmp = self.line.split()
            if len(tmp)==1:
                self.command = tmp[0]
            elif len(tmp)>1:
                self.command = tmp[0]
                self.arg = tmp[1]
        elif self.format=="Princeton":
            self.site = self.line[0]
            self.freq = float(self.line[15:24])
            self.mjd = float(self.line[24:44])
            self.imjd = int(self.mjd)
            if self.line[29]=='.':
                self.fmjd = float(self.line[29:44])
            elif self.line[30]=='.':
                self.fmjd = float(self.line[30:44])
            self.error = float(self.line[44:53])
            try:
                self.ddm = float(self.line[68:78])
            except:
                self.ddm = 0.0
        elif self.format=="Tempo2":
            # This could use more error catching...
            fields = self.line.split()
            self.name = fields.pop(0)
            self.freq = float(fields.pop(0))
            mjdstr = fields.pop(0)
            self.mjd = float(mjdstr)
            self.imjd = int(self.mjd)
            self.fmjd = float(mjdstr[mjdstr.find('.'):])
            self.error = float(fields.pop(0))
            self.site = fields.pop(0)
            # All the rest should be flags
            for i in range(0,len(fields),2):
                self.flags[fields[i].lstrip('-')] = fields[i+1]
        elif self.format=="Parkes" or self.format=="ITOA":
            raise RuntimeError, \
                "TOA format '%s' not implemented yet" % self.format

    def is_toa(self):
        if self.format in ("Tempo2", "Princeton", "Parkes", "ITOA"):
            return True
        else:
            return False


class TOAs(list):

    def get_resids(self,units='us'):
        if units=='us':
            return numpy.array([t.res.res_us for t in self if t.is_toa()])
        elif units=='phase':
            return numpy.array([t.res.res_phase for t in self if t.is_toa()])

    def get_freq(self):
        return numpy.array([t.freq for t in self if t.is_toa()])

    def get_mjd(self):
        return numpy.array([t.mjd for t in self if t.is_toa()])

    def get_err(self):
        return numpy.array([t.error for t in self if t.is_toa()])

    def get_flag(self,flag,f=lambda x: x):
        return numpy.array([f(t.flags[flag]) for t in self if t.is_toa()])


    def read_toa_file(filename, process_includes=True, ignore_blanks=True, top=True):
        """Read the given filename and return a list of toa objects 
        parsed from it.  Will recurse to process INCLUDE-d files unless
        process_includes is set to False.  top is used internally for
        processing INCLUDEs and should always be set to True."""
        f = open(filename, "r")
        toas = toalist([])
        skip = False
        if top: read_toa_file.info = None
        for l in f.readlines():
            newtoa = toa(l)
            if newtoa.format=="Command":
                if newtoa.command=="SKIP":
                    skip = True
                elif newtoa.command=="NOSKIP":
                    skip = False
                    continue
                if skip: continue
                if newtoa.command=="INFO":
                    read_toa_file.info = newtoa.arg
                if newtoa.command=="INCLUDE" and process_includes:
                    toas += read_toa_file(newtoa.arg,
                            ignore_blanks=ignore_blanks,top=False)
                else:
                    toas += [newtoa]
            elif skip:
                continue
            elif newtoa.format in ("Blank", "Unknown"):
                if not ignore_blanks:
                    toas += [newtoa]
            else:
                newtoa.info = read_toa_file.info
                toas += [newtoa]
        f.close()
        return toas
