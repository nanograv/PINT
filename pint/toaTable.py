import re, sys, os, mpmath, cPickle
import numpy
from . import utils
from . import observatories as observatories_module
from . import erfautils
import spice
import astropy.time as time
from astropy.time.core import SECS_PER_DAY
import astropy.table as table
import astropy.units as u
from astropy.utils.iers import IERS_A, IERS_A_URL
from astropy.utils.data import download_file
from .spiceutils import objPosVel, load_kernels
from pint import pintdir
from astropy import log



toa_commands = ("DITHER", "EFAC", "EMAX", "EMAP", "EMIN", "EQUAD", "FMAX",
                "FMIN", "INCLUDE", "INFO", "JUMP", "MODE", "NOSKIP", "PHA1",
                "PHA2", "PHASE", "SEARCH", "SIGMA", "SIM", "SKIP", "TIME",
                "TRACK", "ZAWGT", "FORMAT", "END")


observatories = observatories_module.read_observatories()
iers_a_file = download_file(IERS_A_URL, cache=True)
iers_a = IERS_A.open(iers_a_file)

def toa_format(line, fmt="Unknown"):
    """Determine the type of a TOA line.

    Identifies a TOA line as one of the following types:  Comment, Command,
    Blank, Tempo2, Princeton, ITOA, Parkes, Unknown."""
    if line[0] == 'C' or line[0] == '#':
        return "Comment"
    elif line.startswith(toa_commands):
        return "Command"
    elif re.match(r"^\s+$", line):
        return "Blank"
    elif re.match(r"[0-9a-z@] ", line):
        return "Princeton"
    elif re.match(r"  ", line) and len(line) > 41 and line[41] == '.':
        return "Parkes"
    elif len(line) > 80 or fmt == "Tempo2":
        return "Tempo2"
    elif re.match(r"\S\S", line) and len(line) > 14 and line[14] == '.':
        # FIXME: This needs to be better
        return "ITOA"
    else:
        return "Unknown"



def get_obs(obscode):
    for name in observatories:
        if obscode in observatories[name].aliases:
            return name
    raise ValueError("cannot identify observatory '%s'!" % obscode)

def parse_TOA_line(line, fmt="Unknown"):
    MJD = None
    d = {}
    fmt = toa_format(line, fmt)
    d["format"] = fmt
    if fmt == "Command":
        d[fmt] = line.split()
    elif fmt == "Princeton":
        fields = line.split()
        d["obs"] = get_obs(line[0].upper())
        d["freq"] = float(fields[1])
        d["error"] = float(fields[3])
        ii, ff = fields[2].split('.')
        MJD = (int(ii), float("0."+ff))
        try:
            d["ddm"] = float(fields[4])
        except IndexError:
            d["ddm"] = 0.0     
    elif fmt == "Tempo2":
        # This could use more error catching...
        fields = line.split()
        d["name"] = fields[0]
        d["freq"] = float(fields[1])
        ii, ff = fields[2].split('.')
        MJD = (int(ii), float("0."+ff))
        d["error"] = float(fields[3])
        d["obs"] = get_obs(fields[4].upper())
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
    elif fmt == "Parkes" or fmt == "ITOA": 
        raise RuntimeError(
            "TOA format '%s' not implemented yet" % fmt)
    return MJD,d                           


class TOAs(object):
    """
    A container for toas
    """

    def __init__(self,toafile=None):
        self.toaTable = None     # initialize a table varible but put as None 
        if toafile:
        #FIXME: work with file-like objects   
            if type(toafile) in [tuple,list]:
                self.filename = None
                for infile in taofile:
                    self.read_toafile_table(infile)

            else:
                pth,ext = os.path.splitext(toafile)
                if ext == ".pickle":
                    toafile = pth
                self.read_toa_file_table(toafile)
                self.filename = toafile
            
            self.get_table()
            self.get_time_convert()    
                    
        else:
            self.commands = []
            self.filename = None
          
           
    def read_toa_file_table(self, filename, process_includes=True, top=True):
        """Read the given filename and return a table of toas.

        Will recurse to process INCLUDE-d files unless
        process_includes is set to False.
        """
        if top:
            self.toas = []
            self.cdict = {"EFAC": 1.0, "EQUAD": 0.0,
                          "EMIN": 0.0, "EMAX": 1e100,
                          "FMIN": 0.0, "FMAX": 1e100,
                          "INFO": None, "SKIP": False,
                          "TIME": 0.0, "PHASE": 0,
                          "PHA1": None, "PHA2": None,
                          "MODE": 1, "JUMP": [False, 0],
                          "FORMAT": "Unknown", "END": False}
            self.commands = []
            self.observatories = set() 
            self.obs = []              # Toa observatory id 
            self.freq = []             # Toa observation frequency
            self.error = []            # Toa error array

        with open(filename,"r") as f:
            for l in f.readlines():
                MJD, d = parse_TOA_line(l, fmt=self.cdict["FORMAT"]) 
               

                if d["format"] == "Command":
                    cmd = d["Command"][0]
                    self.commands.append((d["Command"], len(self.toas)))
                    if cmd == "SKIP":
                        self.cdict[cmd] = True
                        continue
                    elif cmd == "NOSKIP":
                        self.cdict["SKIP"] = False
                        continue
                    elif cmd == "END":
                        self.cdict[cmd] = True
                        break
                    elif cmd in ("TIME", "PHASE"):
                        self.cdict[cmd] += float(d["Command"][1])
                    elif cmd in ("EMIN", "EMAX", "EFAC", "EQUAD",\
                                 "PHA1", "PHA2", "FMIN", "FMAX"):
                        self.cdict[cmd] = float(d["Command"][1])
                        if cmd in ("PHA1", "PHA2", "TIME", "PHASE"):
                            d[cmd] = d["Command"][1]
                        elif cmd == "INFO":
                            self.cdict[cmd] = d["Command"][1]
                            d[cmd] = d["Command"][1]
                        elif cmd == "FORMAT":
                            if d["Command"][1] == "1":
                                self.cdict[cmd] = "Tempo2"
                        elif cmd == "JUMP":
                            if self.cdict[cmd][0]:
                                self.cdict[cmd][0] = False
                                self.cdict[cmd][1] += 1
                            else:
                                self.cdict[cmd][0] = True
                        elif cmd == "INCLUDE" and process_includes:  
                            # Save FORMAT in a tmp
                            fmt = self.cdict["FORMAT"]
                            self.cdict["FORMAT"] = "Unknown"
                            self.read_toa_file(d["Command"][1], top=False)
                            # re-set FORMAT
                            self.cdict["FORMAT"] = fmt        
                        else:
                            continue
                if (self.cdict["SKIP"] or
                    d["format"] in ("Blank", "Unknown", "Comment", "Command")):
                    continue
                elif self.cdict["END"]:                     
                    if top:
                    # Clearn up our temporaries uesd when reading TOAs
                        del self.cdict
                    return
                else:
                    self.toas.append(MJD)
                    self.obs.append(d['obs'])
                    self.freq.append(d['freq'])
                    self.error.append(d['error'])
        return      

    def get_table(self):
        """
        get_table() returns a data table in astropy talbe formate. The basic 
        table column includes:
        toa: 2D numpy array, toa integer part in float, toa fraction part in float
        freq: 1D numpy array, toa observation frequency
        obs: 1D string list, toa observatory ID
        error: 1D numpy array, toa error 
        """
        self.dataTable = table.Table([ numpy.array(self.toas,dtype = float),
                                     numpy.array(self.freq), self.obs, 
                                     numpy.array(self.error) ],
                                     names = ('toa_utc', 'freq', 'obs', 'error'),
                                     meta = {'filename':self.filename})   

        return
    def apply_clock_corrections(self):
        """Apply observatory clock corrections.
        
        Apply clock corrections to all the TOAs where corrections
        are available.  This routine actually changes
        the value of the TOA, although the correction is also listed
        as a new flag for the TOA called 'clkcorr' so that it can be
        reversed if necessary.  This routine also applies all 'TIME'
        commands and treats them exactly as if they were a part of the
        observatory clock corrections.
        """
        return



    def get_time_convert(self,clock_correction = False):
        """
        get_time_convert() takes utc time in mjd convert to different time
        formate. 
        clock_correction is an optional varible for apply observatory specified
        clock correction
        Step:
        1. Group data by observatory
        2. If clock_correction is true
           Apply clock_correction, 
           If not, 
           do nothing
        3. Creat time object with input of toa and time scale and formate 
           and observatory     
        """
        self.dataTable['tt'] = numpy.zeros((1,2))
        self.dataTable['tdb'] = numpy.zeros((1,2))
         
        self.dataTable = self.dataTable.group_by('obs')
        
        if clock_correction == True:
            pass
        
        for ii,obsname in  enumerate(self.dataTable.groups.keys._data):
            # set up astropy object
            obsname = obsname[0]    # make obsname a string not a (obsname,) 
            mask = self.dataTable.groups.keys['obs'] == obsname
            mjdtoa = time.Time(self.dataTable.groups[mask]['toa_utc'][:,0],\
                               self.dataTable.groups[mask]['toa_utc'][:,1], 
                               scale = 'utc', format='mjd',\
                               lon = observatories[obsname].geo[0],\
                               lat=observatories[obsname].geo[1],precision=9) 
                            
            setattr(self,'time_at_'+obsname,mjdtoa)       
            # add tt and tdb to table
            self.dataTable.groups[ii]['tt'][:,0] = mjdtoa.tt.jd1
            self.dataTable.groups[ii]['tt'][:,1] = mjdtoa.tt.jd2
            self.dataTable.groups[ii]['tdb'][:,0] = mjdtoa.tdb.jd1
            self.dataTable.groups[ii]['tdb'][:,1] = mjdtoa.tdb.jd2
        return





