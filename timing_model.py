from astropy.coordinates.angles import Angle

class Parameter(object):
    """
    Parameter(name=None, value=None, units=None, description=None, 
                uncertainty=None, frozen=True, aliases=[],
                parse_value=float, print_value=str)

        Class describing a single timing model parameter.  Takes the following
        inputs:

        name is the name of the parameter.

        value is the current value of the parameter.

        units is a string giving the units.

        description is a short description of what this parameter means.

        uncertainty is the current uncertainty of the value.

        frozen is a flag specifying whether "fitters" should adjust the
          value of this parameter or leave it fixed.

        aliases is an optional list of strings specifying alternate names
          that can also be accepted for this parameter.

        parse_value is a function that converts string input into the
          appropriate internal representation of the parameter (typically
          floating-point but could be any datatype).

        print_value is a function that converts the internal value to
          a string for output.

    """

    def __init__(self, name=None, value=None, units=None, description=None, 
            uncertainty=None, frozen=True, aliases=[],
            parse_value=float, print_value=str):
        self.value = value
        self.name = name
        self.units = units
        self.description = description
        self.uncertainty = uncertainty
        self.frozen = frozen
        self.aliases = aliases
        self.parse_value=parse_value
        self.print_value=print_value

    def __str__(self):
        return self.name + " (" + self.units + ") " \
            + self.print_value(self.value) + " +/- " + str(self.uncertainty)

    def set(self,value):
        """
        Parses a string 'value' into the appropriate internal representation
        of the parameter.
        """
        self.value = self.parse_value(value)

    def add_alias(self, alias):
        """
        Add a name to the list of aliases for this parameter.
        """
        aliases.append(alias)

    def as_parfile_line(self):
        """
        Return a parfile line giving the current state of the parameter.
        """
        line = "%-10s %25s" % (self.name, self.print_value(self.value))
        if self.uncertainty != None:
            line += " %d %s" % (0 if self.frozen else 1, str(self.uncertainty))
        elif not self.frozen:
            line += " 1" 
        return line

class TimingModel(object):

    def __init__(self):
        self.params = []  # List of model parameter names
        self.delay_funcs = [] # List of delay component functions
        self.phase_funcs = [] # List of phase component functions

    def add_param(self, param):
        setattr(self, param.name, param)
        self.params += [param.name,]

    def __add__(self, other):
        # Combine two timing model objects into one
        # TODO: How to deal with conflicts in names/etc...
        result = TimingModel()
        result.__dict__ = dict(self.__dict__.items() + other.__dict__.items())
        result.params = self.params + other.params
        result.phase_funcs = self.phase_funcs + other.phase_funcs
        result.delay_funcs = self.delay_funcs + other.delay_funcs
        return result

    def compute_phase(self, toa):
        """
        Compute the model-predicted pulse phase for the given toa.
        """
        # First compute the delay to "pulsar time"
        delay = self.compute_delay(toa)

        # Then compute the relevant pulse phase
        for pf in self.phase_funcs:
            phase += pf(toa - delay) # This is just a placeholder until we
                                     # define what datatype 'toa' has, and
                                     # how to add/subtract from it, etc.
        return phase

    def compute_delay(self, toa):
        """
        Compute the total delay which will be subtracted from the given
        TOA to get time of emission at the pulsar.
        """
        delay = 0.0
        for df in self.delay_funcs:
            delay += getattr(self,df)(toa)
        return delay

    def __str__(self):
        result = ""
        for par in self.params:
            result += str(getattr(self,par)) + "\n"
        return result

    def as_parfile(self):
        """
        Returns a parfile representation of the entire model as a string.
        """
        result = ""
        for par in self.params:
            result += getattr(self,par).as_parfile_line() + '\n'
        return result

    def read_parfile(self, filename):
        """
        Read values from the specified parfile into the model parameters.
        """
        pfile = open(filename,'r')
        for l in pfile.readlines():
            k = l.split()
            name = k[0]
            val = None
            fit = False
            err = None
            if len(k)>=2:
                val = k[1]
            if len(k)>=3:
                if int(k[2])>0: 
                    fit = True
            if len(k)==4:
                err = k[3]
            par_name = None
            if hasattr(self,name):
                par_name = name
            else:
                for par in self.params:
                    if name in getattr(self,par).aliases:
                        par_name = par
            if par_name:
                getattr(self,par_name).set(val)
                getattr(self,par_name).uncertainty = err
                getattr(self,par_name).frozen = not fit
            else:
                # unrecognized parameter, could
                # print a warning or something
                pass

class Astrometry(TimingModel):

    def __init__(self):
        TimingModel.__init__(self)

        self.add_param(Parameter(name="RA",
            units="H:M:S",
            description="Right ascension (J2000)",
            aliases=["RAJ"],
            parse_value=lambda x: Angle(x+'h').hour,
            print_value=lambda x: Angle(x,unit='h').to_string(sep=':', 
                precision=8)))

        self.add_param(Parameter(name="DEC",
            units="D:M:S",
            description="Declination (J2000)",
            aliases=["DECJ"],
            parse_value=lambda x: Angle(x+'deg').degree,
            print_value=lambda x: Angle(x,unit='deg').to_string(sep=':',
                alwayssign=True, precision=8)))

        # etc, also add PM, PX, ...

class Spindown(TimingModel):

    def __init__(self):
        TimingModel.__init__(self)

        self.add_param(Parameter(name="F0",
            units="Hz", 
            description="Spin frequency",
            aliases=["F"],
            print_value=lambda x: '%.15f'%x))

        self.add_param(Parameter(name="F1",
            units="Hz/s", 
            description="Spin-down rate"))

        self.add_param(Parameter(name="TZRMJD",
            units="MJD", 
            description="Reference epoch for phase"))

        self.add_param(Parameter(name="PEPOCH",
            units="MJD", 
            description="Reference epoch for spin-down"))

    def simple_spindown_phase(self,toa):
        """
        Placeholder function for simple spindown phase.
        Still need to figure out data types for toa, mjd, make
        sure the right precision is in use, etc.  This is just here
        to show the structure of how this should work.

        Also still need to figure out a straightforward way to get
        this function into the generic TimingModel class as required.
        """
        # If TZRMJD is not defined, use the toa itself
        if self.TZRMJD.value==None:
            self.TZRMJD.value = toa
        dt = toa - self.TZRMJD.value
        phase = dt*self.F0.value + 0.5*dt*dt*self.F1.value
        return phase
