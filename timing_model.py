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

        frozen is a flag specifying whether "fitters" should modify the
          value or leave it fixed.

        aliases is an optional list of strings specifying alternate names
          that can also be accepted for this parameter.

        parse_value is a function that converts string input into the
          appropriate internal representation of the parameter (typically
          floating-point).

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

    def parfile_line(self):
        """
        Return a parfile line giving the current state of the parameter.
        """
        line = "%-10s %25s %d" % (self.name, self.print_value(self.value),
                0 if self.frozen else 1)  
        if self.uncertainty != None:
            line += " %s" % (str(self.uncertainty))
        return line

class TimingModel(object):

    def __init__(self):
        self.params = []

    def add_param(self, param):
        setattr(self, param.name, param)
        self.params += [param.name,]

    def __add__(self, other):
        # Combine two timing model objects into one
        # TODO: Check for conflicts in parameter names?
        result = TimingModel()
        result.__dict__ = dict(self.__dict__.items() + other.__dict__.items())
        result.params = self.params + other.params
        return result

    def __str__(self):
        result = ""
        for par in self.params:
            result += str(getattr(self,par)) + "\n"
        return result

    def as_parfile(self):
        """
        Returns a parfile representation of the entire mode as a string.
        """
        result = ""
        for par in self.params:
            result += getattr(self,par).parfile_line() + '\n'
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
            print_value=lambda x: Angle(x,unit='h').to_string(sep=':')))

        self.add_param(Parameter(name="DEC",
            units="D:M:S",
            description="Declination (J2000)",
            aliases=["DECJ"],
            parse_value=lambda x: Angle(x+'deg').degree,
            print_value=lambda x: Angle(x,unit='deg').to_string(sep=':')))

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


