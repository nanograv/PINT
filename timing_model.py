
class Parameter(object):

    def __init__(self, name=None, value=None, units=None, description=None, 
            uncertainty=None, frozen=True, aliases=[]):
        self.value = value
        self.name = name
        self.units = units
        self.description = description
        self.uncertainty = uncertainty
        self.frozen = frozen
        self.aliases = aliases

    def __str__(self):
        return self.name + " (" + self.units + ") value=" \
            + str(self.value) + "+/-" + str(self.uncertainty)

    def add_alias(self, alias):
        aliases.append(alias)

class TimingModel(object):

    def __init__(self, filename=None):
        pass

    def add_param(self, param):
        setattr(self, param.name, param)

    def __add__(self, other):
        # Combine two timing model objects into one
        # TODO: Check for conflicts in parameter names?
        result = TimingModel()
        result.__dict__ = dict(self.__dict__.items() + other.__dict__.items())
        return result

    def __str__(self):
        result = ""
        for par in self.__dict__.keys():
            result += str(getattr(self,par)) + "\n"
        return result

    def read_parfile(self, filename):
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
                for par in self.__dict__.keys():
                    if name in getattr(self,par).aliases:
                        par_name = par
            if par_name:
                getattr(self,par_name).value = val
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
            aliases=["RAJ"]))

        self.add_param(Parameter(name="DEC",
            units="D:M:S",
            description="Declination (J2000)",
            aliases=["DECJ"]))

        # etc, also add PM, PX, ...

class Spindown(TimingModel):

    def __init__(self):
        TimingModel.__init__(self)

        self.add_param(Parameter(name="F0",
            units="Hz", 
            description="Spin frequency",
            aliases=["F"]))

        self.add_param(Parameter(name="F1",
            units="Hz/s", 
            description="Spin-down rate"))


