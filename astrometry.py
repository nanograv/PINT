# astrometry.py
# Defines Astrometry timing model class
import astropy
from astropy.coordinates.angles import Angle
from timing_model import Parameter, TimingModel

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
