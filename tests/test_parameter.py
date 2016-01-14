from pint.models import parameter as p
from pint.models import model_builder as mb
from astropy.coordinates.angles import Angle
from pint.utils import time_from_mjd_string, time_to_longdouble, str2longdouble,\
                       longdouble2string

# Test parameter Class

param1 = p.Parameter(name="RAJ",
            units="H:M:S",
            description="Right ascension (J2000)",
            aliases=["RAJ"],
            parse_value=lambda x: Angle(x+'h'),
            print_value=lambda x: x.to_string(sep=':',precision=8))
print "Before assigning param1 value."
print "param1 bare value is ", param1.bare_value
print "param1 value is      ", param1.value,"With type ", type(param1.value)
print "param1 unit is       ", param1.units
print "param1 base unit is  ", param1.base_unit

# change param1
param1.set('19:30:52')
print "After assigning param1 value."
print "param1 bare value is ", param1.bare_value
print "param1 value is      ", param1.value,"With type ", type(param1.value)
print "param1 unit is       ", param1.units
print "param1 base unit is  ", param1.base_unit

# Test MJDparameter Class
paramMJD = p.MJDParameter(name="TZRMJD",
                        description="Reference epoch for phase = 0.0",
                        parse_value=lambda x: time_from_mjd_string(x, scale='tdb'))
print "Before assigning paramMJD value."
print "paramMJD bare value is ", longdouble2string(paramMJD.bare_value)
print "paramMJD value is      ", paramMJD.value,"With type ", type(paramMJD.value)
print "paramMJD unit is       ", paramMJD.units
print "paramMJD base unit is  ", paramMJD.base_unit
paramMJD.set('54001.012345678901234')
print "After assigning paramMJD value."
print "paramMJD bare value is ", longdouble2string(paramMJD.bare_value)
print "paramMJD value is      ", paramMJD.value, "With type ", type(paramMJD.value)
print "paramMJD unit is       ", paramMJD.units
print "paramMJD base unit is  ", paramMJD.base_unit

model = mb.get_model('J1744-1134.basic.par')

for pn in model.params:
    p = getattr(model,pn)
    print "Parameter name :", p.name
    print "Value is ", p.value, "Bare_value is ",p.bare_value
    print "Unit is ", p.units, "Base unit is ", p.base_unit
