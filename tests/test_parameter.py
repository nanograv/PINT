from pint.models import parameter as p
from pint.models import model_builder as mb
from astropy.coordinates.angles import Angle
from pint.utils import time_from_mjd_string, time_to_longdouble, str2longdouble,\
                       longdouble2string,data2longdouble

# Test parameter Class

param1 = p.Parameter(name="RAJ",
            units="H:M:S",
            description="Right ascension (J2000)",
            aliases=["RAJ"],
            parse_value=lambda x: Angle(x+'h'),
            print_value=lambda x: x.to_string(sep=':',precision=8))
print "Before assigning param1 value."
print "param1 num value is ", param1.num_value
print "param1 value is      ", param1.value,"With type ", type(param1.value)
print "param1 unit is       ", param1.units
print "param1 num unit is  ", param1.num_unit

# change param1
param1.set('19:30:52')
print "After assigning param1 value."
print "param1 num value is ", param1.num_value
print "param1 value is      ", param1.value,"With type ", type(param1.value)
print "param1 unit is       ", param1.units
print "param1 num unit is  ", param1.num_unit
# Change num_value
print "Change num_values"
param1.num_value = 10.0
print "param1 num value is ", param1.num_value
print "param1 value is      ", param1.value,"With type ", type(param1.value)
print "param1 unit is       ", param1.units
print "param1 num unit is  ", param1.num_unit
# Test MJDparameter Class
paramMJD = p.MJDParameter(name="TZRMJD",
                        description="Reference epoch for phase = 0.0",
                        parse_value=lambda x: time_from_mjd_string(x, scale='tdb'))
print "Before assigning paramMJD value."
print "paramMJD num value is ", longdouble2string(paramMJD.num_value)
print "paramMJD value is      ", paramMJD.value,"With type ", type(paramMJD.value)
print "paramMJD unit is       ", paramMJD.units
print "paramMJD num unit is  ", paramMJD.num_unit
paramMJD.set('54001.012345678901234')
print "After assigning paramMJD value."
print "paramMJD num value is ", longdouble2string(paramMJD.num_value)
print "paramMJD value is      ", paramMJD.value, "With type ", type(paramMJD.value)
print "paramMJD unit is       ", paramMJD.units
print "paramMJD num unit is  ", paramMJD.num_unit
#change num_value
paramMJD.num_value = data2longdouble('54001.012345678901234')
print "After assigning paramMJD num_value."
print "paramMJD num value is ", longdouble2string(paramMJD.num_value)
print "paramMJD value is      ", paramMJD.value, "With type ", type(paramMJD.value)
print "paramMJD unit is       ", paramMJD.units
print "paramMJD num unit is  ", paramMJD.num_unit

print '\n\n\n\n'
model = mb.get_model('J1744-1134.basic.par')

for pn in model.params:
    p = getattr(model,pn)
    print "Parameter name :", p.name
    print "Value is ", p.value, "num_value is ",p.num_value
    print "Unit is ", p.units, "num unit is ", p.num_unit
