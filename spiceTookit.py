import spice
import sys
import numpy as np

###### load kernels  #######
def loadKernel(fileName):  # fileName is a list of all the kernels with directoryneeded
	for name in fileName:
		if(fileName!=[]):
			spice.furnsh(name)     # Load the kernels
		else:
			print "No Kernels has been loaded."
		

def objPosVel2SSB(objName,et):
# Returns a solar system object position and velocity in J2000 ssb coordinate
# Requires SPK,LPSK kernels
# In J2000 ssb coordinate 
	objName.upper()    # Make the object name upper level
	state,lt = spice.spkzer(objName,et,"J2000","NONE","SSB")   
	# Reading the position and velocity for a solar system object
	# CALL SPKEZR(TARGET,ET,COORDINATE,CORRECTION,OBSERVOR)
	# State is a vector. First three elements are position [x,y,z] in J2000 ssb
	# Second three elements are velocity [dx/dt,dy/dt,dz/dt] in j2000
	# Lt is the light time
	
	return state,lt

def getobsJ2000(posITRF,et):
# Returns observatory rectangular coordinates in J2000 Earth centered coordinate
# Requires PCK kernels.
# posITRF is a double vector of [x,y,z] in ITRF coordinate in km
	state = np.array(posITRF+[0,0,0])
	xform = spice.sxform("ITRF93","J2000",et) 
	# Transformation matrix from ITRF93 to J2000
	# CALL SXFORM(COORDINATE FROM, COORDINATE TO, ET)
	xform = np.matrix(xform)
	jstate = np.dot(xform,state)
	# Coordinate transformation.
	# jstate is a vector. First three elements are position [x,y,z] in J2000 	earth centered
        # Second three elements are velocity [dx/dt,dy/dt,dz/dt] in j2000 earth centered 
	return jstate


def ITRF2GEO(posITRF):
	'''
	Converts from earth rectangular coordinate to Geodetic coordinate, In put will be the rectangular three coordinates [x,y,z]
	Kernerl file PCK is required .
	'''
	dim,value = spice.bodvcd(399, "RADII", 3)
	# Reuturns Earh radii [larger equatorial radius,smaller equatorial radius,polar radius]
	# dim is the dimension of returned values
	# Value is the returned values
 	rEquatr   =  value[0]; 
      	rPolar    =  value[2];
	print rEquatr,rPolar
	 # Calculate the flattening factor for earth 
        f = (rEquatr - rPolar) / rEquatr
	# Calculate the geodetic coordinate on earth. lon,lat are in radius. alt is the same unit with in put posITRF
	lon,lat,alt = spice.recgeo(posITRF,rEquatr,f)
	# Return longitude and latitude in degree
	lonDeg = spice.convrt(lon, "RADIANS", "DEGREES")
	latDeg = spice.convrt(lat, "RADIANS", "DEGREES")
	return lon,lonDeg,lat,latDeg,alt

DDDDD

	


 
