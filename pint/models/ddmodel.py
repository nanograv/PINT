import numpy as np
import functools
import collections
from astropy import log
from .timing_model import Cache
from scipy.optimize import newton

class DDmdoel(object):
	"""A frist test of DD model. 
	   See T. Damour and N. Deruelle Annales de l' I.H.P.(1986)
	   Parameters
	   @n
	   @PB
	   @ECC
	   @T0
	   @eT
	   @ar 
	   @sini
	   @omg0
	   @k
	   @deltaR
	   @deltaTheta
       @gamma
       @mpr
       @A
       @B
	"""
	def __init__(self,):
		self.binaryName = 'DD'
		self.PB = 0.0
		self.ECC = 0.0
		self.T0 = 0.0
    def ecc_anmls(self,E):
    	E-self.ECC*np.sin(E) - 2*np.pi*(self.t-self.T0)/self.PB
    def drv_ecc_anmls(self,E):
    	1-self.Ecc*np.cos(E)
	def compute_eccentric_anomalies(self,t):
        self.t = t
        Eest = 2*np.pi*(self.t-self.T0)/self.PB
        E = netwon.(self.ecc_anmls,Eest, maxiter=500)
	def binary_shapiro_delay(self):
		pass
	def binary_E_delay(self):
		pass 
