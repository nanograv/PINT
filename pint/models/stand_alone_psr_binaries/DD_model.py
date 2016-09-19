from .binary_generic import PSR_BINARY
from pint.models.timing_model import Cache
import numpy as np
import astropy.units as u
import astropy.constants as c
from pint import ls,GMsun,Tsun


class DDmodel(PSR_BINARY):
    """This is a class independent from PINT platform for pulsar DD binary model.
    Refence: T. Damour and N. Deruelle(1986)
    It is a subclass of PSR_BINARY class defined in file binary_generic.py in
    the same dierectory. This class is desined for PINT platform but can be used
    as an independent module for binary delay calculation.
    To interact with PINT platform, a pulsar_binary wrapper is needed.
    See the source file pint/models/pulsar_binary_dd.py

    Return
    ----------
    A dd binary model class with paramters, delay calculations and derivatives.
    Example
    ----------
    >>> import numpy
    >>> t = numpy.linspace(54200.0,55000.0,800)
    >>> binary_model = DDmodel()
    >>> paramters_dict = {'A0':0.5,'ECC':0.01}
    >>> binary_model.update_input(t, paramters_dict)
    Here the binary has time input and parameters input, the delay can be
    calculated.
    """
    def __init__(self, t=None, input_params=None):
        super(DDmodel, self).__init__()
        self.binary_name = 'DD'
        # Add parameter that specific for DD model, with default value and units
        self.param_default_value.update({'A0':0*u.second,'B0':0*u.second,
                               'DR':0*u.Unit(''),'DTH':0*u.Unit(''),
                               'GAMMA':0*u.second,})
        # If any parameter has aliases, it should be updated
        #self.param_aliases.update({})
        self.binary_params = self.param_default_value.keys()

        self.dd_interVars = ['er','eTheta','beta','alpha','Dre','Drep','Drepp',
                             'nhat', 'TM2']
        self.add_inter_vars(self.dd_interVars)
        self.set_param_values() # Set parameters to default values.
        self.binary_delay_funcs += [self.DDdelay]
        self.d_binarydelay_d_par_funcs += [self.d_DDdelay_d_par]
        if t is not None:
            self.t = t
        if input_params is not None:
            self.update_input(param_dict=input_params)
    # calculations for delays in DD model
    # Calculate er
    @Cache.cache_result
    def er(self):
        return self.ecc()+self.DR

    @Cache.cache_result
    def d_er_d_DR(self):
        return np.longdouble(np.ones(len(self.tt0)))*u.Unit("")

    @Cache.cache_result
    def d_er_d_par(self,par):
        if par not in self.binary_params:
            errorMesg = par + "is not in binary parameter list."
            raise ValueError(errorMesg)

        if par in ['DR']:
            dername = 'd_er_d_'+par
            return getattr(self,dername)()
        else:
            dername = 'd_ecc_d_'+par
            if hasattr(self,dername):
                return getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.tt0)))

    ##########
    @Cache.cache_result
    def eTheta(self):
        return self.ecc()+self.DTH
    @Cache.cache_result
    def d_eTheta_d_DTH(self):
        return np.longdouble(np.ones(len(self.tt0)))*u.Unit("")
    @Cache.cache_result
    def d_eTheta_d_par(self,par):
        if par not in self.binary_params:
            errorMesg = par + "is not in parameter list."
            raise ValueError(errorMesg)

        if par in ['DTH']:
            dername = 'd_eTheta_d_'+par
            return getattr(self,dername)()
        else:
            dername = 'd_ecc_d_'+par
            if hasattr(self,dername):
                return getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.tt0)))
    ##########
    @Cache.cache_result
    def alpha(self):
        """Alpha defined in
           T. Damour and N. Deruelle(1986)equation [46]
           alpha = A1/c*sin(omega)
        """
        sinOmg = np.sin(self.omega())
        return self.a1()/c.c*sinOmg

    @Cache.cache_result
    def d_alpha_d_par(self,par):
        """T. Damour and N. Deruelle(1986)equation [46]
           alpha = a1/c*sin(omega)

           dAlpha/dA1 = 1.0/c*sin(omega)

           dAlpha/dPar = a1/c*cos(omega)*dOmega/dPar
        """

        if par not in self.binary_params:
            errorMesg = par + "is not in binary parameter list."
            raise ValueError(errorMesg)

        if par in ['A1','A1DOT']:
            dername = 'd_alpha_d_'+par
            return getattr(self,dername)()

        else:
            dername = 'd_omega_d_'+par # For parameters only in Ae
            if hasattr(self,dername):
                cosOmg=np.cos(self.omega())
                return self.a1()/c.c*cosOmg*getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.tt0)))

    @Cache.cache_result
    def d_alpha_d_A1(self):
        sinOmg = np.sin(self.omega())
        return 1.0/c.c*sinOmg

    @Cache.cache_result
    def d_alpha_d_A1DOT(self):
        sinOmg = np.sin(self.omega())
        return self.tt0/c.c*sinOmg
    ##############################################

    @Cache.cache_result
    def beta(self):
        """Beta defined in
           T. Damour and N. Deruelle(1986)equation [47]
           beta = A1/c*(1-eTheta**2)**0.5*cos(omega)
        """
        eTheta = self.eTheta()
        cosOmg = np.cos(self.omega())
        return self.a1()/c.c*(1-eTheta**2)**0.5*cosOmg

    @Cache.cache_result
    def d_beta_d_par(self,par):
        """beta = A1/c*(1-eTheta**2)**0.5*cos(omega)
           eTheta = ecc+Dth  ??
           dBeta/dA1 = 1.0/c*(1-eTheta**2)**0.5*cos(omega)
           dBeta/dECC = A1/c*((-(e+dr)/sqrt(1-(e+dr)**2)*cos(omega)*de/dECC-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dECC
           dBeta/dEDOT = A1/c*((-(e+dr)/sqrt(1-(e+dr)**2)*cos(omega)*de/dEDOT-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dEDOT
           dBeta/dDth = A1/c*(-(e+dr)/sqrt(1-(e+dr)**2)*cos(omega)
           Other parameters
           dBeta/dPar = -A1/c*(1-eTheta**2)**0.5*sin(omega)*dOmega/dPar
        """
        if par not in self.binary_params:
            errorMesg = par + "is not in binary parameter list."
            raise ValueError(errorMesg)

        if par in ['A1','ECC','EDOT','DTH','A1DOT']:
            dername = 'd_beta_d_'+par
            return getattr(self,dername)()

        else:
            dername = 'd_omega_d_'+par # For parameters only in omega
            if hasattr(self,dername):
                eTheta = self.eTheta()
                a1 = self.a1()
                sinOmg = np.sin(self.omega())
                return -a1/c.c*(1-eTheta**2)**0.5*sinOmg*getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.tt0)))

    @Cache.cache_result
    def d_beta_d_A1(self):
        """dBeta/dA1 = 1.0/c*(1-eTheta**2)**0.5*cos(omega)
        """
        eTheta = self.eTheta()
        cosOmg = np.cos(self.omega())
        return 1.0/c.c*(1-eTheta**2)**0.5*cosOmg

    @Cache.cache_result
    def d_beta_d_A1DOT(self):
        """dBeta/dA1DOT = (t-T0)/c*(1-eTheta**2)**0.5*cos(omega)
        """
        eTheta = self.eTheta()
        cosOmg = np.cos(self.omega())
        return self.tt0/c.c*(1-eTheta**2)**0.5*cosOmg

    @Cache.cache_result
    def d_beta_d_ECC(self):
        """dBeta/dECC = A1/c*((-(e+dtheta)/sqrt(1-(e+dtheta)**2)*cos(omega)*de/dECC-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dECC
           de/dECC = 1
        """
        eTheta = self.eTheta()
        a1 = (self.a1()).decompose()
        sinOmg = np.sin(self.omega())
        cosOmg = np.cos(self.omega())
        return a1/c.c*((-eTheta)/np.sqrt(1-eTheta**2)*cosOmg- \
               (1-eTheta**2)**0.5*sinOmg*self.d_omega_d_par('ECC'))

    @Cache.cache_result
    def d_beta_d_EDOT(self):
        """dBeta/dEDOT = A1/c*((-(e+dtheta)/sqrt(1-(e+dtheta)**2)*cos(omega)*de/dEDOT- \
           (1-eTheta**2)**0.5*sin(omega)*domega/dEDOT
           de/dEDOT = tt0
        """
        eTheta = self.eTheta()
        a1 = (self.a1()).decompose()
        sinOmg = np.sin(self.omega())
        cosOmg = np.cos(self.omega())
        return a1/c.c*((-eTheta)/np.sqrt(1-eTheta**2)*cosOmg*self.tt0- \
               (1-eTheta**2)**0.5*sinOmg*self.d_omega_d_par('EDOT'))

    @Cache.cache_result
    def d_beta_d_DTH(self):
        """dBeta/dDth = a1/c*((-(e+dr)/sqrt(1-(e+dr)**2)*cos(omega)
        """
        eTheta = self.eTheta()
        cosOmg = np.cos(self.omega())

        return self.a1()/c.c*(-eTheta)/np.sqrt(1-eTheta**2)*cosOmg

    ##################################################
    @Cache.cache_result
    def Dre(self):
        """Dre defined in
           T. Damour and N. Deruelle(1986)equation [48]
           Dre = alpha*(cos(E)-er)+(beta+gamma)*sin(E)
        """
        er = self.er()
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        return self.alpha()*(cosE- er)+   \
               (self.beta()+self.GAMMA)*sinE

    @Cache.cache_result
    def d_Dre_d_par(self,par):
        """Dre = alpha*(cos(E)-er)+(beta+gamma)*sin(E)
           dDre = alpha*(-der-dE*sin(E)) + (cos[E]-er)*dalpha +
                  (dBeta+dGamma)*sin(E) + (beta+gamma)*cos(E)*dE

           dDre/dpar = alpha*(-der/dpar-dE/dpar*sin(E)) +
                       (cos[E]-er)*dalpha/dpar +
                       (dBeta/dpar+dGamma/dpar)*sin(E) +
                       (beta+gamma)*cos(E)*dE/dpar

            er = e + Dr
        """
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        # First term
        term1 = self.alpha()*(-self.prtl_der('er',par)-self.prtl_der('E',par)*sinE)
        # Second term
        term2 = (cosE-self.er())*self.prtl_der('alpha',par)
        # Third term
        term3 = (self.prtl_der('beta',par)+self.prtl_der('GAMMA',par))*sinE
        # Fourth term
        term4 = (self.beta()+self.GAMMA)*cosE*self.prtl_der('E',par)

        return term1 + term2 + term3 +term4

    #################################################
    @Cache.cache_result
    def Drep(self):
        """Dervitive of Dre respect to E
           T. Damour and N. Deruelle(1986)equation [49]
           Drep = -alpha*sin(E)+(beta+gamma)*cos(E)
        """
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        return -self.alpha()*sinE+(self.beta()+self.GAMMA)*cosE

    @Cache.cache_result
    def d_Drep_d_par(self,par):
        """Drep = -alpha*sin(E)+(beta+gamma)*cos(E)
           dDrep = -alpha*cos(E)*dE + cos(E)*(dbeta+dgamma)
                   -(beta+gamma)*dE*sin(E)-dalpha*sin(E)
           dDrep/dPar = -sin(E)*dalpha/dPar
                        -(alpha*cos(E)+(beta+gamma)*sin(E))*dE/dPar
                        + cos(E)(dbeta/dPar+dgamma/dPar)

        """
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        # first term
        term1 = -sinE*self.prtl_der('alpha',par)
        # second term
        term2 = -(self.alpha()*cosE+  \
                (self.beta()+self.GAMMA)*sinE)*self.prtl_der('E',par)
        # Third term
        term3 = cosE*(self.prtl_der('beta',par)+self.prtl_der('GAMMA',par))

        return term1+term2+term3

    #################################################
    @Cache.cache_result
    def Drepp(self):
        """Dervitive of Drep respect to E
           T. Damour and N. Deruelle(1986)equation [50]
           Drepp = -alpha*cos(E)-(beta+GAMMA)*sin(E)
        """
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        return -self.alpha()*cosE-(self.beta()+self.GAMMA)*sinE

    @Cache.cache_result
    def d_Drepp_d_par(self,par):
        """Drepp = -alpha*cos(E)-(beta+GAMMA)*sin(E)
           dDrepp = -(beta+gamma)*cos(E)*dE - cos(E)*dalpha
                    +alpha*sin(E)*dE - (dbeta+dgamma)*sin(E)

           dDrepp/dPar = -cos(E)*dalpha/dPar
                         +(alpha*sin(E)-(beta+gamma)*cos(E))*dE/dPar
                         -(dbeta/dPar+dgamma/dPar)*sin(E)
        """
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        # first term
        term1 = -cosE*self.prtl_der('alpha',par)
        # second term
        term2 = (self.alpha()*sinE -  \
                (self.beta()+self.GAMMA)*cosE)*self.prtl_der('E',par)
        # Third term
        term3 = -sinE*(self.prtl_der('beta',par)+self.prtl_der('GAMMA',par))

        return term1+term2+term3
    #################################################

    @Cache.cache_result
    def nhat(self):
        """nhat defined as
           T. Damour and N. Deruelle(1986)equation [51]
           nhat = n/(1-ecc*cos(E))
           n = 2*pi/PB # should here be M()
        """
        cosE = np.cos(self.E())
        return 2.0*np.pi/self.PB.to('second')/(1-self.ecc()*cosE)

    @Cache.cache_result
    def d_nhat_d_par(self,par):
        """nhat = n/(1-ecc*cos(E))
           n = 2*pi/PB # should here be M()?
           dnhat = -2*pi*dPB/PB^2*(1-ecc*cos(E))
                   -2*pi*(-cos(E)*decc+ecc*sin(E)*dE)/PB*(1-ecc*cos(E))^2

           dnhat/dPar = -2*pi/(PB*(1-ecc*cos(E))*((dPB/dPar)/PB -
                        (-cos(E)*decc/dPar+ecc*sin(E)*dE/dpar)/(1-e*cos(E)))
        """
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        oneMeccTcosE = (1-self.ecc()*cosE)
        fctr = -2*np.pi/self.PB/oneMeccTcosE

        return fctr*(self.prtl_der('PB',par)/self.PB - \
               (cosE*self.prtl_der('ecc',par)- \
                self.ecc()*sinE*self.prtl_der('E',par))/oneMeccTcosE)

    #################################################
    @Cache.use_cache
    def delayInverse(self):
        """DD model Inverse timing delay.
        T. Damour and N. Deruelle(1986)equation [46-52]
        This part is convert the delay argument from proper time to coordinate
        time. The Romoer delay and Einstein are included in the calculation.
        It uses there iterations to approximate the Roemer delay and Einstein
        delay.
        T. Damour and N. Deruelle(1986)equation [43]. The equation [52] gives a
        taylor expension of equation [43].
        u - e*sin(u) = n(t-T0)
        nhat = du/dt
        nhatp  = d^2u/dt^2
        Drep = dDre/du
        Drepp = d^2Dre/du^2
        Dre(t-Dre(t-Dre(t)))  =  Dre(u) - Drep(u)*nhat*Dre(t-Dre(t))
                              =  Dre(u) - Drep(u)*nhat*(Dre(u)-Drep(u)*nhat*Dre(t))
                                 + 1/2 (Drepp(u)*nhat^2 + Drep(u) * nhat * nhatp) * (Dre(t)-...)^2
                              = Dre(t)*(1 - nhat * Drep(u) + (nhat*Drep)^2 +
                                1/2*nhat^2* Dre*Drepp - 1/2*e*sin(u)/(1-e*cos(u)*nhat^2*Drep*Drep))
        Here u is equivalent to E in the function.
        """
        Dre = self.Dre()
        Drep = self.Drep()
        Drepp = self.Drepp()
        nHat = self.nhat()
        e = self.ecc()
        sinE = np.sin(self.E())
        cosE = np.cos(self.E())
        return (Dre*(1-nHat*Drep+(nHat*Drep)**2+1.0/2*nHat**2*Dre*Drepp-\
                1.0/2*e*sinE/(1-e*cosE)*nHat**2*Dre*Drep)).decompose()

    @Cache.use_cache
    def d_delayI_d_par(self,par):
        """Derivative on delay inverse.
        """
        e = self.ecc()
        sE = np.sin(self.E())
        cE = np.cos(self.E())
        dE_dpar = self.prtl_der('E',par)
        decc_dpar = self.prtl_der('ecc',par)

        Dre = self.Dre()
        Drep = self.Drep()
        Drepp = self.Drepp()
        nHat = self.nhat()
        delayI = self.delayInverse()

        dDre_dpar = self.d_Dre_d_par(par)
        dDrep_dpar = self.d_Drep_d_par(par)
        dDrepp_dpar = self.d_Drepp_d_par(par)
        dnhat_dpar = self.d_nhat_d_par(par)
        oneMeccTcosE = (1-e*cE) # 1-e*cos(E)

        x =  -1.0/2.0*e*sE/oneMeccTcosE # -1/2*e*sin(E)/(1-e*cos(E))

        dx_dpar = -sE/(2*oneMeccTcosE**2)*decc_dpar+e*(e-cE)/(2*oneMeccTcosE**2)*dE_dpar

        diDelay_dDre = 1+(Drep*nHat)**2+Dre*Drepp*nHat**2+Drep*nHat*(2*Dre*nHat*x-1)
        diDelay_dDrep = Dre*nHat*(2*Drep*nHat+Dre*nHat*x-1)
        diDelay_dDrepp = (Dre*nHat)**2/2
        diDelay_dnhat = Dre*(-Drep+2*Drep**2*nHat+nHat*Dre*Drepp+2*x*nHat*Dre*Drep)
        diDelay_dx = (Dre*nHat)**2*Drep

        return dDre_dpar*diDelay_dDre + dDrep_dpar*diDelay_dDrep + \
               dDrepp_dpar*diDelay_dDrepp + dx_dpar*diDelay_dx+ \
               dnhat_dpar*diDelay_dnhat

    #################################################
    @Cache.use_cache
    def delayS(self):
        """Binary shapiro delay
           T. Damour and N. Deruelle(1986)equation [26]
        """
        e = self.ecc()
        cE = np.cos(self.E())
        sE = np.sin(self.E())
        sOmega = np.sin(self.omega())
        cOmega = np.cos(self.omega())
        TM2 = self.M2.value*Tsun

        sDelay = -2*TM2* np.log(1-e*cE-self.SINI*(sOmega*(cE-e)+
                 (1-e**2)**0.5*cOmega*sE))
        return sDelay

    @Cache.use_cache
    def d_delayS_d_par(self,par):
        """dsDelay/dPar = dsDelay/dTM2*dTM2/dPar+
                          dsDelay/decc*decc/dPar+
                          dsDelay/dE*dE/dPar+
                          dsDelay/domega*domega/dPar+
                          dsDelay/dSINI*dSINI/dPar
        """
        e = self.ecc()
        cE = np.cos(self.E())
        sE = np.sin(self.E())
        sOmega = np.sin(self.omega())
        cOmega = np.cos(self.omega())
        TM2 = self.M2.value*Tsun

        logNum = 1-e*cE-self.SINI*(sOmega*(cE-e)+
                 (1-e**2)**0.5*cOmega*sE)
        dTM2_dpar = self.prtl_der('TM2',par)
        dsDelay_dTM2 = -2*np.log(logNum)
        decc_dpar = self.prtl_der('ecc',par)
        dsDelay_decc = -2*TM2/logNum*(-cE-self.SINI*(-e*cOmega*sE/np.sqrt(1-e**2)-sOmega))
        dE_dpar = self.prtl_der('E',par)
        dsDelay_dE =  -2*TM2/logNum*(e*sE-self.SINI*(np.sqrt(1-e**2)*cE*cOmega-sE*sOmega))
        domega_dpar = self.prtl_der('omega',par)
        dsDelay_domega = -2*TM2/logNum*self.SINI*((cE-e)*cOmega-np.sqrt(1-e**2)*sE*sOmega)
        dSINI_dpar = self.prtl_der('SINI',par)
        dsDelay_dSINI = -2*TM2/logNum*(-np.sqrt(1-e**2)*cOmega*sE-(cE-e)*sOmega)
        return dTM2_dpar*dsDelay_dTM2 + decc_dpar*dsDelay_decc + \
               dE_dpar*dsDelay_dE +domega_dpar*dsDelay_domega +  \
               dSINI_dpar*dsDelay_dSINI

    #################################################
    @Cache.use_cache
    def delayE(self):
        """Binary Einstein delay
            T. Damour and N. Deruelle(1986)equation [25]
        """
        sinE = np.sin(self.E())
        return self.GAMMA

    @Cache.use_cache
    def d_delayE_d_par(self,par):
        """eDelay = gamma*sin[E]
           deDelay_dPar = deDelay/dgamma*dgamma/dPar +
                          deDelay/dE*dE/dPar
        """
        cE = np.cos(self.E())
        sE = np.sin(self.E())

        return sE*self.prtl_der('GAMMA',par)+self.GAMMA*cE*self.prtl_der('E',par)
    #################################################

    @Cache.use_cache
    def delayA(self):
        """Binary Abberation delay
            T. Damour and N. Deruelle(1986)equation [27]
        """
        omgPlusAe = self.omega()+self.nu()
        et = self.ecc()
        sinOmg = np.sin(self.omega())
        cosOmg = np.cos(self.omega())
        aDelay = self.A0*(np.sin(omgPlusAe)+et*sinOmg)+\
                 self.B0*(np.cos(omgPlusAe)+et*cosOmg)
        return aDelay

    @Cache.use_cache
    def d_delayA_d_par(self,par):
        """aDelay = A0*(sin(omega+E)+e*sin(omega))+B0*(cos(omega+E)+e*cos(omega))
           daDelay/dpar = daDelay/dA0*dA0/dPar+     (1)
                          daDelay/dB0*dB0/dPar+     (2)
                          daDelay/domega*domega/dPar+    (3)
                          daDelay/dnu*dnu/dPar+        (4)
                          daDelay/decc*decc/dPar        (5)
        """
        e = self.ecc()
        sOmega = np.sin(self.omega())
        cOmega = np.cos(self.omega())
        snu = np.sin(self.nu())
        cnu = np.cos(self.nu())
        A0 = self.A0
        B0 = self.B0
        omgPlusAe = self.omega()+self.nu()
        if par =='A0':
            return e*sOmega+np.sin(omgPlusAe)
        elif par == 'B0':
            return e*cOmega+np.cos(omgPlusAe)
        else:
            domega_dpar = self.prtl_der('omega',par)
            daDelay_domega = A0*(np.cos(omgPlusAe)+e*cOmega)- \
                             B0*(np.sin(omgPlusAe)+e*sOmega)

            dnu_dpar = self.prtl_der('nu',par)
            daDelay_dnu = A0*np.cos(omgPlusAe)-B0*np.sin(omgPlusAe)

            decc_dpar = self.prtl_der('ecc',par)
            daDelay_decc = A0*sOmega+B0*cOmega

            return domega_dpar*daDelay_domega + dnu_dpar*daDelay_dnu + decc_dpar*daDelay_decc
    #################################################

    @Cache.use_cache
    def DDdelay(self):
        """Full DD model delay"""
        return self.delayInverse()+self.delayS()+self.delayA()

    @Cache.use_cache
    def d_DDdelay_d_par(self,par):
        """Full DD model delay derivtive
        """
        return self.d_delayI_d_par(par)+self.d_delayS_d_par(par)+ \
               self.d_delayA_d_par(par)
