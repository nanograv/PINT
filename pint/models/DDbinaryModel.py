import numpy as np
import functools
import collections
from astropy import log
from .timing_model import Cache
from .PSR_binary import PSRbinary
from .parameter import Parameter, MJDParameter
from scipy.optimize import newton
import astropy.units as u
import astropy.constants as c
from pint import ls,GMsun,Tsun

class DD(PSRbinary):
    def __init__(self,):
        super(DD, self).__init__()
        self.BinaryModelName = 'DD'

        self.add_param(Parameter(name="A0",
             units="s",
             description="DD model aberration parameter A0",
             parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="B0",
             units="s",
             description="DD model aberration parameter B0",
             parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="GAMMA",
             units="second",
             description="Binary Einsten delay GAMMA term",
             parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="DR",
             units="",
             description="Relativistic deformation of the orbit",
             parse_value=np.double),binary_param = True)

        self.add_param(Parameter(name="DTH",
             units="",
             description="Relativistic deformation of the orbit",
             parse_value=np.double),binary_param = True)


        self.add_param(Parameter(name="SINI",
             units="",
             description="Sine of inclination angle",
             parse_value=np.double),binary_param = True)

        self.binary_delay_funcs += [self.DDdelay,]
        self.inter_vars += ['er','eTheta','beta','alpha']
        self.parDefault.update({'A0':0,'B0':0,'DR':0,'DTH':0,'GAMMA':0,'SINI':0})
    def setup(self):
        super(DD,self).setup()
        for par in self.binary_params:
            if getattr(self,par).value == None:
                print getattr(self,par).name
        self.set_default_values(self.parDefault)
    #################################################
    # Derivity for delays in DD model
    @Cache.use_cache
    def er(self):
        return self.ecc()+self.DR.value
    @Cache.use_cache
    def d_er_d_DR(self):
        return np.longdouble(np.ones(len(self.tt0)))*u.Unit("")
    @Cache.use_cache
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
    @Cache.use_cache
    def eTheta(self):
        return self.ecc()+self.DTH.value
    @Cache.use_cache
    def d_eTheta_d_DTH(self):
        return np.longdouble(np.ones(len(self.tt0)))*u.Unit("")
    @Cache.use_cache
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
    @Cache.use_cache
    def alpha(self):
        """Alpha defined in
           T. Damour and N. Deruelle(1986)equation [46]
           alpha = A1/c*sin(omega)
        """
        return self.a1()/c.c*self.sinOmg
    @Cache.use_cache
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
                return self.a1()/c.c*self.cosOmg*getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.tt0)))

    @Cache.use_cache
    def d_alpha_d_A1(self):
        return 1.0/c.c*self.sinOmg

    @Cache.use_cache
    def d_alpha_d_A1DOT(self):
        return self.tt0/c.c*self.sinOmg
    ##############################################
    @Cache.use_cache
    def beta(self):
        """Beta defined in
           T. Damour and N. Deruelle(1986)equation [47]
           beta = A1/c*(1-eTheta**2)**0.5*cos(omega)
        """
        eTheta = self.eTheta()
        return self.a1()/c.c*(1-eTheta**2)**0.5*self.cosOmg
    @Cache.use_cache
    def d_beta_d_par(self,par):
        """beta = A1/c*(1-eTheta**2)**0.5*cos(omega)
           eTheta = ecc+Dth  ??
           dBeta/dA1 = 1.0/c*(1-eTheta**2)**0.5*cos(omega)
           dBeta/dECC = A1/c*((-(e+dr)/sqrt((e+dr)**2)*cos(omega)*de/dECC-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dECC
           dBeta/dEDOT = A1/c*((-(e+dr)/sqrt((e+dr)**2)*cos(omega)*de/dEDOT-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dEDOT
           dBeta/dDth = A1/c*(-(e+dr)/sqrt((e+dr)**2)*cos(omega)
           Other parameters
           dBeta/dPar = -A1/c*(1-eTheta**2)**0.5*sin(omega)*dOmega/dPar
        """
        if par not in self.binary_params:
            errorMesg = par + "is not in binary parameter list."
            raise ValueError(errorMesg)

        if par in ['A1','ECC','EDOT','Dth','A1DOT']:
            dername = 'd_beta_d_'+par
            return getattr(self,dername)()

        else:
            dername = 'd_omega_d_'+par # For parameters only in omega
            if hasattr(self,dername):
                eTheta = self.eTheta()
                a1 = self.a1()
                return -a1/c.c*(1-eTheta**2)**0.5*self.sinOmg*getattr(self,dername)()
            else:
                return np.longdouble(np.zeros(len(self.tt0)))
    @Cache.use_cache
    def d_beta_d_A1(self):
        """dBeta/dA1 = 1.0/c*(1-eTheta**2)**0.5*cos(omega)
        """
        eTheta = self.eTheta()
        return 1.0/c.c*(1-eTheta**2)**0.5*self.cosOmg

    @Cache.use_cache
    def d_beta_d_A1DOT(self):
        """dBeta/dA1DOT = (t-T0)/c*(1-eTheta**2)**0.5*cos(omega)
        """
        eTheta = self.eTheta()
        return self.tt0/c.c*(1-eTheta**2)**0.5*self.cosOmg

    @Cache.use_cache
    def d_beta_d_ECC(self):
        """dBeta/dECC = A1/c*((-(e+dtheta)/sqrt((e+dtheta)**2)*cos(omega)*de/dECC-
                        (1-eTheta**2)**0.5*sin(omega)*domega/dECC
           de/dECC = 1
        """
        eTheta = self.eTheta()
        a1 = (self.a1()).decompose()

        return a1/c.c*((-eTheta)/np.sqrt(eTheta**2)*self.cosOmg- \
               (1-eTheta**2)**0.5*self.sinOmg*self.d_omega_d_par('ECC'))

    @Cache.use_cache
    def d_beta_d_EDOT(self):
        """dBeta/dEDOT = A1/c*((-(e+dtheta)/sqrt((e+dtheta)**2)*cos(omega)*de/dEDOT- \
           (1-eTheta**2)**0.5*sin(omega)*domega/dEDOT
           de/dEDOT = tt0
        """
        eTheta = self.eTheta()
        a1 = (self.a1()).decompose()
        return a1/c.c*((-eTheta)/np.sqrt(eTheta**2)*self.cosOmg*self.tt0- \
               (1-eTheta**2)**0.5*self.sinOmg*self.d_omega_d_par('EDOT'))
    @Cache.use_cache
    def d_beta_d_Dth(self):
        """dBeta/dDth = a1/c*((-(e+dr)/sqrt((e+dr)**2)*cos(omega)
        """
        eTheta = self.eTheta()
        return self.a1()/c.c*(-eTheta)/np.sqrt(eTheta**2)*self.cosOmg



    ##################################################
    @Cache.use_cache
    def Dre(self):
        """Dre defined in
           T. Damour and N. Deruelle(1986)equation [48]
           Dre = alpha*(cos(E)-er)+(beta+gamma)*sin(E)
        """
        er = self.er()
        return self.alpha()*(self.cosE- er)+   \
               (self.beta()+self.GAMMA.value)*self.sinE
    @Cache.use_cache
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
        # First term
        term1 = self.alpha()*(-self.der('er',par)-self.der('E',par)*self.sinE)
        # Second term
        term2 = (self.cosE-self.er())*self.der('alpha',par)
        # Third term
        term3 = (self.der('beta',par)+self.der('GAMMA',par))*self.sinE
        # Fourth term
        term4 = (self.beta()+self.GAMMA.value)*self.cosE*self.der('E',par)

        return term1 + term2 + term3 +term4
    #################################################
    @Cache.use_cache
    def Drep(self):
        """Dervitive of Dre respect to E
           T. Damour and N. Deruelle(1986)equation [49]
           Drep = -alpha*sin(E)+(beta+gamma)*cos(E)
        """
        return -self.alpha()*self.sinE+(self.beta()+self.GAMMA.value)*self.cosE
    @Cache.use_cache
    def d_Drep_d_par(self,par):
        """Drep = -alpha*sin(E)+(beta+gamma)*cos(E)
           dDrep = -alpha*cos(E)*dE + cos(E)*(dbeta+dgamma)
                   -(beta+gamma)*dE*sin(E)-dalpha*sin(E)
           dDrep/dPar = -sin(E)*dalpha/dPar
                        -(alpha*cos(E)+(beta+gamma)*sin(E))*dE/dPar
                        + cos(E)(dbeta/dPar+dgamma/dPar)

        """
        # first term
        term1 = -self.sinE*self.der('alpha',par)
        # second term
        term2 = -(self.alpha()*self.cosE+  \
                (self.beta()+self.GAMMA.value)*self.sinE)*self.der('E',par)
        # Third term
        term3 = self.cosE*(self.der('beta',par)+self.der('GAMMA',par))

        return term1+term2+term3

    #################################################
    @Cache.use_cache
    def Drepp(self):
        """Dervitive of Drep respect to E
           T. Damour and N. Deruelle(1986)equation [50]
           Drepp = -alpha*cos(E)-(beta+GAMMA)*sin(E)
        """
        return -self.alpha()*self.cosE-(self.beta()+self.GAMMA.value)*self.sinE
    @Cache.use_cache
    def d_Drepp_d_par(self,par):
        """Drepp = -alpha*cos(E)-(beta+GAMMA)*sin(E)
           dDrepp = -(beta+gamma)*cos(E)*dE - cos(E)*dalpha
                    +alpha*sin(E)*dE - (dbeta+dgamma)*sin(E)

           dDrepp/dPar = -cos(E)*dalpha/dPar
                         +(alpha*sin(E)-(beta+gamma)*cos(E))*dE/dPar
                         -(dbeta/dPar+dgamma/dPar)*sin(E)
        """

        # first term
        term1 = -self.cosE*self.der('alpha',par)
        # second term
        term2 = (self.alpha()*self.sinE -  \
                (self.beta()+self.GAMMA.value)*self.cosE)*self.der('E',par)
        # Third term
        term3 = -self.sinE*(self.der('beta',par)+self.der('GAMMA',par))

        return term1+term2+term3
    #################################################
    @Cache.use_cache
    def nhat(self):
        """nhat defined as
           T. Damour and N. Deruelle(1986)equation [51]
           nhat = n/(1-ecc*cos(E))
           n = 2*pi/PB # should here be M()
        """
        return 2.0*np.pi/self.PB.value.to('second')/(1-self.ecc()*self.cosE)
    @Cache.use_cache
    def d_nhat_d_par(self,par):
        """nhat = n/(1-ecc*cos(E))
           n = 2*pi/PB # should here be M()
           dnhat = -2*pi*dPB/PB^2*(1-ecc*cos(E))
                   -2*pi*(-cos(E)*decc+ecc*sin(E)*dE)/PB*(1-ecc*cos(E))^2

           dnhat/dPar = -2*pi/(PB*(1-ecc*cos(E))*((dPB/dPar)/PB -
                        (-cos(E)*decc/dPar+ecc*sin(E)*dE/dpar)/(1-e*cos(E)))
        """
        oneMeccTcosE = (1-self.ecc()*self.cosE)
        fctr = -2*np.pi/self.PB.value/oneMeccTcosE

        return fctr*(self.der('PB',par)/self.PB.value - \
               (self.cosE*self.der('ecc',par)+ \
                self.ecc()*self.sinE*self.der('E',par))/oneMeccTcosE)
    #################################################
    @Cache.use_cache
    def delayInverse(self):
        """DD model Inverse timing delay.
           T. Damour and N. Deruelle(1986)equation [46-52]

           From proper time to coordinate time.
           The Romoer delay and Einstein are included.
        """
        Dre = self.Dre()
        Drep = self.Drep()
        Drepp = self.Drepp()
        nHat = self.nhat()
        e = self.ecc()
        return (Dre*(1-nHat*Drep+(nHat*Drep)**2+1.0/2*nHat**2*Dre*Drepp-\
                1.0/2*e*self.sinE/(1-e*self.cosE)*nHat**2*Dre*Drep)).decompose()
    @Cache.use_cache
    def d_delayI_d_par(self,par):
        """ddelayI/dPar = dDre/dPar*delayI/Dre       [1] half
                          +Dre*d(delayI/Dre)/dPar    [2] half
           d(delayI/Dre)/dPar = -nhat*dDrep/dPar-Drep*dnhat/dPar   part (1)
                                +2*Drep*nhat*(nhat*dDrep/dPar+Drep*dnhat/dPar  part (2)
                                +1/2*nhat*(Drepp*nhat*dDre/dPar+Dre*nhat*dDrepp/dPar+2*Dre*Drepp*dnhat/dPar) part (3)
                                +Part4    part (4)
           Define x:= -1.0/2*ecct*sin(E)/(1-ecct*cos(E))
           part4 = nhat*(Dre*Drep*nhat*dx/dPar+x*(Drep*nHat*dDre/dPar+Dre*nHat*dDrep/dPar+
                   2*Dre*Drep*dnhat/dPar))
           dx/dPar = (-ecc*cos(E)*dE/dPar-sin(E)*decc/dPar
                     +ecc*sin(E)*(-cos(E)*decc/dPar+ecc*sin(E)*dE/dPar)/(1-ecc*cos(E)))/(2*(1-ecc*cos(E)))
        """
        e = self.ecc()
        sE = self.sinE
        cE = self.cosE
        dE_dpar = self.der('E',par)
        decc_dpar = self.der('ecc',par)

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
        #First half
        H1 = dDre_dpar*delayI/Dre
        # For second half
        part1 = -nHat*dDrep_dpar-Drep*dnhat_dpar
        part2 = 2*Drep*nHat*(nHat*dDrep_dpar+Drep*dnhat_dpar)
        part3 = 1.0/2.0*nHat*(Drepp*nHat*dDre_dpar+Dre*nHat*dDrepp_dpar+2*Dre*Drepp*dnhat_dpar)
        part4 = nHat*(Dre*Drep*nHat*dx_dpar+  \
                x*(Drep*nHat*dDre_dpar+Dre*nHat*dDrep_dpar+2*Dre*Drep*dnhat_dpar))

        H2 = Dre*(part1+part2+part3+part4)

        return H1+H2
    @Cache.use_cache
    def d_delayI_d_par2(self,par):
        """
        """
        e = self.ecc()
        sE = self.sinE
        cE = self.cosE
        dE_dpar = self.der('E',par)
        decc_dpar = self.der('ecc',par)

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
        cE = self.cosE
        sE = self.sinE
        sOmega = self.sinOmg
        cOmega = self.cosOmg


        sDelay = -2*self.TM2* np.log(1-e*cE-self.SINI.value*(sOmega*(cE-e)+
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
        cE = self.cosE
        sE = self.sinE
        sOmega = self.sinOmg
        cOmega = self.cosOmg

        logNum = 1-e*cE-self.SINI.value*(sOmega*(cE-e)+
                 (1-e**2)**0.5*cOmega*sE)
        dTM2_dpar = self.der('TM2',par)
        dsDelay_dTM2 = -2*np.log(logNum)
        decc_dpar = self.der('ecc',par)
        dsDelay_decc = -2*self.TM2/logNum*(-cE-self.SINI.value*(-e*cOmega*sE/np.sqrt(1-e**2)-sOmega))
        dE_dpar = self.der('E',par)
        dsDelay_dE =  -2*self.TM2/logNum*(e*sE-self.SINI.value*(np.sqrt(1-e**2)*cE*cOmega-sE*sOmega))
        domega_dpar = self.der('omega',par)
        dsDelay_domega = -2*self.TM2/logNum*self.SINI.value*((cE-e)*cOmega-np.sqrt(1-e**2)*sE*sOmega)
        dSINI_dpar = self.der('SINI',par)
        dsDelay_dSINI = -2*self.TM2/logNum*(-np.sqrt(1-e**2)*cOmega*sE-(cE-e)*sOmega)
        return dTM2_dpar*dsDelay_dTM2 + decc_dpar*dsDelay_decc + \
               dE_dpar*dsDelay_dE +domega_dpar*dsDelay_domega +  \
               dSINI_dpar*dsDelay_dSINI
    #################################################
    @Cache.use_cache
    def delayE(self):
        """Binary Einstein delay
            T. Damour and N. Deruelle(1986)equation [25]
        """
        return self.GAMMA*self.sinE
    @Cache.use_cache
    def d_delayE_d_par(self,par):
        """eDelay = gamma*sin[E]
           deDelay_dPar = deDelay/dgamma*dgamma/dPar +
                          deDelay/dE*dE/dPar
        """
        cE = self.cosE
        sE = self.sinE

        return sE*self.der('GAMMA',par)+self.GAMMA.value*cE*self.der('E',par)
    #################################################
    @Cache.use_cache
    def delayA(self):
        """Binary Abberation delay
            T. Damour and N. Deruelle(1986)equation [27]
        """
        omgPlusAe = self.omega()+self.nu()
        et = self.ecc()
        aDelay = self.A0.value*(np.sin(omgPlusAe)+et*self.sinOmg)+\
                 self.B0.value*(np.cos(omgPlusAe)+et*self.cosOmg)
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
        sOmega = self.sinOmg
        cOmega = self.cosOmg
        snu = np.sin(self.nu())
        cnu = np.cos(self.nu())
        A0 = self.A0.value
        B0 = self.B0.value
        omgPlusAe = self.omega()+self.nu()
        if par =='A0':
            return e*sOmega+np.sin(omgPlusAe)
        elif par == 'B0':
            return e*cOmega+np.cos(omgPlusAe)
        else:
            domega_dpar = self.der('omega',par)
            daDelay_domega = A0*(np.cos(omgPlusAe)+e*cOmega)- \
                             B0*(np.sin(omgPlusAe)+e*sOmega)

            dnu_dpar = self.der('nu',par)
            daDelay_dnu = A0*np.cos(omgPlusAe)-B0*np.sin(omgPlusAe)

            decc_dpar = self.der('ecc',par)
            daDelay_decc = A0*sOmega+B0*cOmega

            return domega_dpar*daDelay_domega + dnu_dpar*daDelay_dnu + decc_dpar*daDelay_decc
    #################################################
    @Cache.use_cache
    def DDdelay(self):
        """Full DD model delay"""
        return self.delayInverse()+self.delayS()+self.delayA()

    @Cache.use_cache
    def delayR(self):  # Is this needed?
        """Binary Romoer delay
            T. Damour and N. Deruelle(1986)equation [24]
        """

        rDelay = self.a1()/c.c*(self.sOmg*(self.cosEcc_A-self.er)   \
                 +(1-self.eTheta**2)**0.5*self.cOmg*self.sinEcc_A)
        return rDelay.decompose()
    @Cache.use_cache
    def d_delay_d_par(self,par):
        """Full DD model delay derivtive
        """
        return self.d_delayI_d_par(par)+self.d_delayS_d_par(par)+ \
               self.d_delayA_d_par(par)
