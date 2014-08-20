#Cython wrapper for materials in math_common

include "gas.pxi"


cdef class gas:
    cdef Gas* g
    def __cinit__(self,gasstring):
        self.g=new Gas(pyxstring(gasstring))
    def __dealloc(self):
        del self.g
    cpdef a_rho(self,n.ndarray[n.float64_t,ndim=1] T,n.ndarray[n.float64_t,ndim=1] p):
        cdef vd Tvec=dndtovec(T)
        cdef vd pvec=dndtovec(p)    
        return dvectond(self.g.rho(Tvec,pvec))
    cpdef rho(self,T,p):
        if(type(T))==type(1.) or type(T)==type(1):
            if(type(p))==type(1.) or type(p)==type(1):
               # print "Pressure and temperature are both scalars"
               return self.g.rho(<double> T,<double> p)
            else:
                # print "Temperature is scalar and pressure is array"
                Tar=n.array(p.shape,float)
                Tar[:]=T
                return self.a_rho(Tar,p)
        else:
            if type(p)==type(1.) or type(p)==type(1):
                # print "Pressure is scalar and temperature is array"
                # print "T:", T.shape
                
                par=n.zeros((T.shape[0],),float)
                par[:]=p
                # print "p:",p
                # print "par:",par
                # print "T:",T
                return self.a_rho(T,par)
            else:
                # print "Pressure and temperature are scalars"
                return self.a_rho(<double> T,<double> p)
         
    cpdef a_p(self,n.ndarray[n.float64_t,ndim=1] T,n.ndarray[n.float64_t,ndim=1] rho):
        cdef vd Tvec=dndtovec(T)
        cdef vd rhovec=dndtovec(rho)    
        return dvectond(self.g.p(Tvec,rhovec))
    cpdef p(self,T,rho):
       if(type(T))==type(1.) or type(T)==type(1):
          if(type(rho))==type(1.) or type(rho)==type(1):        
              return self.g.p(<double> T,<double> rho)
          else:
              print "Error: incompatible types for pressure"
       else:
          if(type(rho))!=type(1.) or type(rho)!=type(1):        
            return self.a_p(T,rho)
          else:
              print "Error: incompatible types for pressure"
          
    
    cpdef a_cp(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.g.cp(Tvec))
    cpdef cp(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.g.cp(<double> T)
        else:
            return self.a_cp(T)
        
    cpdef a_pr(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.g.pr(Tvec))
    cpdef pr(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.g.pr(<double> T)
        else:
            return self.a_pr(T)
    cpdef a_h(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.g.h(Tvec))
    cpdef h(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.g.h(<double> T)
        else:
            return self.a_h(T)
    cpdef a_cv(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.g.cv(Tvec))
    cpdef cv(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.g.cv(<double> T)
        else:
            return self.a_cv(T)
    cpdef a_e(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.g.e(Tvec))
    cpdef e(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.g.e(<double> T)
        else:
            return self.a_e(T)
    cpdef a_beta(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.g.beta(Tvec))
    cpdef beta(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.g.beta(<double> T)
        else:
            return self.a_beta(T)
    cpdef a_gamma(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.g.gamma(Tvec))
    cpdef gamma(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.g.gamma(<double> T)
        else:
            return self.a_gamma(T)
    cpdef a_cm(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.g.cm(Tvec))
    cpdef cm(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.g.cm(<double> T)
        else:
            return self.a_cm(T)
    cpdef a_mu(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.g.mu(Tvec))
    cpdef mu(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.g.mu(<double> T)
        else:
            return self.a_mu(T)
    cpdef a_kappa(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.g.kappa(Tvec))
    cpdef kappa(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.g.kappa(<double> T)
        else:
            return self.a_kappa(T)






