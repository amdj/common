# distutils: language = c++

#Cython wrapper for materials in math_common

include "solid.pxi"

cdef class solid:
    cdef Solid* s
    def __cinit__(self,matstring):
        self.s=new Solid(matstring)
    def __dealloc(self):
        del self.g
    cpdef a_rho(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.s.rho(Tvec))
    cpdef rho(self,T):
        if(type(T))==type(1.):
            return self.s.rho(<double> T)
        else:
            return self.a_rho(T)
         
    cpdef a_cs(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.s.cs(Tvec))
    cpdef cs(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.s.cs(<double> T)
        else:
            return self.a_cs(T)
        
    # cpdef a_e(self,n.ndarray[n.float64_t,ndim=1] T):
    #     cdef vd Tvec=dndtovec(T)
    #     return dvectond(self.s.e(Tvec))
    # cpdef e(self,T):
    #     if(type(T))==type(1.) or type(T)==type(1):
    #         return self.s.e(<double> T)
    #     else:
    #         return self.a_e(T)
    cpdef a_kappa(self,n.ndarray[n.float64_t,ndim=1] T):
        cdef vd Tvec=dndtovec(T)
        return dvectond(self.s.kappa(Tvec))
    cpdef kappa(self,T):
        if(type(T))==type(1.) or type(T)==type(1):
            return self.s.kappa(<double> T)
        else:
            return self.a_kappa(T)






