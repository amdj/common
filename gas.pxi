include "armapy.pxi"

cdef extern from "material.h" namespace "gases":
   cdef cppclass Gas:
      Gas(string)

      d rho(d T,d p)
      vd rho(vd T,vd p)
      vd rho(vd T,d p)
      d p(d T,d rho)
      vd p(vd T,vd rho)
      
      d cp(d T)
      vd cp(vd T)
      d pr(d T)
      vd pr(vd T)
      d h(d T)
      vd h(vd T)
      d cv(d T)
      vd cv(vd T)
      d e(d T)
      vd e(vd T)
      d beta(d T)
      vd beta(vd T)
      d gamma(d T)
      vd gamma(vd T)
      d cm(d T)
      vd cm(vd T)
      d mu(d T)
      vd mu(vd T)
      d kappa(d T)
      vd kappa(vd T)
      
