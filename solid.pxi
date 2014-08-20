include "armapy.pxi"
#hoi

cdef extern from "material.h" namespace "solids":
   cdef cppclass Solid:
      Solid(string)
      d rho(d T)
      vd rho(vd T)
      d cs(d T)
      vd cs(vd T)
      d kappa(d T)
      vd kappa(vd T)
      
