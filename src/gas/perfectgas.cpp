// perfectgas.cpp
//
// last-edit-by: J.A. de Jong 
// 
// Description:
//
//////////////////////////////////////////////////////////////////////

#include "perfectgas.h"
#ifdef DEBUG_GAS
#define checkzero(x)				\
  TRACE(-1,"Divide by zero testing entered.");	\
  try {if(min(abs(x))<1e-13) throw 0; }    \
  catch(int a){					\
    TRACE(0,"Divide by zero encountered.");	\
  }
#else
#define checkzero(x)
#endif

namespace gases {
  
  vd PerfectGas::rho(const vd&T,const vd&p) const {
    checkzero(T);
    return p/Rs()/T;
  }
  vd PerfectGas::rho(const vd&T,d p) const {
    checkzero(T);
    return p/Rs()/T;
  }
  vd PerfectGas::p(const vd& T,const vd& rho) const {
    return rho%(Rs()*T);
  }
  vd PerfectGas::cv(const vd& T) const {
    return cp(T)-Rs();
  }
  vd PerfectGas::e(const vd& T) const {
    return h(T)-Rs()*T;
  }
  vd PerfectGas::beta(const vd& T) const {
    return 1/T;
  }
  vd PerfectGas::cm(const vd& T) const {
    return sqrt(gamma(T)*Rs()%T);
  }
  d PerfectGas::rho(d T,d p) const {
    checkzero(T);
    return p/Rs()/T;
  }
  d PerfectGas::p(d T,d rho) const {
    return rho*Rs()*T;
  }
  d PerfectGas::cv(d T) const {
    return cp(T)-Rs();
  }
  d PerfectGas::e(d T) const {
    return h(T)-Rs()*T;
  }
  d PerfectGas::beta(d T) const {
    checkzero(T);
    return 1/T;
  }
  d PerfectGas::cm(d T) const {
    d csq=gamma(T)*Rs()*T;
    return sqrt(csq);
  }
  
} // namespace gases

//////////////////////////////////////////////////////////////////////
