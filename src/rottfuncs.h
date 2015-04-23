#pragma once
#include "vtypes.h"


namespace rottfuncs{
  #ifndef SWIG
  SPOILNAMESPACE
  const c sqI=sqrt(I);
  const c sq2=sqrt(2.0);
  const c sqmI=sqrt(-1.0*I);

  template <class vcorc>
  vcorc f_inviscid(const vcorc& rh_over_delta){
    TRACE(0,"f_inviscid");
    return 0.0*rh_over_delta;
  }
  template <class vcorc>
  vcorc f_vert(const vcorc& rh_over_delta){ //Vertical plate f_nu/f_k
    TRACE(0,"f_vert");
    c a=I+1.0;
    return tanh(a*rh_over_delta)/(a*rh_over_delta);
  }
  template <class vcorc>
  vcorc f_blapprox(const vcorc& rh_over_delta){
    TRACE(0,"f_blapprox()");
    return (1.0-I)/rh_over_delta/2.0;
  }
  #endif
  class RottFuncs {		// Thermoviscous Rott functions
    vc (*f_ptr)(const vc& rh_over_delta);
    c (*f_ptrc)(const c& rh_over_delta);
    void setFptr(const string& cshape);
    string cshape;
  public:
    RottFuncs(const string& cshape);
    RottFuncs();                // Uses inviscid
    RottFuncs(const RottFuncs& other);
    #ifndef SWIG
    RottFuncs& operator=(const RottFuncs&);
    #endif
    ~RottFuncs(){}
    vc fx(const vc& rh_over_delta) const {
      TRACE(5,"fx vc");
      return f_ptr(rh_over_delta);
    }
    vc fx(const vd& rh_over_delta) const {
      TRACE(5,"fx vd");
      return f_ptr((1.0+0.0*I)*rh_over_delta);
    }

    c fx(const c& rh_over_delta) const {return f_ptrc(rh_over_delta);}
    c fx(const d& rh_over_delta) const {c n(rh_over_delta,0); return f_ptrc(n);}
    string getCShape() const {return cshape;}
  };

}
