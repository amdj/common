#pragma once

#include <math_common.h>

namespace rottfuncs{
  SPOILNAMESPACE
  const c sqI=sqrt(I);
  const c sq2=sqrt(2.0);
  const c sqmI=sqrt(-1.0*I);

  template <class vcorc>
  vcorc f_vert(const vcorc& rh_over_delta){ //Vertical plate f_nu/f_k
    TRACE(0,"f_vert");
    c a=I+1.0;
    return tanh(a*rh_over_delta)/(a*rh_over_delta);
  }
  vc f_circ(const vc&);
  vc f_blapprox(const vc&);
  vc f_inviscid(const vc&);
  vc f_square(const vc&);

  
  
  
  class rottfuncs{		// Thermoviscous Rott functions
  public:
    rottfuncs();
    rottfuncs(string cshape);
    virtual ~rottfuncs();
    vc fx(const vc& rh_over_delta) const;

  protected:
    vc (*f_ptr)(const vc& rh_over_deltak);
    //c besselj0(c& x);
    string cshape;
  private:
  };

}
