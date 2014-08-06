#pragma once
#include <vtypes.h>


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
  
  
  
  class RottFuncs{		// Thermoviscous Rott functions
  public:
    RottFuncs();
    RottFuncs(const RottFuncs& other);
    RottFuncs& operator=(const RottFuncs&);
    RottFuncs(string cshape);
    vc fx(const vc& rh_over_delta) const;
  private:
    vc (*f_ptr)(const vc& rh_over_deltak);
    void setFptr(string& cshape);
    string cshape;
  };

}
