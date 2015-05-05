// air.cpp
//
// last-edit-by: J.A. de Jong 
// 
// Description:
//
//////////////////////////////////////////////////////////////////////

#include "air.h"

namespace gases {

  namespace  {
    // Air-specific data
    d kappac[]={-0.00227583562,1.15480022E-4,   \
                -7.90252856E-8,4.11702505E-11,  \
                -7.43864331E-15};

    d cpc[]={1047.63657,-0.372589265,           \
             9.45304214E-4,-6.02409443E-7,      \
             1.2858961E-10};
    d muc[]={-8.38278E-7,8.35717342E-8,         \
             -7.69429583E-11,4.6437266E-14,     \
             -1.06585607E-17};
  } // namespace
  
  d Air::h(d T) const {
    return cpc[0]*T+0.5*cpc[1]*pow(T,2)+(1/3.0)*cpc[2]*pow(T,3)+cpc[3]*0.25*pow(T,4)+cpc[4]*(0.2)*pow(T,5);
  }
  vd Air::h(const vd& T) const {
    return cpc[0]*T+0.5*cpc[1]*pow(T,2)+(1/3.0)*cpc[2]*pow(T,3)+cpc[3]*0.25*pow(T,4)+cpc[4]*(0.2)*pow(T,5);
  }

  vd Air::kappa(const vd& T) const {
    return kappac[0]+kappac[1]*T+kappac[2]*pow(T,2)+kappac[3]*pow(T,3)+kappac[4]*pow(T,4);
  }
  d Air::kappa(d T) const {
    return kappac[0]+kappac[1]*T+kappac[2]*pow(T,2)+kappac[3]*pow(T,3)+kappac[4]*pow(T,4);
  }
  vd Air::mu(const vd& T) const {
    return muc[1]*T+muc[2]*pow(T,2)+muc[3]*pow(T,3)+muc[4]*pow(T,4)+muc[0];
  }
  d Air::mu(d T) const {
    return muc[1]*T+muc[2]*pow(T,2)+muc[3]*pow(T,3)+muc[4]*pow(T,4)+muc[0];
  }

  
} // namespace gases

//////////////////////////////////////////////////////////////////////
