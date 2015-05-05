// nitrogen.cpp
//
// last-edit-by: J.A. de Jong 
// 
// Description:
//
//////////////////////////////////////////////////////////////////////

#include "nitrogen.h"
#include "tracer.h"
#include <cassert>
namespace gases {
  
  namespace  {
    // Nitrogen-specific data
    d cpc[]={3.29868,0.00140824,                \
             -3.96322e-6,5.64152e-9,            \
             -2.44486e-12};

    d kappavals[]={6.995e-3,0.0631e-3};
  } // namespace
  
  d Nitrogen::h(d T) const {
    return Rs()*(cpc[0]*T+0.5*cpc[1]*pow(T,2)+(1/3.0)*cpc[2]*pow(T,3)+cpc[3]*0.25*pow(T,4)+cpc[4]*(0.2)*pow(T,5));
  }
  vd Nitrogen::h(const vd& T) const {
    return Rs()*(cpc[0]*T+0.5*cpc[1]*pow(T,2)+(1/3.0)*cpc[2]*pow(T,3)+cpc[3]*0.25*pow(T,4)+cpc[4]*(0.2)*pow(T,5));
  }
  vd Nitrogen::cp(const vd& T) const {
    return Rs()*(cpc[0]+cpc[1]*T+cpc[2]*pow(T,2)+cpc[3]*pow(T,3)+cpc[4]*pow(T,4));
  }
  d Nitrogen::cp(d T) const {
    return Rs()*(cpc[0]+cpc[1]*T+cpc[2]*pow(T,2)+cpc[3]*pow(T,3)+cpc[4]*pow(T,4));
  }
  // Document Mina
  vd Nitrogen::kappa(const vd& T) const {
    return kappavals[1]*T+kappavals[0];
  }
  d Nitrogen::kappa(d T) const {
    return kappavals[1]*T+kappavals[0];
  }
  // http://www.lmnoeng.com/Flow/GasViscosity.php
  vd Nitrogen::mu(const vd& T) const {
    return (0.01781/1000)*(411.55/(T+111))%pow(T/300.55,1.5);
  }
  d Nitrogen::mu(d T) const {
    return (0.01781/1000)*(411.55/(T+111))*pow(T/300.55,1.5);
  }

  
} // namespace gases

//////////////////////////////////////////////////////////////////////
