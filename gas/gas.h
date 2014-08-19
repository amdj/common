#pragma once
#include "vtypes.h"

#define DEBUG_GAS
#ifdef DEBUG_GAS
#define checkzero(x)				\
  TRACE(-1,"Divide by zero testing entered.");	\
  try {if(min(abs(x))<1e-13) throw 0; }		\
  catch(int a){					\
    TRACE(0,"Divide by zero encountered.");	\
  }
#else
#define checkzero(x)
#endif
inline double min(double x) { return(x);}


namespace gases{
  SPOILNAMESPACE
  // Abstract base class for every gas type
  class gas{

  public:
    virtual ~gas(){}
    
    virtual d rho(const d& T,const d& p) const =0;	// Density in kg/m^3
    virtual d p(const d& T,const d& rho) const =0;	// Pressure in Pa
    virtual d pr(const d& T) const =0;		// The Prandtl number (=mu*cp/kappa)
    virtual d cp(const d& T) const =0;		// Specific heat at constant pressure in J/kg.K
    virtual d h(const d& T) const =0;		// Specific enthalpy (J/kg)
    virtual d cv(const d& T) const =0;	// Specific heat at constant volume
    virtual d e(const d& T) const =0;	// Specific energy (J/kg)
    virtual d beta(const d& T) const =0;	// Thermal expansion coefficient
    // (dV/dT)/V at constant p. For an ideal gas, this is equal to 1/T
    virtual d gamma(const d& T) const =0;	// Ratio of specific heats
    virtual d cm(const d& T) const =0;	// Speed of sound (dp/drho)^(1/2) at
    // constant entropy

    virtual d mu(const d& T) const =0;	// Dynamic viscosity
    virtual d kappa(const d&) const =0;	// Thermal conductivity
    
    virtual vd rho(const vd&T,const d& p) const =0;
    virtual vd rho(const vd&,const vd&) const =0;
    virtual vd p(const vd&,const vd&) const =0;	// 
    virtual vd pr(const vd&) const =0;
    virtual vd cp(const vd&) const =0;
    virtual vd h(const vd&) const =0;
    virtual vd cv(const vd&) const =0;
    virtual vd e(const vd&) const =0;
    
    virtual vd beta(const vd&) const =0;
    virtual vd gamma(const vd&) const =0;
    virtual vd cm(const vd&) const =0;

    virtual vd mu(const vd&T) const =0;
    virtual vd kappa(const vd&) const =0;
    virtual d Rsval() const =0;			// Return the specific gas constant
  };

  // An ideal gas
  class idealgas:public gas {
  protected:
    d cpc[5];			// Specific heat parameters
    d kappac[5];
    d muc[5];
    d Rs;			  // Ideal gas constant
  public:

    d Rsval() const;	       // Return Rs as a function
    d rho(const d& T,const d& p) const;	// Density in kg/m^3
    d p(const d& T,const d& rho) const;	// Pressure in Pa
    d cp(const d& T) const;		// Specific heat at constant pressure in J/kg.K
    d pr(const d& T) const;		// The Prandtl number (=mu*cp/kappa)
    d h(const d& T) const;		// Specific enthalpy (J/kg)
    d cv(const d& T) const;	// Specific heat at constant volume
    d e(const d& T) const;	// Specific energy (J/kg)
    d beta(const d& T) const;	// Thermal expansion coefficient
				// (dV/dT)/V at constant p. For an
				// ideal gas, this is equal to 1/T
    d gamma(const d& T) const;	// Ratio of specific heats
    d cm(const d& T) const;	// Speed of sound (dp/drho)^(1/2) at
				// constant entropy

    vd rho(const vd& T,const d& p) const;
    vd rho(const vd&,const vd&) const;
    vd p(const vd&,const vd&) const;
    vd cp(const vd&) const;
    vd pr(const vd&) const;
    vd h(const vd&) const;
    vd cv(const vd&) const;
    vd e(const vd&) const;
    vd beta(const vd&) const;
    vd gamma(const vd&) const;
    vd cm(const vd&) const;
    
    vd mu(const vd&T) const =0; //Pure virtual functions (abstract class)
    d mu(const d& T) const =0;
    d kappa(const d&) const =0;
    vd kappa(const vd&) const =0;
  };
  class air :public idealgas {
  public:
    air();
    d mu(const d& T) const;
    vd mu(const vd&) const;
    d kappa(const d& T) const;
    vd kappa(const vd& T) const;
    virtual ~air();
  };
  class helium :public idealgas {
  public:
    helium();
    vd mu(const vd& T) const;
    vd kappa(const vd& T) const;
    d kappa(const d& T) const;
    d mu(const d& T) const;
    virtual ~helium();
  };

  
} // namespace gases



