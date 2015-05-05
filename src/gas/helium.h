// helium.h
//
// Author: J.A. de Jong 
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef HELIUM_H
#define HELIUM_H
#include "perfectgas.h"

namespace gases {
  
  class Helium :public PerfectGas {
  public:
    Helium():PerfectGas(){}
    d Rs() const {return 2077;}
    d cp(d T) const { return 5195;}
    vd cp(const vd& T) const;
    d h(d T) const {return cp(0.0f)*T;}
    vd h(const vd& T) const;
    vd mu(const vd& T) const;
    vd kappa(const vd& T) const;
    d kappa(d T) const;
    d mu(d T) const;
    virtual ~Helium(){}
  };

  
} // namespace gases


#endif // HELIUM_H
//////////////////////////////////////////////////////////////////////
