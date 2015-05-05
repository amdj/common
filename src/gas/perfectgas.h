// perfectgas.h
//
// Author: J.A. de Jong 
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef PERFECTGAS_H
#define PERFECTGAS_H
#include "gas.h"

namespace gases {
  
  class PerfectGas: public Gas {
  public:
    PerfectGas(): Gas(){}
    ~PerfectGas(){}
    // Not implemented:
    // mu, kappa, h, cp, Rs
    vd rho(const vd& T,d p) const;
    vd rho(const vd& T,const vd& p) const;
    vd p(const vd& T,const vd& rho) const;
    vd cv(const vd& T) const;
    vd e(const vd& T) const;
    vd beta(const vd& T) const; // 1/T
    vd cm(const vd& T) const;

    d rho(d T,d p) const;
    d p(d T,d rho) const;
    d cv(d T) const;
    d e(d T) const;
    d beta(d T) const;   // 1/T
    d cm(d T) const;
    
  };

  
} // namespace gases


#endif // PERFECTGAS_H
//////////////////////////////////////////////////////////////////////

