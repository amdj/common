// air.h
//
// Author: J.A. de Jong 
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef AIR_H
#define AIR_H
#include "perfectgas.h"

namespace gases {
  
  class Air : public PerfectGas {
  public:
    Air():PerfectGas(){}
    d Rs() const;
    d cp(d T) const;
    vd cp(const vd& T) const;
    d h(d T) const;
    vd h(const vd& T) const;
    d mu(d T) const;
    vd mu(const vd&) const;
    d kappa(d T) const;
    vd kappa(const vd& T) const;
    virtual ~Air(){}
  };

  
} // namespace gases


#endif // AIR_H
//////////////////////////////////////////////////////////////////////
