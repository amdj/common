// nitrogen.h
//
// Author: J.A. de Jong 
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef NITROGEN_H
#define NITROGEN_H

#include "perfectgas.h"

namespace gases {
  
  class Nitrogen : public PerfectGas {
  public:
    Nitrogen():PerfectGas(){name="nitrogen";}
    d Rs() const {return 297;}
    d cp(d T) const;
    vd cp(const vd& T) const;
    d h(d T) const;
    vd h(const vd& T) const;
    d mu(d T) const;
    vd mu(const vd&) const;
    d kappa(d T) const;
    vd kappa(const vd& T) const;
    ~Nitrogen(){}
  };

  
} // namespace gases

#endif // NITROGEN_H

//////////////////////////////////////////////////////////////////////
