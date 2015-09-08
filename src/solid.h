#pragma once
#ifndef SOLID_H
#define SOLID_H

#include "vtypes.h"

namespace solids{
  SPOILNAMESPACE
  class Solidmat;

  class Solid{
  public:
    Solid(const string& type="stainless");
    Solid(const Solid&);
    operator string() const {return solidstring;}
    Solid& operator=(const Solid&);
    vd kappa(const vd& T) const;
    d kappa(const d& T) const;
    vd c(const vd& T) const;
    d c(const d& T) const;
    vd rho(const vd& T) const;
    d rho(const d& T) const;
    ~Solid();
  protected:
    Solidmat *sol;
    string solidstring;
  private:
  };



  class Solidmat{
  public:
    Solidmat() {}
    virtual ~Solidmat() {}
    virtual vd kappa(const vd& T) const =0;
    virtual d kappa(const d& T) const =0;
    virtual vd c(const vd& T) const =0;
    virtual d c(const d& T) const =0;
    virtual vd rho(const vd& T) const =0;
    virtual d rho(const d& T) const =0;

  };
  class stainless_hopkins:public Solidmat
  {
  public:
    vd kappa(const vd& T) const;
    d kappa(const d& T) const;
    vd c(const vd& T) const;
    d c(const d& T) const;
    vd rho(const vd& T) const;
    d rho(const d& T) const;
  };

  class stainless: public Solidmat{
  public:
    vd kappa(const vd& T) const;
    d kappa(const d& T) const;
    vd c(const vd& T) const;
    d c(const d& T) const;
    vd rho(const vd& T) const;
    d rho(const d& T) const;

  };
  class copper: public Solidmat{
  public:
    vd kappa(const vd& T) const;
    d kappa(const d& T) const;
    vd c(const vd& T) const;
    d c(const d& T) const;
    vd rho(const vd& T) const;
    d rho(const d& T) const;
  };

  class kapton: public Solidmat{
  public:
    kapton(){}
    ~kapton(){}
    vd kappa(const vd& T) const { return 0.2*(1.0-exp(-T/100.0));}
    d kappa(const d& T) const { return 0.2*(1.0-exp(-T/100.0));}
    vd c(const vd& T) const {return 3.64*T;}
    d c(const d& T) const {return 3.64*T;}
    vd rho(const vd& T) const {return 1445.0-0.085*T;}
    d rho(const d& T) const {return 1445.0-0.085*T;}
  };

} //namespace solids
#endif /* SOLID_H */
