// gas.h
//
// Author: J.A. de Jong 
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef GAS_H
#define GAS_H
#include <string>
#include <armadillo>

namespace gases {
  #ifndef SWIG
  typedef double d;
  typedef arma::Col<double> vd;
  #endif
  #ifdef SWIG
  %catches(std::exception,...) Gas::Gas(const std::string& type);
  #endif
  class Gas{
    // This string should be overwritten!
    Gas* g=nullptr;
  protected:
    std::string name;
    Gas();
  public:
    Gas(const std::string& type);
    Gas(const Gas& other);
    #ifndef SWIG
    Gas& operator=(const Gas& other);
    #endif
    virtual ~Gas();
    #ifndef SWIG
    operator std::string() const {return name;}
    void setGas(const std::string&);
    #endif

    // Finally defined:
    vd gamma(const vd& T) const {return cp(T)/cv(T);}
    vd pr(const vd& T) const {return mu(T)*cp(T)/kappa(T);}
    d pr(d T) const {return mu(T)*cp(T)/kappa(T);}
    d gamma(d T) const {return cp(T)/cv(T);}

    // Virtuals which should be overridden
    virtual d Rs() const {return g->Rs();}
    virtual vd rho(const vd& T,d p) const {return g->rho(T,p);}
    virtual vd rho(const vd& T,const vd& p) const {return g->rho(T,p);}
    virtual vd p(const vd& T,const vd& rho) const {return g->p(T,rho);}
    virtual vd cp(const vd& T) const { return g->cp(T);}
    virtual vd h(const vd& T) const {return g->h(T);}
    virtual vd cv(const vd& T) const {return g->cv(T);}
    virtual vd e(const vd& T) const {return g->e(T);}
    virtual vd beta(const vd& T) const {return g->beta(T);}

    virtual vd cm(const vd& T) const {return g->cm(T);}
    virtual vd mu(const vd& T) const { return g->mu(T);}
    virtual vd kappa(const vd& T) const { return g->kappa(T);}

    virtual d rho(d T,d p) const {return g->rho(T,p);}
    virtual d p(d T,d rho) const {return g->p(T,rho);}
    virtual d cp(d T) const {return g->cp(T);}

    virtual d h(d T) const {return g->h(T);}
    virtual d cv(d T) const {return g->cv(T);}
    virtual d e(d T) const {return g->e(T);}
    virtual d beta(d T) const {return g->beta(T);}

    virtual d cm(d T) const {return g->cm(T);}
    virtual d mu(d T) const { return g->mu(T);}
    virtual d kappa(d T) const { return g->kappa(T);}

   };

} /* namespace gases */

#endif // GAS_H
//////////////////////////////////////////////////////////////////////
