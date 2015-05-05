#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H
#include <string>
#include <armadillo>
/*
Author: J.A. de Jong
Last modified: March 12, 2014

This header file can be used for computing two types of material
 properties: air and helium. TODO Mixtures of ideal gases have to be added

*/

namespace gases {
  typedef double d;
  typedef arma::Col<double> vd;
  // Wrapper class of type gas. Introduce a whole bunch of forwarding methods 
  class Gas{
    // This string should be overwritten!
    std::string name="implementation";
    Gas* g=nullptr;
  protected:
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
    virtual vd gamma(const vd& T) const {return g->cp(T)/g->cv(T);}
    virtual vd pr(const vd& T) const {return g->mu(T)*g->cp(T)/g->kappa(T);}
    virtual d pr(d T) const {return g->mu(T)*g->cp(T)/g->kappa(T);}
    virtual d gamma(d T) const {return g->cp(T)/g->cv(T);}

    // Virtutals which should be overridden
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


#endif	//MATERIAL_H
