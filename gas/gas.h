#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H
/*
Author: J.A. de Jong
Last modified: March 12, 2014

This header file can be used for computing two types of material
 properties: air and helium. TODO Mixtures of ideal gases have to be added

*/

#include "idealgas.h"
#include "vtypes.h"


namespace gases {
  SPOILNAMESPACE
  // Wrapper class of type gas. Introduce a whole bunch of forwarding methods 
  class Gas{
  public:
    Gas(string type="air");
    Gas(const Gas& other);
    Gas& operator=(const Gas& other);
    ~Gas();
    void setGas(const string&);
    const string& getGas() const {return type;}

   private:
    perfectgas* m=nullptr;
    string type="air";

  public:
    d Rs() const { return m->Rsval();}

    // The whole bunch of forwarding functions
    vd rho(const vd& T,const d& p) const {return m->rho(T,p);}
    vd rho(const vd& T,const vd& p) const {return m->rho(T,p);}
    vd p(const vd& T,const vd& rho) const {return m->p(T,rho);}
    vd cp(const vd& T) const {
      std::cout << "T:" << std::endl;
      std::cout << "T:"<<T << std::endl;
      return m->cp(T);}
    vd pr(const vd& T) const {return m->pr(T);}
    vd h(const vd& T) const {return m->h(T);}
    vd cv(const vd& T) const {return m->cv(T);}
    vd e(const vd& T) const {return m->e(T);}
    vd beta(const vd& T) const {return m->beta(T);}
    vd gamma(const vd& T) const {return m->gamma(T);}
    vd cm(const vd& T) const {return m->cm(T);}
    vd mu(const vd& T) const { return m->mu(T);}
    vd kappa(const vd& T) const { return m->kappa(T);}

    d rho(const d& T,const d& p) const {return m->rho(T,p);}
    d p(const d& T,const d& rho) const {return m->p(T,rho);}
    d cp(const d& T) const {return m->cp(T);}
    d pr(const d& T) const {return m->pr(T);}
    d h(const d& T) const {return m->h(T);}
    d cv(const d& T) const {return m->cv(T);}
    d e(const d& T) const {return m->e(T);}
    d beta(const d& T) const {return m->beta(T);}
    d gamma(const d& T) const {return m->gamma(T);}
    d cm(const d& T) const {return m->cm(T);}
    d mu(const d& T) const { return m->mu(T);}
    d kappa(const d& T) const { return m->kappa(T);}

   };

} /* namespace gases */


#endif	//MATERIAL_H
