#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H
/*
Author: J.A. de Jong
Last modified: March 12, 2014

This header file can be used for computing two types of material
 properties: air and helium. TODO Mixtures of ideal gases have to be added

*/


#include "vtypes.h"
#include "gas.h"
#include "solid.h"



namespace gases {
  SPOILNAMESPACE
  // Wrapper class of type gas. Introduce a whole bunch of forwarding methods 
  class Gas{
  public:
    Gas();
    Gas(string type);
    Gas(const Gas& other);
    Gas& operator=(const Gas& other);
    d Rs() const; 
    ~Gas();

   private:
    void setGas(string);
    gas* m=NULL;
    string type;

  public:
    // The whole bunch of forwarding functions
    vd rho(const vd& T,const d& p) const;
    vd rho(const vd& T,const vd& p) const {return m->rho(T,p);}
    vd p(const vd& T,const vd& rho) const {return m->p(T,rho);}
    vd cp(const vd& T) const {return m->cp(T);}
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



   template <class dorvd>
   class Gaspropsbase{
   public:
     Gaspropsbase(dorvd T,dorvd p,gases::Gas& g): gas(g),p(p),T(T){
       TRACE(0,"Gaspropsbase constructor");
       update();

     }
     void operator()(const dorvd& T) {
       TRACE(0,"void Gasprops::operator()(dorvd T)");
       this->T=T;
       update();
     }
     void operator()(const dorvd& T,const dorvd& p) {
       TRACE(0,"void Gasprops::operator()(dorvd T,const d& pm)");
       this->p=p;
       this->T=T;
       update();
     }

     ~Gaspropsbase(){}
     //Contents:
     dorvd kappa,gamma,mu,rho,cp,cm,pr,beta;
     Gas& gas;
     dorvd p,T;
   protected:
     void update(){
       TRACE(0,"Gasprops::update(T)");
       //			TRACE(9,"T:"<< T);
       rho=gas.rho(T,p);
       //			TRACE(9,"rho:"<< rho);
       kappa=gas.kappa(T);
       beta=gas.beta(T);
       mu=gas.mu(T);
       gamma=gas.gamma(T);
       cp=gas.cp(T);
       gamma=gas.gamma(T);
       cm=gas.cm(T);
       pr=gas.pr(T);
     }
   };


   class Gaspropsd : public Gaspropsbase<d> {
  public:
    Gaspropsd(gases::Gas & g): Gaspropsbase(300,1e5,g){}
    Gaspropsd(const d& T,const d& p,gases::Gas & g): Gaspropsbase(T,p,g){}


  };
  class Gaspropsvd:public Gaspropsbase<vd>{
  public:
    Gaspropsvd(gases::Gas & g): Gaspropsbase(300*ones<vd>(1),1e5*ones<vd>(1),g){}
    void operator()(const vd& T,const d& p){
      TRACE(0,"Gaspropsvd operator(vd T,const d& p)");
      pm=p;
      this->p=p*ones<vd>(T.size());
      this->T=T;
      update();
    }
    void operator()(const vd& T){
      TRACE(0,"Gaspropsvd::operator(vd T)");
      this->T=T;
      update();
    }
    void operator()(const vd& T,const vd& p){
      TRACE(0,"Gaspropsvd::operator(vd T, vd p)");
      pm=sum(p)/p.size();
      Gaspropsbase::operator()(T,p);

    }
    d pm;
  };

} /* namespace gases */

namespace solids{

  class Solid{
  public:
    Solid(string type);
    Solid(const Solid&);
    Solid& operator=(const Solid&);
    vd kappa(const vd& T) const;
    d kappa(const d& T) const;
    vd cs(const vd& T) const;
    d cs(const d& T) const;
    vd rho(const vd& T) const;
    d rho(const d& T) const;
    ~Solid();
  protected:
    Solidmat *sol;
    string solidstring;
  private:
  };

  //Container class to keep solid data together
  template <class dorvd>
  class Solidprops{
  public:
    Solidprops(const dorvd T,const d& pm,Solid& s): solid(s),T(T),pm(pm){
      TRACE(0,"solidprops(dorvd T,d pm,solid& s): solid(s),T(T),pm(pm)");
      update();
    }
    Solidprops(Solid& s):solid(s){
      TRACE(0,"solidprops(solid& g)");

    }
    void operator()(const dorvd T) {
      TRACE(0,"Solidprops::operator()(T)");
      this->T=T;
      update();
    }
    void operator()(const dorvd T,const d& pm) {
      TRACE(0,"Solidprops::operator()(T,pm)");
      this->T=T;
      this->pm=pm;
      update();
    }
    ~Solidprops(){}
    Solid& solid;
    dorvd T,kappa,rho,cs;

  private:
    void update(){
      TRACE(0,"Solidprops::update()");
      kappa=solid.kappa(T);
      rho=solid.rho(T);
      cs=solid.cs(T);
    }
    d pm;

  };


}

#endif	//MATERIAL_H
