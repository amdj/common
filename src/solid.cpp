#include "solid.h"

namespace solids{
  //Stainless steel
  //Container class
  Solid::Solid(const string& name){
    TRACE(3,"solid constructor called");
    solidstring=name;
    if(name.compare("stainless")==0){
      sol=new stainless();
      TRACE(3,"Solid set to stainless");
    }
    else if(name.compare("stainless_hopkins")==0){
      sol=new stainless_hopkins();
      TRACE(3,"Solid set to stainless_hopkins");
    }
    else if(name.compare("copper")==0){
      sol=new copper();
      TRACE(3,"Solid set to copper");

    }
    else if(name.compare("kapton")==0){
      sol=new kapton();
      TRACE(3,"Solid set to kapton");
    }
    else {
      cerr << "Error: no matching solid material found with: " << name << endl;
      abort();
    }
  }
  Solid::Solid(const Solid& other): Solid(other.solidstring){}
  Solid& Solid::operator=(const Solid& other){
    if(sol!=nullptr)
      delete sol;
    Solid(other.solidstring);
    return *this;
  }
  vd Solid::kappa(const vd& T) const {
    #ifdef ANNE_DEBUG
    cout << "Solid--> kappa called" << endl;
    cout << sol << endl;
    #endif
    vd res=sol->kappa(T);
    return res;
  }
  d Solid::kappa(const d& T) const {
    return sol->kappa(T);
  }
  vd Solid::c(const vd& T) const {
    return sol->c(T);
  }
  d Solid::c(const d& T) const {
    return sol->c(T);
  }
  vd Solid::rho(const vd& T) const {
    return sol->rho(T);
  }
  d Solid::rho(const d& T) const {
    return sol->rho(T);
  }
  Solid::~Solid(){
    //dtor
    delete sol;
  }

  vd stainless_hopkins::c(const vd& T) const{return 490*pow(T,0);}
  d stainless_hopkins::c(const d& T) const{return 490;}   
  vd stainless_hopkins::rho(const vd& T) const{return 7900*pow(T,0);}
  d stainless_hopkins::rho(const d& T) const{return 7900;}  
  vd stainless_hopkins::kappa(const vd& T) const{return 14.9*pow(T,0);}
  d stainless_hopkins::kappa(const d& T) const{return 14.9;}   

   
  vd stainless::c(const vd& T) const {
    vd arg=1.7054e-6*pow(T,-0.88962)+22324.0/pow(T,6);
    return pow(arg,-1.0/3.0)+15/T;
  }
  d stainless::c(const d& T) const {
    d arg=1.7054e-6*pow(T,-0.88962)+22324.0/pow(T,6);
    return pow(arg,-1.0/3.0)+15/T;
  }
  vd stainless::rho(const vd& T) const {
    return 8274.55-1055.23*exp(-1.0*pow((T-273.15-2171.05)/2058.08,2));
  }
  d stainless::rho(const d& T) const {
    return 8274.55-1055.23*exp(-1.0*pow((T-273.15-2171.05)/2058.08,2));
  }
  vd stainless::kappa(const vd& T) const {
    return pow(266800.0*pow(T,-5.2)+0.21416*pow(T,-1.6),-0.25);
  }
  d stainless::kappa(const d& T) const {
    return pow(266800.0*pow(T,-5.2)+0.21416*pow(T,-1.6),-0.25);
  }

  //Copper
  vd copper::kappa(const vd& T) const {
    return 398.0-0.567*(T-300.0);
  }
  d copper::kappa(const d& T) const { return 398.0-0.567*(T-300.0); }
  vd copper::c(const vd& T) const {return 420.0*pow(T,0);}
  d copper::c(const d& T) const {return 420.0*pow(T,0);}
  vd copper::rho(const vd& T) const {return 9000.0*pow(T,0);}
  d copper::rho(const d& T) const {return 9000.0*pow(T,0);}




} //namespace solids



