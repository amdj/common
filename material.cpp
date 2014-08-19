#include "material.h"

namespace gases {

  Gas::Gas(){
    WARN("Gas initiated, but type not set! We set default to air.");
    m=new air();
  }
  Gas::Gas(string mattype){
    m=NULL;
    setGas(mattype);
  }
  Gas::Gas(const Gas& other):Gas(other.type){}
  Gas& Gas::operator=(const Gas& other){
    delete m;
    setGas(other.type);
    return *this;
  }
  void Gas::setGas(string mattype)
  {
    TRACE(1,"Gas constructor entered.");
    type=mattype;
    if(type.compare("air")==0)
      {
	TRACE(1,"Gas type selected is air");
	m=new air();
      }
    else if(type.compare("helium")==0){
      TRACE(1,"Gas type selected is helium");
      m=new helium();
    }
    else{
      WARN("Gas type not understood. Exiting.");
      exit(1);
    }
  }
  d Gas::Rs() const { return m->Rsval();}
  Gas::~Gas() {
      delete m;
  }
    // The whole bunch of forwarding functions
  vd Gas::rho(const vd& T,const d& p) const {
      checkzero(T);
      return m->rho(T,p);
  }

} // Namespace gases

namespace solids {


  //Container class
  Solid::Solid(string name){
    TRACE(3,"solid constructor called");
    solidstring=name;
    if(name.compare("stainless")==0){
      sol=new stainless();
      TRACE(3,"Solid set to stainless");
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
      exit(1);
      sol=new stainless();

    }
  }
  Solid::Solid(const Solid& other): Solid(other.solidstring){}
  Solid& Solid::operator=(const Solid& other){
    if(sol!=NULL)
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
  vd Solid::cs(const vd& T) const {
    return sol->cs(T);
  }
  d Solid::cs(const d& T) const {
    return sol->cs(T);
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



} //namespace solids

