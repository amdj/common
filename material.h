#pragma once
#include "vtypes.h"

#define DEBUG_GAS
#ifdef DEBUG_GAS

#define checkzero(x)				\
  TRACE(-1,"Divide by zero testing entered.");	\
  try {if(min(abs(x))<1e-13) throw 0; }		\
  catch(int a){					\
    TRACE(0,"Divide by zero encountered.");	\
  }

#else
#define checkzero(x)
#endif


namespace gases {

  inline double min(double x) { return(x);}

  class idealgas {
  protected:
    double cpc[5];
    double kappac[5];
    double muc[5];

  public:
    idealgas();
    virtual ~idealgas();
    double Rs;

    double rho(double,double);
    double p(double,double);
    double cp(double);
    double pr(double);
    double h(double);
    double cv(double);
    double e(double);
    double beta(double);
    double gamma(double);
    double cm(double);

    vd rho(vd &T,double p);
    vd rho(vd&,vd&);
    vd p(vd&,vd&);
    vd cp(vd&);
    vd pr(vd&);
    vd h(vd&);
    vd cv(vd&);
    vd e(vd&);
    vd beta(vd&);
    vd gamma(vd&);
    vd cm(vd&);

    // Virtual functions
    virtual vd mu(vd &T)=0; //Pure virtual functions (abstract class)
    virtual double mu(double T)=0;
    virtual double kappa(double)=0;
    virtual vd kappa(vd&)=0;

  };

  class air :public idealgas {
  public:
    air();
    double mu(double T);
    vd mu(vd&);
    double kappa(double T);
    vd kappa(vd& T);
    virtual ~air();
  };
  class helium :public idealgas {
  public:
    helium();
    vd mu(vd& T);
    vd kappa(vd &T);
    double kappa(double T);
    double mu(double T);
    virtual ~helium();
  };

  class Gas{
  public:
    Gas(){}
    Gas(string mattype) {
      TRACE(1,"Gas constructor entered.");
      if(mattype.compare("air")==0)
	{
	  TRACE(1,"Gas type selected is air");
	  m=new air();
	}
      else if(mattype.compare("helium")==0){
	TRACE(1,"Gas type selected is helium");
	m=new helium();
      }
    }
    ~Gas() {
      if(m!=NULL)	delete m;
    }


    vd rho(vd& T,d p) {
      checkzero(T);
      return m->rho(T,p);
    }
    vd rho(vd& T,vd& p) {return m->rho(T,p);}
    // vd rho(vd T,vd p) {return m->rho(T,p);}
    vd p(vd& T,vd& rho) {return m->p(T,rho);}
    vd cp(vd& T){return m->cp(T);}
    vd pr(vd& T) {return m->pr(T);}
    vd h(vd& T) {return m->h(T);}
    vd cv(vd& T){return m->cv(T);}
    vd e(vd& T) {return m->e(T);}
    vd beta(vd& T){return m->beta(T);}
    vd gamma(vd& T) {return m->gamma(T);}
    vd cm(vd& T){return m->cm(T);}
    vd mu(vd& T) { return m->mu(T);}
    vd kappa(vd& T) { return m->kappa(T);}

    d rho(d T,d p) {return m->rho(T,p);}
    d p(d T,d rho) {return m->p(T,rho);}
    d cp(d T){return m->cp(T);}
    d pr(d T) {return m->pr(T);}
    d h(d T) {return m->h(T);}
    d cv(d T){return m->cv(T);}
    d e(d T) {return m->e(T);}
    d beta(d T){return m->beta(T);}
    d gamma(d T) {return m->gamma(T);}
    d cm(d T){return m->cm(T);}
    d mu(d T) { return m->mu(T);}
    d kappa(d T) { return m->kappa(T);}


  private:
    idealgas* m;
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
      TRACE(0,"void Gasprops::operator()(dorvd T,d pm)");
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
      //			TRACELOG("T:"<< T);
      rho=gas.rho(T,p);
      //			TRACELOG("rho:"<< rho);
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
    Gaspropsd(d T,d p,gases::Gas & g): Gaspropsbase(T,p,g){}


  };
  class Gaspropsvd:public Gaspropsbase<vd>{
  public:
    Gaspropsvd(gases::Gas & g): Gaspropsbase(300*ones<vd>(1),1e5*ones<vd>(1),g){}
    void operator()(const vd& T,const d& p){
      TRACE(0,"Gaspropsvd operator(vd T,d p)");
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

  class Solidmat{
  public:
    Solidmat() {}
    virtual ~Solidmat() {}

    virtual vd kappa(vd& T)=0;
    virtual d kappa(d T)=0;
    virtual vd cs(vd& T)=0;
    virtual d cs(d T)=0;
    virtual vd rho(vd& T)=0;
    virtual d rho(d T)=0;

  };
  class stainless: public Solidmat{
  public:
    stainless();
    vd kappa(vd& T);
    d kappa(d T);
    vd cs(vd &T);
    d cs(d T);
    vd rho(vd& T);
    d rho(d T);
    ~stainless();

  };
  class copper: public Solidmat{
  public:
    copper();
    vd kappa(vd& T);
    d kappa(d T);
    vd cs(vd &T);
    d cs(d T);
    vd rho(vd& T);
    d rho(d T);
    virtual ~copper();
  };

  class kapton: public Solidmat{
  public:
    kapton(){}
    ~kapton(){}
    vd kappa(vd& T){ return 0.2*(1.0-exp(-T/100.0));}
    d kappa(d T){ return 0.2*(1.0-exp(-T/100.0));}
    vd cs(vd& T){return 3.64*T;}
    d cs(d T){return 3.64*T;}
    vd rho(vd& T){return 1445.0-0.085*T;}
    d rho(d T){return 1445.0-0.085*T;}
  };


  class Solid{
  public:
    Solid(string type);
    vd kappa(vd& T);
    d kappa(d T);
    vd cs(vd& T);
    d cs(d T);
    vd rho(vd& T);
    d rho(d T);
    ~Solid();
  protected:
    Solidmat *sol;
  private:
  };

  //Container class to keep solid data together
  template <class dorvd>
  class Solidprops{
  public:
    Solidprops(dorvd T,d pm,Solid& s): solid(s),T(T),pm(pm){
      TRACE(0,"solidprops(dorvd T,d pm,solid& s): solid(s),T(T),pm(pm)");
      update();
    }
    Solidprops(Solid& s):solid(s){
      TRACE(0,"solidprops(solid& g)");

    }
    void operator()(dorvd T) {
      TRACE(0,"Solidprops::operator()(T)");
      this->T=T;
      update();
    }
    void operator()(dorvd T,d pm) {
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


} //namespace solids






