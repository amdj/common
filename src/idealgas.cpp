 #include "idealgas.h"

 namespace gases {

   d idealgas::rho(const d& T,const d& p) const {
     checkzero(T);
     return p/Rs/T;
   }
   d idealgas::p(const d& T,const d& rho) const {
     return rho*Rs*T;
   }
   d idealgas::cp(const d& T) const {
     return cpc[0]+cpc[1]*T+cpc[2]*pow(T,2)+cpc[3]*pow(T,3)+cpc[4]*pow(T,4);
   }
   d idealgas::pr(const d& T) const {
     return mu(T)*cp(T)/kappa(T);
   }
   d idealgas::beta(const d& T) const {
     checkzero(T);
     return 1/T;
   }
   d idealgas::cm(const d& T) const {
     d csq=gamma(T)*Rs*T;
     return sqrt(csq);
   }
   d idealgas::gamma(const d& T) const {
     checkzero(T);
     return cp(T)/cv(T);
   }
   d idealgas::cv(const d& T) const {
     return cp(T)-Rs;
   }
   d idealgas::e(const d& T) const {
     return h(T)-Rs*T;
   }
   d idealgas::h(const d& T) const {
     return cpc[0]*T+0.5*cpc[1]*pow(T,2)+(1/3.0)*cpc[2]*pow(T,3)+cpc[3]*0.25*pow(T,4)+cpc[4]*(0.2)*pow(T,5);
   }

   vd idealgas::rho(const vd&T,const vd&p) const {
     checkzero(T);
     return p/Rs/T;
   }
   vd idealgas::rho(const vd&T,const d& p) const {
    checkzero(T);
    return p/Rs/T;
  }
  vd idealgas::p(const vd& T,const vd& rho) const {
    return rho%(Rs*T);
  }
  vd idealgas::cp(const vd& T) const {
    return cpc[0]+cpc[1]*T+cpc[2]*pow(T,2)+cpc[3]*pow(T,3)+cpc[4]*pow(T,4);
  }
  vd idealgas::pr(const vd& T) const {
    return mu(T)%cp(T)/kappa(T);
  }
  vd idealgas::beta(const vd& T) const {
    return 1/T;
  }
  vd idealgas::cm(const vd& T) const {
    return sqrt(gamma(T)*Rs%T);
  }
  vd idealgas::gamma(const vd& T) const {
    return cp(T)/cv(T);
  }
  vd idealgas::cv(const vd& T) const {
    return cp(T)-Rs;
  }
  vd idealgas::e(const vd& T) const {
    return h(T)-Rs*T;
  }
  vd idealgas::h(const vd& T) const {
    return cpc[0]*T+0.5*cpc[1]*pow(T,2)+(1/3.0)*cpc[2]*pow(T,3)+cpc[3]*0.25*pow(T,4)+cpc[4]*(0.2)*pow(T,5);
  }

  d idealgas::Rsval() const { return Rs;}

  air::air() {
    //     TODO Auto-generated constructor stub

    Rs = 287;
    
    cpc[0]=1047.63657;
    cpc[1]=-0.372589265;
    cpc[2]=9.45304214E-4;
    cpc[3]=-6.02409443E-7;
    cpc[4]=1.2858961E-10 ;
    kappac[0]=-0.00227583562;
    kappac[1]=1.15480022E-4;
    kappac[2]=-7.90252856E-8;
    kappac[3]=4.11702505E-11;
    kappac[4]=-7.43864331E-15;
    muc[0]=-8.38278E-7;
    muc[1]=8.35717342E-8;
    muc[2]=-7.69429583E-11;
    muc[3]=4.6437266E-14;
    muc[4]=-1.06585607E-17;

  }
  air::~air() {}
  vd air::kappa(const vd& T) const {
    return kappac[0]+kappac[1]*T+kappac[2]*pow(T,2)+kappac[3]*pow(T,3)+kappac[4]*pow(T,4);
  }
  d air::kappa(const d& T) const {
    return kappac[0]+kappac[1]*T+kappac[2]*pow(T,2)+kappac[3]*pow(T,3)+kappac[4]*pow(T,4);
  }
  vd air::mu(const vd& T) const {
    return muc[1]*T+muc[2]*pow(T,2)+muc[3]*pow(T,3)+muc[4]*pow(T,4)+muc[0];
  }
  d air::mu(const d& T) const {
    return muc[1]*T+muc[2]*pow(T,2)+muc[3]*pow(T,3)+muc[4]*pow(T,4)+muc[0];
  }

  helium::helium() {
    // TODO Auto-generated constructor stub
    cpc[0] = 5195;
    cpc[1]=0;
    cpc[2]=0;
    cpc[3]=0;
    cpc[4]=0;
    Rs = 2077;
  }
  helium::~helium() {
    // TODO Auto-generated destructor stub
  }
  d helium::mu(const d& T) const {
    return 0.412e-6*pow(T,0.68014);
  }
  d helium::kappa(const d& T) const {
    return 0.0025672*pow(T,0.716);
  }
  vd helium::mu(const vd& T) const {
    return 0.412e-6*pow(T,0.68014);
  }
  vd helium::kappa(const vd& T) const {
    return 0.0025672*pow(T,0.716);
  }


} /* namespace gases */
