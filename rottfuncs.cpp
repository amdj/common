#include "rottfuncs.h"

namespace rottfuncs{

  using math_common::besselj0;
  using math_common::besselj1_over_x;
  
  vc f_circ(const vc& rh_over_delta) {
    TRACE(0,"f_circ()");
    // vc s=sq2*rh_over_delta;
    c a(-1,1.0);
    vc R_over_delta=2.0*rh_over_delta;
    vc jvarg=a*R_over_delta;
    vc j0=vc(jvarg).transform([](c x) { return besselj0(x); });
    vc j1_over_jxarg=vc(jvarg).transform( [](c x) {return besselj1_over_x(x); });
    return 2.0*j1_over_jxarg/j0;
  }
  
  vc f_square(const vc& rh_over_delta){
    TRACE(0,"f_square()");
    // Adjusted sligthly for improved speed.
    // Needs checking if rh is defined right.
    // For square pore with dimensions AxA : rh=S/PI = A^2 / 4A = A/4
    vc F(rh_over_delta.size(),fillwith::zeros);
    vc C(rh_over_delta.size(),fillwith::zeros);
    us n,m,msq,nsq;
    d pisq=pow(number_pi,2);
    for(n=1; n<11; n=n+2)
      {
	for(m = 1; m<11; m=m+2)
	  {
	    msq=m*m;
	    nsq=n*n;
	    C = 1.0-I*pisq/(32.0*pow(rh_over_delta,2))*(msq+nsq);
	    F+=(1.0/msq/nsq)/C;
	  }
      }
    return 1.0-64.0/(pisq*pisq)*F;
  }
  vc f_blapprox(const vc& rh_over_delta){
    TRACE(0,"f_blapprox(rh/delta)");
    return (1.0-I)/rh_over_delta/2;
  }

  vc f_inviscid(const vc& rh_over_delta){
    TRACE(0,"f_inviscid");
    vc res=zeros<vc>(rh_over_delta.size());
    return res;
  }

  rottfuncs::rottfuncs(){
    f_ptr=&f_inviscid;
  }

  rottfuncs::rottfuncs(string cshape){
    TRACE(8,"rottfuncs::rottfuncs("<<cshape<< ")");
    if(cshape.compare("circ")==0) {
      f_ptr=&f_circ;
    }
    else if(cshape.compare("vert")==0){
      f_ptr=&f_vert<vc>;
    }
    else if(cshape.compare("square")==0){
      f_ptr=&f_square;
    }

    else if(cshape.compare("blapprox")==0){
      f_ptr=&f_blapprox;
    }
    else{
      f_ptr=&f_inviscid;
    }
  }
  vc rottfuncs::fx(const vc& rh_over_delta) const{
    TRACE(0,"rottfuncs::fx(const vc& rh_over_delta)");
    return (*f_ptr)(rh_over_delta);
  }
  rottfuncs::~rottfuncs(){}
} // namespace rottfuncs
