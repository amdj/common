#include "bessel.h"
#include "cbessj.h"
namespace math_common{

  // Equation for a skew sine between x=0 and x=1
  d skewsine(d x){
    if (x<0.0)
      return 0.0;
    else if (x<1.0)
      return x-(1.0/(2.0*number_pi))*sin(x*number_pi*2.0);
    else
      return 1.0;
  }
  vd skewsine(const vd& x){
    vd result=x;
    result.transform( [](d val) {return skewsine(val);});
    return result;
  }  
  c besselj0_smallarg(c x){
    //Abramowich p. 369, polynomial approximation
    //For complex numbers, unfortunately we have to check
    //for both imaginary and real part.
    /* return 1.0		 \
    //   -2.2499997*pow(x/3.0,2) \
    //   +1.2656208*pow(x/3.0,4) \
    //   -0.3163866*pow(x/3.0,6) \
    //   +0.0444479*pow(x/3.0,8) \
    //   -0.0039444*pow(x/3.0,10) \
    //   +0.0002100*pow(x/3.0,12); */

    // External implementation
    d z[2];
    d res[2];
    z[0]=x.real();
    z[1]=x.imag();
    CBESSJ(z,0,res);
    return c(res[0],res[1]);
  }
  c besselj0_largearg(c x){
    // larger than 3" << endl;
    // Abramovich, p. 370
    c f0=0.79788456-0.00000077*(3.0/x)-
      0.00552740*pow(3.0/x,2)-
      0.00009512*pow(3.0/x,3)+
      0.00137237*pow(3.0/x,4)-
      0.00072805*pow(3.0/x,5)+
      0.00014476*pow(3.0/x,6);
    c th0=x-0.78539816-
      0.04166397*(3.0/x)-
      0.00003954*pow(3.0/x,2)+
      0.00262573*pow(3.0/x,3)-
      0.00054125*pow(3.0/x,4)-
      0.00029333*pow(3.0/x,5)+
      0.00013558*pow(3.0/x,6);
    return pow(x,-0.5)*f0*cos(th0);
  }  
  c besselj0(c x){
    d trstart=11.0;		// Transition start point
    d trdelta=1.0;		// Transition deltax
    if (abs(x)<trstart)
      return besselj0_smallarg(x);
    else if(abs(x)<(trstart+trdelta))
      return skewsine((abs(x)-trstart)/trdelta)*besselj0_largearg(x)	\
    	+(1.0-skewsine((abs(x)-trstart)/trdelta))*besselj0_smallarg(x);
    else
      return besselj0_largearg(x);
  }
  c besselj1_over_x_smallarg(c x){
    // return 0.5-0.56249985*pow(x/3.0,2)+0.21093573*pow(x/3.0,4)-0.03954289*pow(x/3.0,6)+0.00443319*pow(x/3.0,8)-0.00031761*pow(x/3.0,10)+0.00001109*pow(x/3.0,12);

    // External implementation
    d z[2];
    d res[2];
    z[0]=x.real();
    z[1]=x.imag();
    CBESSJ(z,1,res);
    return c(res[0],res[1])/x;
  }
  c besselj1_over_x_largearg(c x){
    c f1=
      0.79788456
      +0.00000156*(3.0/x)
      +0.01659667*pow(3.0/x,2)
      +0.00017105*pow(3.0/x,3)
      -0.00249511*pow(3.0/x,4)
      +0.00113653*pow(3.0/x,5)
      -0.00020033*pow(3.0/x,6);

    c th1=x
      -2.35619449
      +0.12499612*(3.0/x)
      +0.00005650*pow(3.0/x,2)
      -0.00637879*pow(3.0/x,3)
      +0.00074348*pow(3.0/x,4)
      +0.00079824*pow(3.0/x,5)
      -0.00029166*pow(3.0/x,6);
    return pow(x,-1.5)*f1*cos(th1);

  }
  c besselj1_over_x(c x) {
    d trstart=8.0;
    d trdelta=1.0;
    if (abs(x)<trstart) 
      return besselj1_over_x_smallarg(x);
    else if(abs(x)<trstart+trdelta)
      return skewsine((abs(x)-trstart)/trdelta)*besselj1_over_x_largearg(x)+(1-skewsine((abs(x)-trstart)/trdelta))*besselj1_over_x_smallarg(x);      
    else {
      return besselj1_over_x_largearg(x);
    }

  }

  } //namespace math_common

