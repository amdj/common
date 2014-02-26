#include "bessel.h"

namespace math_common{

c besselj0(c x){
	//Abramowich p. 369, polynomial approximation
	if (abs(x)<3.0) {
		//For complex numbers, unfortunately we have to check for both imaginary
		//and real part.
		return 1.0\
		-2.2499997*pow(x/3.0,2)\
		+1.2656208*pow(x/3.0,4)\
		-0.3163866*pow(x/3.0,6)\
		+0.0444479*pow(x/3.0,8)\
		-0.0039444*pow(x/3.0,10)\
		+0.0002100*pow(x/3.0,12);
	}
	else {
//		larger than 3" << endl;
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
}
c besselj1_over_x(c x) {
	if (abs(x)<3.0) {
		return 0.5-0.56249985*pow(x/3.0,2)+0.21093573*pow(x/3.0,4)-0.03954289*pow(x/3.0,6)+0.00443319*pow(x/3.0,8)-0.00031761*pow(x/3.0,10)+0.00001109*pow(x/3.0,12);
	}
	else {
		//		cout << "larger than 3" << endl;
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
}

} //namespace math_common
