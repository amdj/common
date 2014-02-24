#pragma once

#include "vtypes.h"
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <boost/tuple/tuple.hpp>
#include "bessel/bessel.h"
#define FUNTOL 1e-9
namespace math_common{

template <class T>
T fsolve(T guess,boost::function<T (T x)> f){
	//Newton-Raphson implementation

	T x=guess;
	T x2,fx,fx2,dfdx;
	d dx;
	d dxnorm=1e-1;
	fx=f(x);
	d funerror=1.0;
	us nloop=0;
	d deltax=1e-5;
	while(funerror>FUNTOL){
		if(abs(x)<1e-8){	deltax=1e-9;}
		else {deltax=1e-3*abs(x);}
		x2=x+deltax;
		fx2=f(x2);
		dfdx=(fx2-fx)/deltax;
		dx=-1.0*fx/dfdx;
		x+=dx;
		dxnorm=abs(dx);
		fx=f(x);
		funerror=abs(fx);
		nloop++;
	}
	//prn("nloop:",nloop);
	return x;
}

vd fsolve(vd guess,boost::function<vd (vd x)> f);

typedef boost::tuple<dmat,vd>  dmatvectuple;
typedef boost::tuple<cmat,vc>  cmatvectuple;
typedef boost::function<dmatvectuple (const us,const vd&)> dmatvecfun;
typedef boost::function<cmatvectuple (const us,const vc&)> cmatvecfun;
dmatvectuple RKsystem(const us i,const d h,const vd& yi,dmatvecfun Cdfun);
cmatvectuple RKsystem(const us i,const d h,const vc& yi,cmatvecfun Cdfun);

const c sqI=sqrt(I);
const c sq2=sqrt(2.0);
const c sqmI=sqrt(-1.0*I);


vc f_circ(vc&);
vc f_blapprox(vc&);
vc f_inviscid(vc&);

template <class vcorc>
vcorc f_vert(vcorc& rh_over_delta){ //Vertical plate f_nu/f_k
	c a=I+1.0;
	return tanh(a*rh_over_delta)/(a*rh_over_delta);
}
class rottfuncs{
	public:
		rottfuncs();
		rottfuncs(string cshape);
		virtual ~rottfuncs();
		vc fx(vc& rh_over_deltanu);

	protected:
		vc (*f_ptr)(vc& rh_over_deltak);
		//c besselj0(c& x);
	private:
};

template <class T>
T ddx_central(us i,const arma::Col<T>& y,const vd& x) {
	us gp=x.size();
	TRACE(0,"ddx_central(i,y,x)");
	assert((i>=0) && (i<gp));
	T result;
	if((i>0) && (i<gp-1)) {
		result=(y(i+1)-y(i-1))/(x(i+1)-x(i-1)); //Central difference
		return result;
	}
	else if(i==0) {
		result=(4*y(1)-3*y(0)-y(2))/(2.0*x(1));
		return result;
		}
	else {
		result=(y(i-2)-4*y(i-1)+3*y(i))/(2.0*(x(i)-x(i-1)));
		return result;
	}
}
template <class T>
arma::Col<T> ddx_central(const arma::Col<T>& y,const vd& x){
	TRACE(0,"ddx_central(vec y,vec x");
	us gp=x.size();
	arma::Col<T> dydx(gp);
	for(us i=0;i<gp;i++){
		dydx(i)=ddx_central(i,y,x);
	}
	return dydx;
}


} //namespace math_common
