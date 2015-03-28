#pragma once
#ifndef _MATH_COMMON_H_
#define _MATH_COMMON_H_

#include <assert.h>
#include "vtypes.h"
#include <tuple>

namespace math_common{

  SPOILNAMESPACE

  typedef std::tuple<dmat,vd>  dmatvectuple;
  typedef std::tuple<cmat,vc>  cmatvectuple;
  typedef std::function<dmatvectuple (const us,const vd&)> dmatvecfun;
  typedef std::function<cmatvectuple (const us,const vc&)> cmatvecfun;

  dmatvectuple RKsystem(const us i,const d h,const vd& yi,dmatvecfun Cdfun);
  cmatvectuple RKsystem(const us i,const d h,const vc& yi,cmatvecfun Cdfun);

  // These constants are so often used I implement them standardly
  const c sqI=sqrt(I);
  const c sq2=sqrt(2.0);
  const c sqmI=sqrt(-1.0*I);

  // Compute the central derivative at position x(i). Warning: for equidistant grids
  // only!
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

#endif /* _MATH_COMMON_H_ */
