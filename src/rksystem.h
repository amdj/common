// rksystem.h
//
// Author: J.A. de Jong 
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef RKSYSTEM_H
#define RKSYSTEM_H
#include "vtypes.h"
#include <functional>
#include <tuple>

namespace math_common {
  SPOILNAMESPACE

  typedef std::tuple<dmat22,vd2>  dmatvectuple;
  typedef std::tuple<cmat22,vc2>  cmatvectuple;

  typedef std::function<dmatvectuple (const us,const vd&)> dmatvecfun;
  typedef std::function<cmatvectuple (const us,const vc&)> cmatvecfun;


  // RKsystem,
  // Solves the system dy/dx = C(x,y)*y+d(x,y) and returns the matrix K and
  // vector l such that y_i+1 = K*y_i + l

  // requires fixed size matrices, for example // 
  // mat::fixed<2,2> and vec::fixed<2> of c_mat::fixed<2,2>
  // i: discrete location for which CdFun needs to be evaluated
  // h: step size
  // Cdfun: a function which returns C and d for input i and y_i
  template<typename vecT,typename matT>
  std::tuple<matT,vecT>
  RK4system(const us i,
           const d h,
           const vecT& yi,
           std::function<std::tuple<matT,vecT>(const us,const vecT&)> CdFun)
  {
    TRACE(5,"RK4system()");
    const d halfh=0.5*h;
    const matT I(fillwith::eye);     // Identity matrix

    // K1
    matT kappa1; vecT lambda1;
    {
      std::tie(kappa1,lambda1)=CdFun(i,yi);
    }
    const vecT K1=kappa1*yi+lambda1;

    // K2
    matT kappa2; vecT lambda2;
    {    
      matT C2_1,C2_2; vecT d2_1,d2_2;
      std::tie(C2_1,d2_1)=CdFun(i,yi+halfh*K1);
      std::tie(C2_2,d2_2)=CdFun(i+1,yi+halfh*K1);
      matT C2=0.5*(C2_1+C2_2);
      vecT d2=0.5*(d2_1+d2_2);

      kappa2=C2*(I+halfh*kappa1);
      lambda2=d2+C2*(halfh*lambda1);
    }
    const vecT K2=kappa2*yi+lambda2;

    // K3
    matT kappa3; vecT lambda3;
    {
      matT C3_1,C3_2; vecT d3_1,d3_2;      
      std::tie(C3_1,d3_1)=CdFun(i,yi+halfh*K2);
      std::tie(C3_2,d3_2)=CdFun(i+1,yi+halfh*K2);
      matT C3=0.5*(C3_1+C3_2);
      vecT d3=0.5*(d3_1+d3_2);
      kappa3=C3*(I+halfh*kappa2);
      lambda3=d3+C3*(halfh*lambda2);
    }
    const vecT K3=kappa3*yi+lambda3;

    // K4
    matT kappa4; vecT lambda4;
    {
      matT C4; vecT d4;
      std::tie(C4,d4)=CdFun(i+1,yi+h*K3);
      kappa4=C4*(I+h*kappa3);
      lambda4=d4+C4*h*lambda3;
    }
    const vecT K4=kappa4*yi+lambda4;

    matT K=I+(h/6.0)*(kappa1+2.0*(kappa2+kappa3)+kappa4);
    vecT l=(h/6.0)*(lambda1+2.0*(lambda2+lambda3)+lambda4);
    return std::make_tuple(K,l);
  }


  // Solve the system using RK4. Example:
  // tie(p1,U1,aTmtrx,aRlid)=solveRK4(x,y0,CdFun);
  
  template<typename vec,typename vecT,typename matT>
  std::tuple<vec,vec,matT,vecT> // Output Tuple
  solveRK4(const vd& x,const vecT& y0,
           std::function<std::tuple<matT,vecT>(const us,const vecT&)> CdFun ) {
    TRACE(15,"solveRK4(x,y0,Cdfun)");


    us gp=x.size();
    vec res0(gp),res1(gp);
    res0(0)=y0(0);
    res1(0)=y0(1);
    d h;                        // Grid distance

    vecT yi=y0;
    vecT yip1;

    matT T(fillwith::eye); vecT r(fillwith::zeros);
    for (us i=0;i<gp-1;i++){

      // Update grid distance
      h=x(i+1)-x(i);
      matT Ki; vecT li;
      // Obtain matrix and vector to compute new point
      std::tie(Ki,li)=RK4system(i,h,yi,CdFun);
      // Update K,R
      T=Ki*T;
      r=Ki*r+li;
      // Update result vector
      yip1=T*y0+r;
      res0(i+1)=yip1(0);
      res1(i+1)=yip1(1);

      yi=yip1;
    }

    return std::make_tuple(res0,res1,T,r);
  }

}

#endif // RKSYSTEM_H
//////////////////////////////////////////////////////////////////////
