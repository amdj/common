#include "fsolve.h"


namespace math_common{

  vd fsolve(vd& guess,vdfun f){
    TRACE(0,"fsolve ");
    //Newton-Raphson implementation
    vd x=guess;
    us size=x.size();
    vd dx(size),fx(size),fx2(size),x2(size);
    dmat dfdx(size,size);
    fx=f(x);

    d funerror;
    d relerror;
    d deltax;
    us i;
    us nloop=0;
    do {
      fx=f(x);
      if(norm(x,2)<1e-8)
	deltax=1e-9;
      else
	deltax=1e-3*norm(x,2);
      for(i=0;i<size;i++){
	x2=x;
	x2(i)+=deltax;
	fx2=f(x2);
	dfdx.col(i)=(fx2-fx)/deltax;      
      }

      dx=-1.0*arma::solve(dfdx,fx);
      x+=dx;
      relerror=norm(dx,2);
      funerror=norm(fx,2);
      nloop++;
    } while(funerror>FUNTOL || relerror>RELTOL);
    return x;
  }

  
  // vd fsolve(vd& guess,vdfun f,boost::function<dmat (vd x)> jac){
  //   TRACE(0,"fsolve ");
  //   //Newton-Raphson implementation
  //   vd x=guess;
  //   vd dx,fx;
  //   dmat dfdx;
  //   fx=f(x);
  //   d funerror;
  //   d relerror;
  //   us nloop=0;
  //   do {
  //     dfdx=jac(f,x);
  //     dx=-1.0*arma::solve(dfdx,fx);
  //     x+=dx;
  //     relerror=abs(dx);
  //     fx=f(x);
  //     funerror=abs(fx);
  //     nloop++;
  //   } while(funerror>FUNTOL || relerror>RELTOL);
  //   return x;
  // }

  // dmat numJac(boost::function<vd (vd x)>& f,vd& x){
  //   TRACE(0,"fsolve numJac");
  //   vd x2,fx2;
  //   dmat dfdx(x.size(),x.size());

  //   vd fx=f(x);    
  //   d deltax;
  //   if(abs(x)<1e-8)
  //     deltax=1e-9;
  //   else
  //     deltax=1e-3*abs(x);
  //   us i;
  //   for(i=0;i<x.size();x++){
  //     x2=x;
  //     x2(i)+=deltax;
  //     fx2=f(x2);
  //     dfdx.submat(0,i,x.size()-1,i)=(fx2-fx)/deltax;      
  //   }
  //   return dfdx;
  // }

  // vd fsolve(vd& guess,boost::function<vd (vd x)>& f){
  //   typedef boost::function<vd (vd x)> vdfun;
  //   boost::function<dmat (vdfun f,vd x)> Jac;
  //   // Jac=boost::bind(&numJac,_1);
  //   // return fsolve(guess,f,Jac);
  // }


  
} // namespace math_common
  











