#include "math_common.h"


namespace math_common{





vc f_circ(vc& rh_over_delta) {
	vc jv1arg=(I-1.0)*rh_over_delta;
	vc jv0=vc(jv1arg).transform([](c x) { return besselj0(x); });
	vc jv1_over_jv1arg=vc(jv1arg).transform( [](c x) {return besselj1_over_x(x); });
	return jv1_over_jv1arg/jv0;
}
vc f_blapprox(vc& rh_over_delta){
	TRACELOG("f_blapprox(rh/delta)");
	return (1.0-I)/rh_over_delta/2;
}



vc f_inviscid(vc& rh_over_delta){
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

	else if(cshape.compare("blapprox")==0){
		f_ptr=&f_blapprox;
	}
	else{
		f_ptr=&f_inviscid;
	}
}
vc rottfuncs::fx(vc& rh_over_delta){
	TRACE(0,"rottfuncs::fx(vc& rh_over_delta)");
	return (*f_ptr)(rh_over_delta);
}
rottfuncs::~rottfuncs(){
	//dtor
}

dmatvectuple RKsystem(const us i,const d h,const vd& yi,dmatvecfun Cdfun){
	TRACE(2,"dmatvectuple RKsystem(i,h,yi,Cdfun)");

	us size=yi.size();
	dmat K=zeros<dmat>(size,size);
	dmat eI(size,size,fillwith::eye);
	vd l(size,fillwith::zeros);

	dmatvectuple kappalambda1=Cdfun(i,yi);

	dmat kappa1=boost::get<0>(kappalambda1);
	vd lambda1=boost::get<1>(kappalambda1);
	vd K1=kappa1*yi+lambda1;

	dmatvectuple CD2_1=Cdfun(i,yi+0.5*h*K1);
	dmatvectuple CD2_2=Cdfun(i+1,yi+0.5*h*K1);
	dmat C2=0.5*(CD2_1.get<0>()+CD2_2.get<0>());
	vd D2=0.5*(CD2_1.get<1>()+CD2_2.get<1>());
	dmat kappa2=C2*(eI+0.5*h*kappa1);
	vd lambda2=0.5*h*C2*lambda1+D2;
	vd K2=kappa2*yi+lambda2;

	dmatvectuple CD3_1=Cdfun(i,yi+0.5*h*K2);
	dmatvectuple CD3_2=Cdfun(i+1,yi+0.5*h*K2);
	dmat C3=0.5*(CD3_1.get<0>()+CD3_2.get<0>());
	vd D3=0.5*(CD3_1.get<1>()+CD3_2.get<1>());
	dmat kappa3=C3*(eI+0.5*h*kappa2);
	vd lambda3=0.5*h*C3*lambda2+D3;
	vd K3=kappa3*yi+lambda3;

	dmatvectuple CD4=Cdfun(i+1,yi+h*K3);
	dmat C4=CD4.get<0>();
	vd D4=CD4.get<1>();
	dmat kappa4=C4*(eI+h*kappa3);
	vd lambda4=h*C3*lambda3+D4;
	vd K4=kappa4*yi+lambda4;


	K=eI+(h/6.0)*(kappa1+2.0*(kappa2+kappa3)+kappa4);
	l=(h/6.0)*(lambda1+2.0*(lambda2+lambda3)+lambda4);
	dmatvectuple res(K,l);
	return res;
}
cmatvectuple RKsystem(const us i,const d h,const vc& yi,cmatvecfun Cdfun){
	TRACE(2,"cmatvectuple RKsystem("<<i<<","<<h<<",yi,Cdfun)");
	// Compute The 2x2 K matrix, and the 2x1 l vector for given 2x2 C matrix and  2x1 d vector
	us size=yi.size();
	cmat K=zeros<cmat>(size,size);
	cmat eI(size,size,fillwith::eye);
	vc l(size,fillwith::zeros);

	cmatvectuple kappalambda1=Cdfun(i,yi);

	cmat kappa1=boost::get<0>(kappalambda1);
	vc lambda1=boost::get<1>(kappalambda1);
	vc K1=kappa1*yi+lambda1;

	cmatvectuple CD2_1=Cdfun(i,yi+0.5*h*K1);
	cmatvectuple CD2_2=Cdfun(i+1,yi+0.5*h*K1);
	cmat C2=0.5*(CD2_1.get<0>()+CD2_2.get<0>());
	vc D2=0.5*(CD2_1.get<1>()+CD2_2.get<1>());
	cmat kappa2=C2*(eI+0.5*h*kappa1);
	vc lambda2=0.5*h*C2*lambda1+D2;
	vc K2=kappa2*yi+lambda2;

	cmatvectuple CD3_1=Cdfun(i,yi+0.5*h*K2);
	cmatvectuple CD3_2=Cdfun(i+1,yi+0.5*h*K2);
	cmat C3=0.5*(CD3_1.get<0>()+CD3_2.get<0>());
	vc D3=0.5*(CD3_1.get<1>()+CD3_2.get<1>());
	cmat kappa3=C3*(eI+0.5*h*kappa2);
	vc lambda3=0.5*h*C3*lambda2+D3;
	vc K3=kappa3*yi+lambda3;

	cmatvectuple CD4=Cdfun(i+1,yi+h*K3);
	cmat C4=CD4.get<0>();
	vc D4=CD4.get<1>();
	cmat kappa4=C4*(eI+h*kappa3);
	vc lambda4=h*C3*lambda3+D4;
	vc K4=kappa4*yi+lambda4;


	K=eI+(h/6.0)*(kappa1+2.0*(kappa2+kappa3)+kappa4);
	l=(h/6.0)*(lambda1+2.0*(lambda2+lambda3)+lambda4);
	cmatvectuple res(K,l);
	return res;
}


vd fsolve(vd guess,boost::function<vd (vd x)> f){
	us ndofs=guess.size();
	dmat Jac=zeros<dmat>(ndofs,ndofs);
	//Newton-Raphson implementation
	vd x=guess;
	vd x2(ndofs),fx(ndofs),fx2(ndofs);
	//d dxnorm;//=1e-1;
	vd dx;
	fx=f(x);
	d funerror=1.0;
	us nloop=0;
	d deltax=1e-5;
	while(funerror>FUNTOL){
		for(us j=0;j<ndofs;j++){
			if(abs(x(j))<1e-8){	deltax=1e-9;}
			else {deltax=1e-3*abs(x(j));}
			x2=vd(x); x2(j)+=deltax;
			fx2=f(x2);
			Jac.col(j)=(fx2-fx)/deltax;
		}
		dx=-1.0*solve(Jac,fx);
		x+=dx;
		//dxnorm=norm(dx,2);
		fx=f(x);
		funerror=norm(fx,2);
		TRACE(-1,"Iteration " << nloop << ", function error: " << funerror);
		nloop++;
	}

return x;
}




}//namespace math_common
