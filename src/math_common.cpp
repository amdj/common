

namespace math_common{
  
  // cmatvectuple RKsystem(const us i,const d h,const vc& yi,cmatvecfun Cdfun){
  //   TRACE(2,"cmatvectuple RKsystem("<<i<<","<<h<<",yi,Cdfun)");
  //   // Compute The 2x2 K matrix, and the 2x1 l vector for given 2x2 C matrix and  2x1 d vector
  //   us size=yi.size();
  //   cmat K=zeros<cmat>(size,size);
  //   cmat eI(size,size,fillwith::eye);
  //   vc l(size,fillwith::zeros);

  //   cmatvectuple kappalambda1=Cdfun(i,yi);

  //   cmat kappa1=std::get<0>(kappalambda1);
  //   vc lambda1=std::get<1>(kappalambda1);
  //   vc K1=kappa1*yi+lambda1;

  //   cmatvectuple CD2_1=Cdfun(i,yi+0.5*h*K1);
  //   cmatvectuple CD2_2=Cdfun(i+1,yi+0.5*h*K1);
  //   cmat C2=0.5*(std::get<0>(CD2_1)+std::get<0>(CD2_2));
  //   vc D2=0.5*(std::get<1>(CD2_1)+std::get<1>(CD2_2));
  //   cmat kappa2=C2*(eI+0.5*h*kappa1);
  //   vc lambda2=0.5*h*C2*lambda1+D2;
  //   vc K2=kappa2*yi+lambda2;

  //   cmatvectuple CD3_1=Cdfun(i,yi+0.5*h*K2);
  //   cmatvectuple CD3_2=Cdfun(i+1,yi+0.5*h*K2);
  //   cmat C3=0.5*(std::get<0>(CD3_1)+std::get<0>(CD3_2));
  //   vc D3=0.5*(std::get<1>(CD3_1)+std::get<1>(CD3_2));
  //   cmat kappa3=C3*(eI+0.5*h*kappa2);
  //   vc lambda3=0.5*h*C3*lambda2+D3;
  //   vc K3=kappa3*yi+lambda3;

  //   cmatvectuple CD4=Cdfun(i+1,yi+h*K3);
  //   cmat C4=std::get<0>(CD4);
  //   vc D4=std::get<1>(CD4);
  //   cmat kappa4=C4*(eI+h*kappa3);
  //   vc lambda4=h*C3*lambda3+D4;
  //   vc K4=kappa4*yi+lambda4;


  //   K=eI+(h/6.0)*(kappa1+2.0*(kappa2+kappa3)+kappa4);
  //   l=(h/6.0)*(lambda1+2.0*(lambda2+lambda3)+lambda4);
  //   cmatvectuple res(K,l);
  //   return res;
  // }


}//namespace math_common
