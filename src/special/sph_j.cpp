// sph_j.cpp
//
// last-edit-by: J.A. de Jong 
// 
// Description:
//
//////////////////////////////////////////////////////////////////////

#include "sph_j.h"
#include "exception.h"
#include <gsl/gsl_sf_bessel.h>

namespace special {
  vd sph_j(int n,d x){
	  if (n < 1)
		  throw MyError("n needs to be > 1");
    vd result(n);
    gsl_sf_bessel_jl_steed_array(n,x,result.memptr());
    return result;
  }

  dmat sph_j(int n, const vd& x){
	if (n < 1)
		throw MyError("n needs to be > 1");

    dmat result(x.size(),n);
	for (us i = 0; i < x.size(); i++)
		result.row(i) = sph_j(n, x(i)).t();
    return result;
  }
} // namespace special

//////////////////////////////////////////////////////////////////////
