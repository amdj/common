#pragma once
#ifndef SPH_J_H_
#define SPH_J_H_
#include "vtypes.h"

namespace special{
	#ifndef SWIG
	SPOILNAMESPACE
	#endif

    #ifdef SWIG
    %catches(std::exception,...) sph_j(int n,d x);
    %catches(std::exception,...) sph_j(int n,const vd& x);
    #endif
	// Returns the first n spherical Bessel functions for a given double value
	vd sph_j(int n,d x);
  // Returns a 2D array. The first axis is the value of x and the
  // second axis (say the column number) corresponds to the n'th
  // spherical Bessel function value. Hence the result vector looks like:

  // [ j_0(x_0) j_1(x_0) ... j_n(x_0) ]
  // [ j_0(x_1) j_1(x_1_ ... j_n(x_1) ]
  //     .       .
  //     .       .
  //     .       .
  // [ j_0(x_n) j_1(x_n) ... j_n(x_n) ]
  
	dmat sph_j(int n, const vd& x);

} // namespace special
#endif // SPH_J_H_
