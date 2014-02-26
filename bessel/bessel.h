#pragma once
#include <complex>

namespace math_common{

typedef std::complex<double> c;
c besselj0(c x);
c besselj1_over_x(c x);


} //namespace math_common
