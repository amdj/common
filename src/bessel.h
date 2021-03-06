#pragma once
#include "vtypes.h"

namespace math_common{
  SPOILNAMESPACE
  // Bessel function of first kind and zero'th order for complex numbers
  c besselj0(c x);
  // Besself function of first kind and first order for complex numbers
  c besselj1_over_x(c x);

} //namespace math_common
