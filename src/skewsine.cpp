#include "skewsine.h"

namespace math_common{
  // Equation for a skew sine between x=0 and x=1
  d skewsine(d x){
    if (x<0.0)
      return 0.0;
    else if (x<1.0)
      return x-(1.0/(2.0*number_pi))*sin(x*number_pi*2.0);
    else
      return 1.0;
  }
  vd skewsine(const vd& x){
    vd result=x;
    result.transform( [](d val) {return skewsine(val);});
    return result;
  }

} // namespace math_common
