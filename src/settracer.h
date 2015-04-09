#include "tracer.h"

namespace tracer {

  template<int& tracername>
  void setTracer(int tracelevel){
    #if TRACER==1
    tracername=tracelevel;
    #endif
    
  }      
} // namespace tracer



