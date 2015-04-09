#pragma once
#ifndef _FSOLVE_H_
#define _FSOLVE_H_

// #include <boost/function.hpp>
// #include <boost/bind.hpp>
#include <functional>
#include "vtypes.h"

#ifndef DXNORM
#define DXNORM (1e-5)
#endif

#ifndef FUNTOL
#define FUNTOL (1e-9)
#endif

#ifndef RELTOL
#define RELTOL (1e-9)
#endif


#ifndef MAXITER
#define MAXITER (1000)
#endif
using namespace std::placeholders;    /* adds visibility of _1, _2, _3,...*/

namespace math_common{
  SPOILNAMESPACE
  using namespace std::placeholders;    /* adds visibility of _1, _2, _3,...*/
  // Not using boost anymore
  
  typedef std::function<vd (const vd& x)> vdfun;

  typedef std::function<dmat (const vd& x)> dmatfun;  
  vd fsolve(const vd& guess,vdfun& f);

  typedef std::function<d (const d& x)> dfun;  
  class Fsolverd{
    d funtol;
    d reltol;
    us maxiter;
    bool verbose;
  public:
    Fsolverd(d funtol=1e-6,d reltol=1e-6,us maxiter=100,bool verbose=false):
      funtol(funtol),
      reltol(reltol),
      maxiter(maxiter),
      verbose(verbose)
    {}
    d operator()(dfun& f,const d& guess);
    ~Fsolverd(){}
    void setVerbose(bool verbose){this->verbose=verbose;}
  };
  
} // namespace math_common
  
#endif /* _FSOLVE_H_ */










