%module common
%{
  #define SWIG_FILE_WITH_INIT
  #define PY_ARRAY_UNIQUE_SYMBOL npy_array
  #include "rottfuncs.h"
  #include "gas.h"
  #include "solid.h"
  #include "settracer.h"

  TRACETHIS
  inline void setCommonTracer(int t) {
    tracer::setTracer<TRACERNAME>(t);
  }
    
%}
%include "consolecolors.h"
%include "std_string.i"
%include "arma_numpy.i"
%include "std_complex.i"
typedef std::complex<double> c;
typedef std::string string;
%include "rottfuncs.h"
void setCommonTracer(int);

%include "gas.h"


namespace solids{

  class Solid{
  public:
    Solid(string type);
    Solid(const Solid&);
    vd kappa(const vd& T) const;
    d kappa(const d& T) const;
    vd cs(const vd& T) const;
    d cs(const d& T) const;
    vd rho(const vd& T) const;
    d rho(const d& T) const;
    ~Solid();
  };
} // namespace solids
