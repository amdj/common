%module common
%{
  #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
  #define PY_ARRAY_UNIQUE_SYMBOL npy_array
  #include "rottfuncs.h"
  #include "gas.h"
  #include "solid.h"
%}
%include "std_string.i"
%include "arma_numpy.i"
%include "std_complex.i"
typedef std::string string;
typedef std::complex<double> c;

namespace rottfuncs{


  class RottFuncs {		// Thermoviscous Rott functions
  public:
    RottFuncs(const string& cshape);
    ~RottFuncs();
    vc fx(const vd& rh_over_delta) const;
    vc fx(const vc& rh_over_delta) const;
    c fx(const c& rh_over_delta) const;
    c fx(const d& rh_over_delta) const;
    string getCShape() const;
  };

} // namespace rottfuncs


namespace gases {

  class Gas{
  public:
    Gas(string type="air");
    Gas(const Gas& other);
    ~Gas();

    d Rs() const;

    // The whole bunch of forwarding functions
    vd rho(const vd& T,const d& p) const;
    vd rho(const vd& T,const vd& p) const;
    
    vd p(const vd& T,const vd& rho) const;
    vd cp(const vd& T) const;
    vd pr(const vd& T) const;
    vd h(const vd& T) const;
    vd cv(const vd& T) const;
    vd e(const vd& T) const;
    vd beta(const vd& T) const;
    vd gamma(const vd& T) const;
    vd cm(const vd& T) const;
    vd mu(const vd& T) const;
    vd kappa(const vd& T) const;

    d rho(const d& T,const d& p) const;
    d p(const d& T,const d& rho) const;
    d cp(const d& T) const;
    d pr(const d& T) const;
    d h(const d& T) const;
    d cv(const d& T) const;
    d e(const d& T) const;
    d beta(const d& T) const;
    d gamma(const d& T) const;
    d cm(const d& T) const;
    d mu(const d& T) const;
    d kappa(const d& T) const;

   };

} /* namespace gases */

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