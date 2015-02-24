%module gas
%{
  #define PY_ARRAY_UNIQUE_SYMBOL npy_array
  #include "gas.h"
%}
%include "std_string.i"
%include "arma_numpy.i"
 // Test


namespace gases {

  class Gas{
  public:
    Gas(std::string type="air");
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

