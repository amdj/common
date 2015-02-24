%module solid
%{
  #include "solid.h"
%}

%include "arma_numpy.i"
namespace solids{

  class Solid{
  public:
    Solid(std::string type);
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


