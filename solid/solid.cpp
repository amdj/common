 #include "solid.h"

 namespace solids{
   //Stainless steel
   vd stainless::cs(const vd& T) const {
     vd arg=1.7054e-6*pow(T,-0.88962)+22324.0/pow(T,6);
     return pow(arg,-1.0/3.0)+15/T;
   }
   d stainless::cs(const d& T) const {
     d arg=1.7054e-6*pow(T,-0.88962)+22324.0/pow(T,6);
     return pow(arg,-1.0/3.0)+15/T;
   }
   vd stainless::rho(const vd& T) const {
     return 8274.55-1055.23*exp(-1.0*pow((T-273.15-2171.05)/2058.08,2));
   }
   d stainless::rho(const d& T) const {
     return 8274.55-1055.23*exp(-1.0*pow((T-273.15-2171.05)/2058.08,2));
   }
   vd stainless::kappa(const vd& T) const {
     return pow(266800.0*pow(T,-5.2)+0.21416*pow(T,-1.6),-0.25);
   }
   d stainless::kappa(const d& T) const {
     return pow(266800.0*pow(T,-5.2)+0.21416*pow(T,-1.6),-0.25);
   }

   //Copper
   vd copper::kappa(const vd& T) const {
     return 398.0-0.567*(T-300.0);
   }
   d copper::kappa(const d& T) const { return 398.0-0.567*(T-300.0); }
   vd copper::cs(const vd& T) const {return 420.0*pow(T,0);}
   d copper::cs(const d& T) const {return 420.0*pow(T,0);}
   vd copper::rho(const vd& T) const {return 9000.0*pow(T,0);}
   d copper::rho(const d& T) const {return 9000.0*pow(T,0);}
} // namespace solids
