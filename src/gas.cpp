// gas.cpp
//
// last-edit-by: J.A. de Jong 
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#include "tracer.h"
#include "exception.h"

#include "gas.h"
#include "air.h"
#include "helium.h"

namespace gases {

  using std::string;
  // protected constructor
  Gas::Gas() {}
  Gas::Gas(const Gas& other):Gas(other.name){}
  Gas::Gas(const string& mattype){
    setGas(mattype);
  }
  Gas::~Gas() {
    delete g;
  }
  Gas& Gas::operator=(const Gas& other){
    setGas(other.name);
    return *this;
  }
  void Gas::setGas(const string& mattype) {
    TRACE(15,"Gas::setGas(gasname)");
    delete g;
    if(mattype.compare("air")==0) {
      TRACE(15,"Gas type selected is air");
      g=new Air();
      name=mattype;
    }
    else if(mattype.compare("helium")==0){
      TRACE(15,"Gas type selected is helium");
      g=new Helium();
      name=mattype;
    }
    else{
      WARN("Gas type not understood. Type tried: " << mattype);
      throw MyError("Invalid gas type");
    }
  }

} // Namespace gases

//////////////////////////////////////////////////////////////////////

