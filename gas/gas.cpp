#include "gas.h"

namespace gases {

  Gas::Gas(string mattype){
    setGas(mattype);
  }
  Gas::Gas(const Gas& other):Gas(other.type){}
  Gas& Gas::operator=(const Gas& other){
    setGas(other.type);
    return *this;
  }
  void Gas::setGas(const string& mattype)
  {
    delete m;
    TRACE(15,"Gas::setGas("<<mattype<<")");
    if(mattype.compare("air")==0)
      {
        TRACE(15,"Gas type selected is air");
        m=new air();
        type=mattype;
      }
    else if(mattype.compare("helium")==0){
      TRACE(15,"Gas type selected is helium");
      m=new helium();
      type=mattype;
    }
    else{
      WARN("Gas type not understood. Doing nothing. Type stays: " << type);
    }
  }
  Gas::~Gas() {
      delete m;
  }

} // Namespace gases

