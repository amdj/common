// pylock.cpp
//
// last-edit-by: J.A. de Jong 
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#include "tracer.h"
#include "Python.h"
#include "pylock.h"

namespace{
  PyGILState_STATE gstate;
}
namespace python{
  PyLock::PyLock() {
    TRACE(20,"PyLock::PyLock()");
      gstate=PyGILState_Ensure();
  }
  PyLock::~PyLock(){
    TRACE(20,"PyLock::~PyLock()");
    /* Release the thread. No Python API allowed beyond this point. */
    PyGILState_Release(gstate);
  }
}
//////////////////////////////////////////////////////////////////////
