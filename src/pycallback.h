// pycallback.h
//
// Author: J.A. de Jong 
//
// Description:
//
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef PYCALLBACK_H
#define PYCALLBACK_H
#include "vtypes.h"

namespace python {
  d PythonCallBackDouble(double a, void *clientdata);
  c PythonCallBackComplex(double a, void *clientdata);
}


#endif // PYCALLBACK_H
//////////////////////////////////////////////////////////////////////

