// pycallback.cpp
//
// last-edit-by: J.A. de Jong 
// 
// Description:
//
//////////////////////////////////////////////////////////////////////
#include <Python.h>
#include "pycallback.h"
#include "exception.h"

namespace python {
  
  d PythonCallBackDouble(double a, void *clientdata) {
    PyObject *func, *arglist;
    PyObject *result;
    double    dres = 0;
   
    func = (PyObject *) clientdata;               // Get Python function
    arglist = Py_BuildValue("(d)",a);             // Build argument list
    result = PyEval_CallObject(func,arglist);     // Call Python
    Py_DECREF(arglist);                           // Trash arglist
    if (PyFloat_Check(result)) {                                 // If no errors, return double
      dres = PyFloat_AsDouble(result);
      Py_XDECREF(result);
    }
    else {
      Py_XDECREF(result);
      throw MyError("Cannot convert result of function to double");
    }
    return dres;
  }
  c PythonCallBackComplex(double a, void *clientdata) {
    TRACE(15,"c PythonCallBackComplex()");
    PyObject *func, *arglist;
    PyObject *result;
    Py_complex res;

    func = (PyObject *) clientdata;               // Get Python function
    arglist = Py_BuildValue("(d)",a);             // Build argument list
    result = PyEval_CallObject(func,arglist);     // Call Python
    Py_DECREF(arglist);                           // Trash arglist
    if (PyComplex_Check(result)) {                                 // If no errors, return double
      // This can be done, as C++ and C complex objects are
      // memory-compatible
      res=PyComplex_AsCComplex(result);
      Py_XDECREF(result);
      return c(res.real,res.imag);
    }
    else {
      Py_XDECREF(result);
      throw MyError("Cannot convert result of function to complex");
    }

  }
  
  
} // namespace python

//////////////////////////////////////////////////////////////////////



