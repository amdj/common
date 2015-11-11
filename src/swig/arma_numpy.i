%module arma_numpy
%{
  // #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
  #define PY_ARRAY_UNIQUE_SYMBOL npy_array
  #include <numpy/ndarrayobject.h>
  /* #define NO_IMPORT_ARRAY */
  #define SWIG_FILE_WITH_INIT
  #include <cerrno>
  #include <cstdlib>
  #include "vtypes.h"
  #include "arma_numpy.h"
  %}


/* Call import_array() on loading module*/
%init%{
  TRACE(10, "Import array called");
  import_array();
  %}

typedef double d;
typedef unsigned us;


%typecheck(SWIG_TYPECHECK_DOUBLE) // This argument just tells swig in which order to check things
float, double,
  const float &, const double &
{
  $1 = (PyFloat_Check($input) || PyInt_Check($input) || PyLong_Check($input)) ? 1 : 0;
}
%typecheck(SWIG_TYPECHECK_COMPLEX)
c,c&,const c &
{
  $1 = (PyComplex_Check($input) ) ? 1 : 0;
}

%typecheck(1096) vd{
  $1=check_vd($input);
 }
%typecheck(1095) vc,vc&,const vc& {
  $1=check_vc($input);
}

%typecheck(1094) vd2& {
  $1=check_vd2($input);
 }
%typecheck(1093) vc2& {
  $1=check_vc2($input);
 }

%typecheck(1097) dmat,dmat&,const dmat& {
  $1=check_dmat($1);
}
// %typecheck(1097) cmat,cmat&,const cmat& {
//   $1=check_cmat($1);
// }


%typemap(in) vd& (vd temp) {
  if(!check_vd($input)){
    PyErr_SetString(PyExc_ValueError,"Expected a one-dimensional array of doubles");
    return NULL;
  }
  temp=vd_from_npy_nocpy((PyArrayObject*) $input);
  $1=&temp;
 }
%typemap(in) vc& (vc temp) {
  if(!check_vc($input)){
    PyErr_SetString(PyExc_ValueError,"Expected a one-dimensional array of complex doubles");
    return NULL;
  }
  temp=vc_from_npy_nocpy((PyArrayObject*) $input);
  $1=&temp; 
 }

%typemap(in) vd2& (vd2 temp) {
  if(!check_vd2($input)){
    PyErr_SetString(PyExc_ValueError,"Expected a one-dimensional array of doubles of length 2");
    return NULL;
  }      
  temp=vd2_from_npy((PyArrayObject*) $input);
  $1=&temp;
 }
%typemap(in) vc2& (vc2 temp) {
  if(!check_vc2($input)){
    PyErr_SetString(PyExc_ValueError,"Expected a one-dimensional array of complex doubles of length 2");
    return NULL;
  }      
  temp=vc2_from_npy((PyArrayObject*) $input);
  $1=&temp;
 }

%typemap(in) dmat& (dmat temp) {
  if(!check_dmat($input)){
    PyErr_SetString(PyExc_ValueError,"Expected a two-dimensional array of doubles");
    return NULL;
  }      
  temp=dmat_from_npy((PyArrayObject*) $input);
  $1=&temp;
 }
%typemap(in) cmat& (cmat temp) {
  if(!check_cmat($input)){
    PyErr_SetString(PyExc_ValueError,"Expected a two-dimensional array of complex doubles");
    return NULL;
  }
  temp=cmat_from_npy((PyArrayObject*) $input);
  $1=&temp;
 }

%typemap(out) vd {
  // std::cout << "vd to numpy\n";
  $result=npy_from_vd($1);
 }
%typemap(out) vc {
  // std::cout << "vc to numpy\n";
  $result=npy_from_vc($1);
 }
%typemap(out) vd& {
  // std::cout << "vd& to numpy\n\n";
  const vd& res=*$1;
  $result=npy_from_vd(res);
 }
%typemap(out) vc& {
  // std::cout << "vc& to numpy\n\n";
  const vc& res=*$1;
  $result=npy_from_vc(res);
 }
%typemap(out) dmat {
  $result=npy_from_dmat($1);
 }
%typemap(out) (const dmat&) {
  $result=npy_from_dmat(*$1);
}
%typemap(out) dmat22 {
  $result=npy_from_dmat22($1);
 }
%typemap(out) cmat22 {
  $result=npy_from_cmat22($1);
 }

