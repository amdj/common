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
%typecheck(1096) vd,vd&,const vd&{
  // cout << "Check if it is a double array...\n";
  $1=(PyArray_Check($input) &&
      (PyArray_TYPE((PyArrayObject*) $input)==NPY_DOUBLE) &&
      (PyArray_NDIM((PyArrayObject*) $input)==1))?1:0;
  // if($1){
  //   cout << "Is is a double array.\n";
  // }
}
%typecheck(1095) vc,vc&,const vc& {
  // cout << "Check if it is a complex array..\n";
  $1=(PyArray_Check($input) &&
      (PyArray_TYPE((PyArrayObject*) $input)==NPY_COMPLEX128) &&
      (PyArray_NDIM((PyArrayObject*) $input)==1))?1:0;
  // if($1){
  //   cout << "Is is a complex array..\n";
  // }
 }
%typecheck(1094) vd2& {
  // cout << "Check if it is a double array of size 2...\n";
  $1=(PyArray_Check($input) &&
      (PyArray_TYPE((PyArrayObject*) $input)==NPY_DOUBLE) &&
      (PyArray_NDIM((PyArrayObject*) $input)==1) &&
      (PyArray_DIMS((PyArrayObject*) $input)[0]==2)
      )?1:0;
  // if($1){
  //   cout << "Is is a double array of size 2..\n";
  // }
}
%typecheck(1093) vc2& {
  // cout << "Check if it is a complex array of size 2...\n";
  $1=(PyArray_Check($input) &&
      (PyArray_TYPE((PyArrayObject*) $input)==NPY_COMPLEX128) &&
      (PyArray_NDIM((PyArrayObject*) $input)==1) &&
      (PyArray_DIMS((PyArrayObject*) $input)[0]==2)
      )?1:0;
  // if($1){
  //   cout << "Is is a complex array of size 2..\n";
  // }
}
%typemap(in) vd& (vd temp) {
  temp=vd_from_npy_nocpy((PyArrayObject*) $input);
  $1=&temp;
}
%typemap(in) vc& (vc temp) {
  // cout << "Converting array to vc...\n";
  temp=vc_from_npy_nocpy((PyArrayObject*) $input);
  $1=&temp; 
}
%typemap(in) vd2& (vd2 temp) {
  temp=vd2_from_npy((PyArrayObject*) $input);
  $1=&temp;
}
%typemap(in) vc2& (vc2 temp) {
  temp=vc2_from_npy((PyArrayObject*) $input);
  $1=&temp;
}

%typemap(out) vd {
  // cout << "vd to numpy\n";
  $result=npy_from_vd($1);
}
%typemap(out) vc {
  // cout << "vc to numpy\n";
  $result=npy_from_vc($1);
}
%typemap(out) vd& {
  // cout << "vd& to numpy\n\n";
  const vd& res=*$1;
  $result=npy_from_vd(res);
}
%typemap(out) vc& {
  // cout << "vc& to numpy\n\n";
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

