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
  TRACE(2, "Import array called");
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
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) vd,vd&,const vd&{
  TRACE(2,"Check if it is a double array...");
  $1=(PyArray_Check($input) &&
      (PyArray_TYPE((PyArrayObject*) $input)==NPY_DOUBLE) &&
      (PyArray_NDIM((PyArrayObject*) $input)==1))?1:0;
  if($1){
    TRACE(2,"Is is a double array.");
  }
}
%typecheck(1092) vc,vc&,const vc& {
  TRACE(2,"Check if it is a complex array..");
  $1=(PyArray_Check($input) &&
      (PyArray_TYPE((PyArrayObject*) $input)==NPY_COMPLEX128) &&
      (PyArray_NDIM((PyArrayObject*) $input)==1))?1:0;
  if($1){
    TRACE(2,"Is is a complex array..");
  }
 }
%typecheck(1094) vd2,vd2&,const vd2& {
  TRACE(2,"Check if it is a double array of size 2...");
  $1=(PyArray_Check($input) &&
      (PyArray_TYPE((PyArrayObject*) $input)==NPY_DOUBLE) &&
      (PyArray_NDIM((PyArrayObject*) $input)==1) &&
      (PyArray_DIMS((PyArrayObject*) $input)[0]==2)
      )?1:0;
  if($1){
    TRACE(2,"Is is a double array of size 2..");
  }
}
%typecheck(1093) vc2,vc2&,const vc2& {
  TRACE(2,"Check if it is a complex array of size 2...");
  $1=(PyArray_Check($input) &&
      (PyArray_TYPE((PyArrayObject*) $input)==NPY_COMPLEX128) &&
      (PyArray_NDIM((PyArrayObject*) $input)==1) &&
      (PyArray_DIMS((PyArrayObject*) $input)[0]==2)
      )?1:0;
  if($1){
    TRACE(2,"Is is a complex array of size 2..");
  }
}
%typemap(in) vd& (vd temp) {
  temp=vd_from_npy_nocpy((PyArrayObject*) $input);
  $1=&temp;
}
%typemap(in) vd=vd&;
%typemap(in) vc& (vc temp) {
  TRACE(2,"Converting array to vc...");
  temp=vc_from_npy_nocpy((PyArrayObject*) $input);
  $1=&temp; 
}
%typemap(in) vc=vc&;
%typemap(in) vd2& (vd2 temp) {
  temp=vd2_from_npy((PyArrayObject*) $input);
  $1=&temp;
}
%typemap(in) vc2& (vc2 temp) {
  temp=vc2_from_npy((PyArrayObject*) $input);
  $1=&temp;
}

%typemap(out) vd {
  TRACE(2,"vd to numpy");
  $result=npy_from_vd($1);
}
%typemap(out) vc {
  TRACE(2,"vc to numpy");
  $result=npy_from_vc($1);
}
%typemap(out) vd& {
  TRACE(2,"vd& to numpy\n");
  const vd& res=*$1;
  $result=npy_from_vd(res);
}
%typemap(out) vc& {
  TRACE(2,"vc& to numpy\n");
  const vc& res=*$1;
  $result=npy_from_vc(res);
}
%typemap(out) dmat {
  $result=npy_from_dmat($1);
 }
%typemap(out) dmat22 {
  $result=npy_from_dmat22($1);
}
%typemap(out) cmat22 {
  $result=npy_from_cmat22($1);
}

