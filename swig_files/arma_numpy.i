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
  // std::cout << "Import array called" << std::endl;
  import_array();
%}

// A better way is the following
// Convert from numpy to vd and vice versa
class vd{
 public:
  virtual ~vd();
  d operator()(us);
};

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
  $1=PyArray_FROMANY($input,NPY_DOUBLE,1,1,NPY_ARRAY_C_CONTIGUOUS)?1:0;
  // if($1)
  //   cout << "Its a float array!\n";
}
%typecheck(1091) vc,vc&,const vc& {
  $1=PyArray_FROMANY($input,NPY_COMPLEX128,1,1,NPY_ARRAY_C_CONTIGUOUS)?1:0;
  // if($1)
  //   cout << "Its a complex array!\n";
 }

%typemap(in) vd& (vd temp) {
  temp=vd_from_npy_nocpy((PyArrayObject*) $input);
  $1=&temp;
}
%typemap(in) vc& (vc temp) {
  cout << "Vc run\n";
  temp=vc_from_npy_nocpy((PyArrayObject*) $input);
  $1=&temp;
}

%typemap(out) vd {
  $result=npy_from_vd($1);
}
%typemap(out) vc {
  $result=npy_from_vc($1);
}
%typemap(out) vd& {
  const vd& res=*$1;
  $result=npy_from_vd(res);
}
%typemap(out) dmat22 {
  $result=npy_from_dmat22($1);
}
%typemap(out) cmat22 {
  $result=npy_from_cmat22($1);
}

