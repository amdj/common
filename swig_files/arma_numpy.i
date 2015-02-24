%module arma_numpy
%{
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
  std::cout << "Import array called" << std::endl;
  import_array();
%}

// A better way is the following
// Convert from numpy to vd and vice versa
class vd{
 public:
  virtual ~vd();
};

typedef double d;
typedef unsigned us;


%typecheck(SWIG_TYPECHECK_DOUBLE) // This argument just tells swig in which order to check things
	float, double,
	const float &, const double &
{
  $1 = (PyFloat_Check($input) || PyInt_Check($input) || PyLong_Check($input)) ? 1 : 0;
}
%typecheck(SWIG_TYPECHECK_DOUBLE_ARRAY) vd,vd&,
					const vd&
{
  $1=PyArray_Check($input);
}



%typemap(in) vd& (vd temp) {
  std::cout << "Making aray..." << std::endl;
  PyArrayObject* arr = (PyArrayObject*) PyArray_FROM_OTF($input, NPY_DOUBLE, NPY_IN_ARRAY);
  if(arr==NULL)
    return NULL;
  std::cout << "Making array done..." << std::endl;
  std::cout << "Getting ndims..." << std::endl;
  int ndims=PyArray_NDIM(arr);
  std::cout << "Got ndims..." << std::endl;
  if(ndims!=1){
    PyErr_SetString(PyExc_TypeError,"Number of dimensions not equal to 1");
    return NULL;
  }
  // std::cout << "Checked for number of dims." << std::endl;
  temp=vd_from_npy(arr);
  
  // std::cout << "test" << std::endl;
  // // std::cout << "Temp built." << std::endl;
  // std::cout << temp << std::endl;
  $1=&temp;
}

%typemap(out) vd {
  std::cout << "Returning object.." << std::endl;
  $result=npy_from_vd($1);
}
%typemap(out) vd& {
  const vd& res=*$1;
  $result=npy_from_vd(res);
}

