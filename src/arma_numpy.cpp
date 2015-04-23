// arma_numpy.cpp
//
// last-edit-by: J.A. de Jong 
// 
// Description: Conversion functions from PyArrayObjects to Armadillo
// objects and vice versa
//
//////////////////////////////////////////////////////////////////////
#define PY_ARRAY_UNIQUE_SYMBOL npy_array
#define NO_IMPORT_ARRAY
#include "arma_numpy.h"

// Annes conversion functions. In a later stage they have to be
// generalized for arrays of arbitrary dimensions

PyObject *npy_from_vd(const vd &in) {
  long int size = in.size();
  npy_intp dims[1] = {size};

  // This code should be improved massively?
  if (size == 0) {
    std::cout << "Size is zero!\n";
    return nullptr;
  }
  PyArrayObject *array =
      (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  if (array == nullptr) {
    return nullptr;
  }
  double *pydat = (double *)PyArray_DATA(array);
  mempcpy(pydat, in.memptr(), size * sizeof(double));

  return PyArray_Return(array);
}
PyObject *npy_from_vd2(const vd2 &in) {
  long int size = 2;
  npy_intp dims[1] = {size};

  // This code should be improved massively?
  if (size == 0) {
    std::cout << "Size is zero!\n";
    return nullptr;
  }
  PyArrayObject *array =
      (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_DOUBLE);
  if (array == nullptr) {
    return nullptr;
  }
  double *pydat = (double *)PyArray_DATA(array);
  mempcpy(pydat, in.memptr(), size * sizeof(double));

  return PyArray_Return(array);
}
PyObject *npy_from_vc2(const vc2 &in) {
  TRACE(0,"npy_from_vc2(vc&)");
  long int size = 2;
  npy_intp dims[1] = {size};

  // This code should be improved massively?
  if (size == 0) {
    std::cout << "Size is zero!\n";
    return nullptr;
  }
  PyArrayObject *array =
      (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
  if (array == nullptr) {
    return nullptr;
  }
  npy_cdouble *pydat = (npy_cdouble *)PyArray_DATA(array);
  mempcpy(pydat, in.memptr(), size * sizeof(c));

  return PyArray_Return(array);
}
PyObject *npy_from_vc(const vc &in) {
  TRACE(0,"npy_from_vc(vc&)");
  std::cout << "npy_from_vc" << std::endl;
  long int size = in.size();
  npy_intp dims[1] = {size};

  // This code should be improved massively?
  if (size == 0) {
    std::cout << "Size is zero!\n";
    return nullptr;
  }
  PyArrayObject *array =
      (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
  if (array == nullptr) {
    std::cout << "Array is null" << std::endl;
    return nullptr;
  }
  npy_cdouble *pydat = (npy_cdouble *)PyArray_DATA(array);
  mempcpy(pydat, in.memptr(), size * sizeof(npy_cdouble));
  return (PyObject*)array;
}

vd vd_from_npy(const PyArrayObject *const in) {
  npy_intp size=PyArray_DIMS(in)[0]; // Extract first 
  double *pydata = (double *)PyArray_DATA(in);
  vd result(size);
  memcpy(result.memptr(),pydata,size*sizeof(double));
  return result;
}
vd vd_from_npy_nocpy(const PyArrayObject *const in) {
  npy_intp size=PyArray_DIMS(in)[0]; // Extract first 
  double *pydata = (double *)PyArray_DATA(in);
  return vd((double*)pydata,size,false,false);
}
vc vc_from_npy_nocpy(const PyArrayObject *const in) {
  TRACE(0,"vc_from_npy_nocpy");
  npy_intp size=PyArray_DIMS(in)[0]; // Extract first 
  npy_cdouble *pydata = (npy_cdouble*)PyArray_DATA(in);
  return vc((c*)pydata,size,false,false);
}

vc vc_from_npy(const PyArrayObject *const in) {
  npy_intp size=PyArray_DIMS(in)[0]; // Extract first 
  // std::cout << "Test2" << std::endl;
  npy_cdouble *pydata = (npy_cdouble *)PyArray_DATA(in);
  // std::cout << "Test3" << std::endl;
  vc result(size);
  memcpy(result.memptr(),pydata,size*sizeof(npy_cdouble));
  return result;
}


vc2 vc2_from_npy(const PyArrayObject *const in) {
  npy_intp size=PyArray_DIMS(in)[0]; // Extract first 
  npy_cdouble *pydata = (npy_cdouble *)PyArray_DATA(in);
  vc2 result;
  memcpy(result.memptr(),pydata,size*sizeof(npy_cdouble));
  return result;
}
vd2 vd2_from_npy(const PyArrayObject *const in) {
  npy_intp size=PyArray_DIMS(in)[0]; // Extract first 
  npy_double *pydata = (npy_double *)PyArray_DATA(in);
  vd2 result;
  memcpy(result.memptr(),pydata,size*sizeof(npy_double));
  return result;
}

// Conversion for tranfer matrices
PyObject *npy_from_dmat22(const dmat22 &in) {
  npy_intp dims[2] = {2,2};

  PyArrayObject *array =
    (PyArrayObject *)PyArray_EMPTY(2, dims, NPY_DOUBLE,true);
  if (array == nullptr) {
    std::cout << "Array is null" << std::endl;
    return nullptr;
  }
  double *pydat = (double *)PyArray_DATA(array);
  mempcpy(pydat, in.memptr(), 4 * sizeof(double));
  return (PyObject*)array;
}
// Convert Armadillo matrix to Numpy Array
PyObject *npy_from_dmat(const dmat &in) {
  // dmat in=in2.t();
  npy_intp dims[2] = {(npy_intp) in.n_rows,(npy_intp) in.n_cols};
  npy_intp size=(npy_intp) in.n_rows*in.n_cols;
  // Last 
  PyObject* array = PyArray_EMPTY(2, dims, NPY_DOUBLE, true);
  if (!array || !PyArray_ISFORTRAN(array)) {
    std::cout << "Array creation failed" << std::endl;
    return nullptr;
  }
  double *pydat = (double *)PyArray_DATA(array);
  mempcpy(pydat, in.memptr(), size*sizeof(double));
  return (PyObject*)array;
}
PyObject *npy_from_cmat22(const cmat22 &in) {
  npy_intp dims[2] = {2,2};

  PyArrayObject *array =
    (PyArrayObject *)PyArray_EMPTY(2, dims, NPY_COMPLEX128,true);
  if (array == nullptr) {
    std::cout << "Array is null" << std::endl;
    return nullptr;
  }
  npy_cdouble *pydat = (npy_cdouble *)PyArray_DATA(array);
  mempcpy(pydat, in.memptr(), 4 * sizeof(npy_cdouble));
  return (PyObject*)array;
}

