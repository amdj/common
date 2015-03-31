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
PyObject *npy_from_vc(const vc &in) {
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
  vd result((double*)pydata,size,false,false);
  return result;
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
vc vc_from_npy_nocpy(const PyArrayObject *const in) {
  npy_intp size=PyArray_DIMS(in)[0]; // Extract first 
  npy_cdouble *pydata = (npy_cdouble*)PyArray_DATA(in);
  vc result((c*)pydata,size,false,false);
  return result;
}

// Conversion for tranfer matrices
PyObject *npy_from_dmat22(const dmat22 &in) {
  npy_intp dims[2] = {2,2};

  PyArrayObject *array =
      (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if (array == nullptr) {
    std::cout << "Array is null" << std::endl;
    return nullptr;
  }
  double *pydat = (double *)PyArray_DATA(array);
  mempcpy(pydat, in.memptr(), 4 * sizeof(double));
  return (PyObject*)array;
}
PyObject *npy_from_cmat22(const cmat22 &in) {
  npy_intp dims[2] = {2,2};

  PyArrayObject *array =
      (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
  if (array == nullptr) {
    std::cout << "Array is null" << std::endl;
    return nullptr;
  }
  npy_cdouble *pydat = (npy_cdouble *)PyArray_DATA(array);
  mempcpy(pydat, in.memptr(), 4 * sizeof(npy_cdouble));
  return (PyObject*)array;
}

