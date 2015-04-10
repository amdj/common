// arma_numpy.h
//
// Author: J.A. de Jong 
//
// Description: Header file for functions converting between Numpy
// arrays and armadillo matrices
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef ARMA_NUMPY_H
#define ARMA_NUMPY_H 1

#include <Python.h>
#include <numpy/arrayobject.h>
#include "vtypes.h"

SPOILNAMESPACE
vd vd_from_npy(PyArrayObject const * const);
vd vd_from_npy_nocpy(PyArrayObject const * const); // These do not make copies
PyObject* npy_from_vd(const vd& armavec);
vc vc_from_npy(PyArrayObject const * const);
vc vc_from_npy_nocpy(PyArrayObject const * const);
PyObject* npy_from_vc(const vc& armavec);
PyObject* npy_from_dmat(const dmat&); // matrix double
PyObject* npy_from_dmat22(const dmat22&); // 2x2 matrix double
PyObject* npy_from_cmat22(const cmat22&); // 2x2 matrix complex
#endif // ARMA_NUMPY_H
//////////////////////////////////////////////////////////////////////
