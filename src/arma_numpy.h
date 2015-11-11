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

#ifndef SWIG
SPOILNAMESPACE
#endif

// Check functions
bool check_vd(PyObject *);
bool check_vc(PyObject *);

bool check_vd2(PyObject *);
bool check_vc2(PyObject *);

bool check_vd4(PyObject *);
bool check_vc4(PyObject *);

bool check_dmat(PyObject *);
bool check_cmat(PyObject *);

bool check_dmat22(PyObject *);
bool check_cmat22(PyObject *);


PyObject* npy_from_vd(const vd& armavec);
PyObject* npy_from_vd2(const vd2& armavec);
PyObject* npy_from_vc(const vc& armavec);
PyObject* npy_from_vc2(const vc2& armavec);

vd vd_from_npy(PyArrayObject const * const);
vc vc_from_npy(PyArrayObject const * const);

vd vd_from_npy_nocpy(PyArrayObject const * const); // These do not make copies
vc vc_from_npy_nocpy(PyArrayObject const * const);

vd2 vd2_from_npy(PyArrayObject const * const);
vc2 vc2_from_npy(PyArrayObject const * const);

dmat dmat_from_npy(PyArrayObject*);
cmat cmat_from_npy(PyArrayObject const* const);

PyObject* npy_from_dmat(const dmat&); // matrix double
// from cmat not yet implemented

PyObject* npy_from_dmat22(const dmat22&); // 2x2 matrix double
PyObject* npy_from_cmat22(const cmat22&); // 2x2 matrix complex
#endif // ARMA_NUMPY_H
//////////////////////////////////////////////////////////////////////

