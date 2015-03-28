#pragma once
// file: vtypes.h
// author: J.A. de Jong
// This file defines useful macros for faster typing of numerical code. 

#ifndef VTYPES_H
#define VTYPES_H

#ifndef __cplusplus
#error The c++ compiler should be used.
#endif

#include "tracer.h"		// A tracer instance

// Some header files I (almost) always need
#include <iostream>
#include <iomanip>
#include <vector>		// C++ std vector
#include <string>		// C++ string class
#include <complex>
#include <armadillo>		// Our linear algebra package
#include <functional>

// I need these numbers so often, that they can be in the global namespace
const std::complex<double> I(0,1); //Complex unity
const std::complex<double> minI(0,-1); //Minus complex unity
#define number_pi (arma::datum::pi) // The number 3.14159265359..

typedef unsigned us;		  /* Size type I always use */

#define fillwith arma::fill	    // Fill vector or matrix with ...

//Spoiling global namespace with often used functions and variables 
#define SPOILNAMESPACE								\
  using std::cout; /* Output to stdout */					\
  using std::abs;
  using std::endl;								\
  using std::setprecision;							\
  using std::setiosflags;							\
  using std::cin;		/* Input from stdin */				\
  using std::ostream;								\
  using std::cerr;								\
  using std::string;		/* C++ Strings */				\
  using std::ios;								\
  using std::vector;								\
  /* Armadillo functions and objects we often need */				\
  using arma::ones;								\
  using arma::zeros;								\
  using arma::eye;								\
  using arma::inv;								\
  using arma::det;								\
  using arma::log_det;								\
  using arma::norm;								\
  using arma::span;								\
  using arma::linspace;								\
  using arma::diagmat;								\
  typedef double d;		/* Shortcut for double */			\
  typedef std::complex<double> c; /* Shortcut for c++ complex number */		\
  typedef arma::vec vd;		  /* Column vector of doubles */		\
  typedef arma::cx_vec vc;	  /* Column vector of complex numbers */	\
  typedef arma::Mat<double> dmat; /* (Dense) Matrix of doubles */		\
  typedef arma::sp_mat sdmat;	  /* Sparse matrix of doubles */		\
  typedef arma::cx_mat cmat;	  /* (Dense) matrix of complex numbers */	\
  typedef arma::cx_mat44 cmat44;  /* Fixed size complex */			\
  typedef arma::mat44 dmat44;	  /* .. and so on */				\
  typedef arma::cx_vec4 vc4;							\
  typedef arma::vec4 vd4;							\
  typedef arma::vec3 vd3;							\
  typedef arma::cx_mat22 cmat22;						\
  typedef arma::mat22 dmat22;							\
  typedef arma::cx_vec2 vc2;							\
  typedef arma::vec2 vd2;							


#endif // VTYPES_H









