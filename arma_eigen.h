// File: aram_eigen.h
// Author: J.A. de Jong
// Some functions to convert from Armadillo to Eigen and vice versa.
#pragma once
#ifndef _ARMA_EIGEN_H_
#define _ARMA_EIGEN_H_
#include "vtypes.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>

typedef Eigen::SparseMatrix<double> esdmat; // declares a column-major sparse matrix type of double
typedef Eigen::VectorXd evd;
typedef Eigen::MatrixXd edmat;


namespace math_common{
  SPOILNAMESPACE
  
  // These functions all make copies of the vectors. If you do not
  // want copies, we can use the view functions, as defined below
  esdmat ArmaToEigen(const sdmat& amat); // Conversion from Armadillo
  // sparse square double matrix to eigen sparse square double matrix
  evd ArmaToEigen(const vd& avd);		// Same for vectors
  // sdmat EigenToArma() // Not yet included
  vd EigenToArma(const evd& Evd);	// And back to Armadillo
  dmat EigenToArma(const edmat& emat);	// And back to Armadillo
  // Functions for creating "views" for the buffers
  dmat armaView(edmat EigenMat);	// 
  vd armaView(const evd& Eigenvec);

  // inline Eigen::Map<Eigen::VectorXd> eigenMap(double* data,us size) { return Eigen::Map<Eigen::VectorXd>(data,size); }

  // Insert data in a specific row of matrix Mat
  void insertInRowFromLeftSide(esdmat& Mat,const vd& data,us rownr);
  
  // Insert components at the topleft of the target matrix
  void insertSubMatrixInMatrixTopLeft(esdmat& target,esdmat& submat);

  // class anneTriplets{
  //   vd values;
  //   arma::Col<us> rows;
  //   arma::Col<us> cols;


  // };





  
} // namespace math_common


#endif /* _ARMA_EIGEN_H_ */


