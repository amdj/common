#include "arma_eigen.h"

namespace math_common{

  esdmat ArmaToEigen(const sdmat& armamat){ // Conversion from Armadillo
    TRACE(1,"ArmaToEigen(sdmat&)");
    const d* vals=armamat.values;
    us n_nonzero=armamat.n_nonzero;
    long long unsigned ncols=armamat.n_cols;
    long long unsigned nrows=armamat.n_rows;
    using arma::uword;
    arma::Col<uword> col_ptrs(armamat.col_ptrs,ncols);
    arma::Col<uword> row_indices(armamat.row_indices,n_nonzero);

    us r,c;			// Current row, column
    // TRACE(1,"n_nonzero:"<< n_nonzero);
    typedef Eigen::Triplet<d> triplet;
    
    vector<triplet> tr; tr.reserve(n_nonzero);
    // TRACE(1,"ncols:"<<ncols);
    // Data in the Armadillo sparse matrix is stored as a compressed
    // sparse column format. This means that a pointer is present
    // which points to the first nonzero value in column j as being
    // col_ptr[j]. However, this gives problems for empty columns. The
    // solution armadillo uses for this case is is that index of the
    // previous column is filled in on that place. This makes it all
    // quite complicated.
    // TRACE(-1,"colptrs:"<<arma::Col<us>(col_ptrs,ncols));

    arma::Col<us> col_indices(n_nonzero,fillwith::zeros);
    // TRACE(1,"col_indices created");
    us colptrcounter=0;
    // TRACE(1,"colptrs:"<< col_ptrs);
    for(us curcolumn=0;curcolumn<ncols;curcolumn++){
      // Check if we are not in the last column
      // TRACE(1,"Current column:"<<curcolumn);
      if(curcolumn<ncols-1){
	// First: check if next value of col_ptr is same, if so: no values in this column
	if(col_ptrs(colptrcounter)!=col_ptrs(colptrcounter+1)){
	  // OK, we can add values to this column
	  // TRACE(1,"")
	  for(us ind=col_ptrs(colptrcounter);ind<col_ptrs(colptrcounter+1);ind++){
	    // TRACE(1,"Index:"<<ind);
	    col_indices(ind)=curcolumn;
	  }
	}
      }
      else{			// Do the things for the last column
	// TRACE(-2,"Last column reached")
	for(us ind=col_ptrs(colptrcounter);ind<n_nonzero;ind++){
	  // TRACE(1,"Index:"<<ind);	    
	  col_indices(ind)=curcolumn;
	}
      }
      
      colptrcounter++;
      // TRACE(-1,"Curcolumn:"<<curcolumn);
    }
    // TRACE(-1,"Col indices:"<< col_indices);
    for(us n=0;n<n_nonzero;n++){
      r=row_indices(n);
      c=col_indices(n);
      TRACE(-5,"c:"<<c);
      tr.push_back(triplet(r,c,vals[n]));
    }
    // TRACE(-3,"Triplet making survived");
    esdmat result(nrows,ncols);
    TRACE(1,"ArmaToEigen: filling matrix..");
    result.setFromTriplets(tr.begin(),tr.end());
    // TRACE(-3,"Triplet setting survived");

    return result;
  }
  evd ArmaToEigen(const vd& avd){ // 
    //
    us size=avd.size();
    // Stupid hack to const-cast
    d* ptr=const_cast<d*>(avd.memptr());
    evd result(size);
    result=Eigen::Map<evd>(ptr,size);
    return result;
  }
  vd EigenToArma(const evd& Evd){
    us size=Evd.rows();
    
    vd result(&Evd[0],size);
    return result;
  }
  dmat EigenToArma(const edmat& eigmat){
    us rows=eigmat.rows();
    us cols=eigmat.cols();    
    return dmat(&eigmat(0,0),rows,cols);
  }

  dmat armaView(edmat Eigenmat)  {
    return dmat(Eigenmat.data(),Eigenmat.rows(),Eigenmat.cols(),false,false);
  }
  vd armaView(const evd& vec)  {
    d* data=const_cast<d*>(vec.data()); // Filthy hack
    return vd(data,vec.rows(),false,false);
  }
  
  void insertInRowFromLeftSide(esdmat& Mat,const vd& data,us rownr){
    TRACE(15,"InserInRowFromLeftSide");
    assert(data.size()<= (us) Mat.cols());
    assert(rownr< (us) Mat.rows());
    us cursize=Mat.nonZeros();
    us newsize=cursize+data.size(); // Allocates more than enough
    us datasize=data.size();
    Mat.reserve(newsize);
    for(us i=0;i<datasize;i++){
      if(data(i)!=0)
	Mat.insert(rownr,i)=data(i);
    } // for
  
  } // insertInRowFromLeftSide

  // void insertSubMatrixInMatrixTopLeft(esdmat& target,const esdmat& source) {
  //   TRACE(15,"insertSubMatrixInMatrixTopLeft()");
  //   assert(source.cols()<=target.cols());
  //   assert(source.rows()<=target.rows());
    
  //   TripletList tr=math_common::getTriplets(source);
  // }
  
  
  
} // namespace math_common









